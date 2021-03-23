import os
import argparse
import datetime
import torch
import torchtext.data as data
import json
from w2v import *
import math
from mydatasets import SemEval,load_semeval_data
from tensorboardX import SummaryWriter
from torchtext.vocab import Vectors
from torch.nn import init
import torch.nn as nn
from torchtext.vocab import GloVe
from models import GCAE_ACSA,GCAE_ATSA



class Instructor:
    def __init__(self,args):
        self.args = args
        self.text_field = data.Field(lower=True,tokenize='moses')
        if not args.aspect_phrase:
            self.aspect_field = data.Field(sequential=False)
        else:
            print('aspect phrase')
            self.aspect_field = data.Field(lower=True, tokenize='moses')
        self.sentiment_field = data.Field(sequential=False)

        self.train_data, self.test_data,self.hard_test_data = load_semeval_data(self.text_field,self.aspect_field,self.sentiment_field,args.dataset_file)
        self.train_iter,self.test_iter, self.hard_test_iter = data.Iterator.splits((self.train_data,self.test_data,self.hard_test_data),batch_sizes=(args.batch_size,len(self.test_data),len(self.hard_test_data)),device=self.args.device)


        self.vectors = Vectors(name='data/glove.6B.300d_test.txt')
        self.vectors.unk_init = torch.Tensor.uniform_(-0.25,0.25)  # follow the paper, but ...
        self.text_field.build_vocab(self.train_data,self.test_data,vectors=self.vectors,unk_init = torch.Tensor.normal_(-0.25,0.25))
        self.aspect_field.build_vocab(self.train_data,self.test_data,vectors=self.vectors)
        self.sentiment_field.build_vocab(self.train_data,self.test_data,vectors=self.vectors)

        self.text_pad_idx = text_field.vocab.stoi[text_field.pad_token]
        self.aspect_pad_idx = aspect_field.vocab.stoi[aspect_field.pad_token]

        args.polarities_dim = len(self.sentiment_field.vocab)-1  # remove conflict
        args.embed_num = len(self.text_field.vocab) 
        args.aspect_num = len(self.aspect_field.vocab)  

        # print("Loading GloVe pre-trained embedding...")
        # self.word_vectors = load_glove_embedding(self.text_field.vocab.itos, args.unif, args.embed_dim)
        # self.embedding = torch.from_numpy(np.asarray(self.embedding, dtype=np.float32))
        # print("Loading pre-trained aspect embedding...")
        # self.aspect_embedding = load_aspect_embedding_from_w2v(self.aspect_field.vocab.itos,self.text_field.vocab.stoi,self.word_vecs)
        # self.aspect_embedding = torch.from_numpy(np.asarray(self.aspect_embedding, dtype=np.float32))

        self.model = args.model_calss(args,self.text_field.vocab.vectors,self.aspect_field.vocab.vectors,text_pad_idx,aspect_pad_idx).to(args.device)
        if args.device.type == 'cuda': # 独显内存分配情况
            print("cuda memory allocated:", torch.cuda.memory_allocated(device=args.device.index))

    def _reset_params(self):
        for p in self.model.parameters():
            if p.requires_grad:
                if len(p.shape)>1:
                    self.args.initializer(p)
                else:
                    stdv = 1. / math.sqrt(p.shape[0])
                    torch.nn.init.uniform_(p, a=-stdv, b=stdv)
    def _train(self,criterion,optimizer,max_test_acc_overall=0):
        writer = SummaryWriter(log_dir='log')
        max_test_acc = 0
        max_hard_test_acc = 0
        global_step = 0  #  a training（ n epoches）
        for epoch in range(self.args.epochs):
            print('>'*50)
            print('epoch ',epoch)
            n_correct, n_total = 0,0
            for batch in self.train_iter:
                global_step += 1
                self.model.train()
                feature,aspect,target = batch.text,batch.aspect,batch.sentiment
                feature.t_() # transpose,  that  equals  batch_first = True 
                if len(feature)<2:
                    continue
                if not args.aspect_phrase:
                    aspect.unsqueeze_(0) # torch.Size([32] => torch.Size([1, 32]
                aspect.t_() # torch.Size([1, 32] => torch.Size([32, 1] ,that  equals  batch_first = True 
                target.sub_(1) # index align
                if self.args.device.type == 'cuda':
                    feature, aspect, target = feature.cuda(), aspect.cuda(), target.cuda()
                optimizer.zero_grad()
                logit = self.model(feature,aspect)
                loss = criterion(logit,target)
                loss.backward()
                optimizer.step()

                n_correct += (torch.argmax(logit, -1) == target).sum().item()
                n_total += len(logit)
                train_acc = n_correct/n_total
                test_acc = self._eval(self.test_iter,criterion)
                hard_test_acc = self._eval(self.hard_test_iter,criterion)
                if test_acc > max_test_acc:
                    max_test_acc = test_acc
                    max_hard_test_acc = hard_test_acc
                    # save model
                    if test_acc > max_test_acc_overall:
                        # linux environment  fix \\ => /
                        if not os.path.exists(os.getcwd()+'\\state_dict'):
                            os.mkdir('state_dict')
                        if self.args.atsa:
                            path = 'state_dict/{0}_{1}_acc{2}_{3}'.format(self.args.model,self.args.atsa_data,round(test_acc,4),round(hard_test_acc,4))
                        else:
                            path = 'state_dict/{0}_{1}_acc{2}_{3}'.format(self.args.model,self.args.acsa_data,round(test_acc,4),round(hard_test_acc,4))
                        torch.save(self.model.state_dict(),path)
                        print('>> saved: ' + path)
                    print('loss: {:.4f}, train_acc: {:.4f}, test_acc: {:.4f}, hard_test_acc: {:.4f}'.format(loss.item(), train_acc, test_acc, hard_test_acc))
                writer.add_scalar('loss',loss,global_step)
                writer.add_scalar('train_acc',train_acc,global_step)
                writer.add_scalar('test_acc',test_acc,global_step)
                writer.add_scalar('hard_test_acc',hard_test_acc,global_step)
        writer.close()

        return max_test_acc,max_hard_test_acc

        
    def _eval(self,data_iter,criterion):
        self.model.eval()
        n_test_correct, n_test_total = 0, 0
        loss = None
        with torch.no_grad():
            for batch in data_iter:
                feature,aspect,target = batch.text,batch.aspect,batch.sentiment
                feature.t_() # transpose,  that  equals  batch_first = True 
                if not args.aspect_phrase:
                    aspect.unsqueeze_(0) # torch.Size([32] => torch.Size([1, 32]
                aspect.t_() # torch.Size([1, 32] => torch.Size([32, 1] ,that  equals  batch_first = True 
                target.sub_(1) # index align
                if self.args.device.type == 'cuda':
                    feature, aspect, target = feature.cuda(), aspect.cuda(), target.cuda()
                logit,_,_ = self.model(feature,aspect)
                loss = criterion(logit,target)
                # loss = criterion(logit,target, size_average=False)
                n_test_correct += (torch.argmax(logit, -1) == target).sum().item()
                
                n_test_total += len(logit)
        eval_acc = n_test_correct / n_test_total
        
        return eval_acc
    
    def run(self,repeats=1):
        criterion = nn.CrossEntropyLoss()
        _params = filter(lambda p:p.requires_grad,self.model.parameters())

        optimizer = self.args.optimizer(_params,lr=self.args.lr,weight_decay=self.args.lr_decay)

        max_test_acc_overall,max_hard_test_acc_overall = 0,0
        max_test_acc_overall_ave, max_hard_test_acc_overall_ave = 0,0
        for i in range(repeats):
            print('repeat: ',i)
            self._reset_params()
            max_test_acc,max_hard_test_acc = self._train(criterion,optimizer,max_test_acc_overall=max_test_acc_overall)
            max_test_acc_overall_ave += max_test_acc
            max_hard_test_acc_overall_ave += max_hard_test_acc
            print('max_test_acc: {0} max_hard_test_acc: {1} '.format(max_test_acc,max_hard_test_acc))
            max_test_acc_overall = max(max_test_acc, max_test_acc_overall)
            max_hard_test_acc_overall = max(max_hard_test_acc, max_hard_test_acc_overall)
            print('#' * 100)
        max_test_acc_overall_ave /= repeats
        max_hard_test_acc_overall_ave /= repeats
        print('max_test_acc_overall: {0} max_hard_test_acc_overall: {1} '.format(max_test_acc_overall,max_hard_test_acc_overall))
        print('max_test_acc_overall_ave: {0} max_hard_test_acc_overall_ave: {1} '.format(max_test_acc_overall_ave,max_hard_test_acc_overall_ave))






if __name__ == '__main__':
    # Hyper Parameterss
    parser = argparse.ArgumentParser(description='CNN text classificer')
    # learning
    parser.add_argument('-lr', type=float, default=0.01, help='initial learning rate [default: 0.001]')
    parser.add_argument('-batch_size', type=int, default=32, help='batch size for training [default: 32]')
    parser.add_argument('-l2', type=float, default=0, help='initial learning rate [default: 0]')
    parser.add_argument('-lr_decay', type=float, default=0, help='learning rate decay')
    parser.add_argument('-momentum', type=float, default=0.99, help='SGD momentum [default: 0.99]')
    parser.add_argument('-epochs', type=int, default=8, help='number of epochs for train [default: 30]')
    parser.add_argument('-optimizer', default='adagrad', type=str)
    
    # logging
    parser.add_argument('-save_dir', type=str, default='snapshot', help='where to save the snapshot')
    parser.add_argument('-log_interval',  type=int, default=10,   help='how many steps to wait before logging training status [default: 10]')
    parser.add_argument('-save_interval', type=int, default=10000, help='how many steps to wait before saving [default:10000]')

    # data 
    parser.add_argument('-acsa_data', type=str, default='2014', help='select the restaurant dataset for ACSA task, "2014" or "large" ')
    parser.add_argument('-atsa_data', type=str, default='rest', help='select the restaurant dataset for ATSA task, "laptop" or "rest" ')
    
    parser.add_argument('-atsa', action='store_true', default=False)
    parser.add_argument('-polarities_dim', default=5, type=int)
    parser.add_argument('-embed_dim', type=int, default=300, help='number of embedding dimension [default: 300]')
    parser.add_argument('-aspect_embed_dim', type=int, default=300, help='number of aspect embedding dimension [default: 300]')
    parser.add_argument('-unif', type=float, help='Initializer bounds for embeddings', default=0.25)
    parser.add_argument('-embed_file', default='glove', help='w2v or glove')
    parser.add_argument('-aspect_phrase', action='store_true', default=False)
        # 为使每一层输出的方差尽可能相等，将参数在某个区间内平均初始化
    parser.add_argument('-initializer', default='xavier_uniform_', type=str)

    # model CNNs
    parser.add_argument('-model', type=str, default='gcae_acsa', help='Model name [default: gcae_acsa]')
    parser.add_argument('-kernel_sizes', type=str, default='3,4,5', help='comma-separated kernel size to use for convolution')
    parser.add_argument('-kernel_num', type=int, default=100, help='number of each kind of kernel [default: 100]')

    # device
    parser.add_argument('-no_cuda', action='store_true', default=False, help='disable the gpu')

    # option
    parser.add_argument('-trials', type=int, default=1, help='the number of trials')
    parser.add_argument('-verbose', type=int, default=0)
    parser.add_argument('-test', action='store_true', default=False, help='train or test')

    args = parser.parse_args()


    ds_atsa_laptop = {'train': 'data/atsa-laptop/atsa_train.json','test': 'data/atsa-laptop/atsa_test.json','hard_test':'data/atsa-laptop/atsa_hard_test.json'}
    ds_atsa_restaurant = {'train': 'data/atsa-restaurant/atsa_train.json','test': 'data/atsa-restaurant/atsa_test.json','hard_test':'data/atsa-restaurant/atsa_hard_test.json'}
    ds_atsa = {'laptop':ds_atsa_laptop,'rest':ds_atsa_restaurant}

    ds_acsa_large = {'train': 'data/acsa-restaurant-large/acsa_train.json','test':'data/acsa-restaurant-large/acsa_test.json','hard_test':'data/acsa-restaurant-large/acsa_hard_test.json'}
    ds_acsa_2014 = {'train': 'data/acsa-restaurant-2014/acsa_train.json','test':'data/acsa-restaurant-2014/acsa_test.json','hard_test':'data/acsa-restaurant-2014/acsa_hard_test.json'}
    ds_acsa = {'large':ds_acsa_large,'2014':ds_acsa_2014}

    ds_files ={'atsa':ds_atsa,'acsa':ds_acsa}


    model_classes = { 
        'gcae_atsa' : GCAE_ATSA,
        'gcae_acsa': GCAE_ACSA
    }

    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal,
        'orthogonal_': torch.nn.init.orthogonal_
        
    }

    optimizers = {
        'adadelta': torch.optim.Adadelta,  # default lr=1.0
        'adagrad': torch.optim.Adagrad,  # default lr=0.01
        'adam': torch.optim.Adam,  # default lr=0.001
        'adamax': torch.optim.Adamax,  # default lr=0.002
        'asgd': torch.optim.ASGD,  # default lr=0.01
        'rmsprop': torch.optim.RMSprop,  # default lr=0.01
        'sgd': torch.optim.SGD,
    }

    args.model_calss = model_classes[args.model]
    if args.atsa:
        args.dataset_file = ds_files['atsa'][args.atsa_data]
    else:
        args.dataset_file = ds_files['acsa'][args.acsa_data]
    args.optimizer = optimizers[args.optimizer]
    args.initializer = initializers[args.initializer]
    torch.backends.cudnn.deterministic = True
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ins = Instructor(args)
    ins.run(5)