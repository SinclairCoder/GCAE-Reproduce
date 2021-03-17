from torchtext import data
from torch.utils.data import Dataset
import re
import os
import json

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()

def get_data_from_json(file_path):
        with open(file_path,'r') as load_f:
            dataset = json.load(load_f)
            return dataset

def load_semeval_data(text_field,aspect_field,sentiment_field,dataset_file):
    semeval_train = get_data_from_json(dataset_file['train'])
    semeval_test = get_data_from_json(dataset_file['test'])
    semeval_hard_test = get_data_from_json(dataset_file['hard_test'])
    print(len(semeval_train))
    print(len(semeval_test))
    print(len(semeval_hard_test))
    train_data = SemEval(text_field,aspect_field,sentiment_field,semeval_train)
    test_data = SemEval(text_field,aspect_field,sentiment_field,semeval_test)
    hard_test_data = SemEval(text_field,aspect_field,sentiment_field,semeval_hard_test)
    return train_data,test_data,hard_test_data


class SemEval(data.Dataset):

    @staticmethod
    def sort_key(ex):
        return len(ex.text)
    
    def __init__(self,text_field,aspect_field,sentiment_field,input_data,**kwargs):
        """ Create an SemEval Dataset instance given a path and fields.
        
        Arguments:
            text_field: The field that will be used for text data.
            aspect_field:  The field that will be used for aspect data.
            sentiment_field: The field that will be used for sentiment data.
            input_data: The examples contain all the data.
            Remaining keyword arguments: Passed to the constructor of data.Dataset.
        
        """
        text_field.preprocessing = data.Pipeline(clean_str)
        fields = [('text',text_field),('aspect',aspect_field),('sentiment',sentiment_field)]
        examples = []
        for e in input_data:
            if 'pp.' in e['sentence']:
                continue
            examples.append(data.Example.fromlist([e['sentence'],e['aspect'],e['sentiment']],fields))
        super(SemEval,self).__init__(examples,fields,**kwargs)
        