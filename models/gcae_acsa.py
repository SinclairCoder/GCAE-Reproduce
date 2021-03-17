import torch
import torch.nn as nn
class GCAE_ACSA(nn.Module):
    def __init__(self,args,text_field_weight,aspect_field_weight):
        super(GCAE_ACSA,self).__init__()

        self.args = args

        V = args.embed_num  # length of vocab
        D = args.embed_dim  # embedding dim
        C = args.polarities_dim 
        A = args.aspect_num  # length of aspect vocab

        Co = args.kernel_num
        Ks = [int(k) for k in args.kernel_sizes.split(',')]
        self.embed = nn.Embedding(V,D)
        # self.embed.weight = nn.Parameter(args.embedding,requires_grad=True)
        self.embed.weight.data.copy_(text_field_weight)

        self.aspect_embed = nn.Embedding(A,args.aspect_embed_dim)
        # self.aspect_embed.weight = nn.Parameter(args.aspect_embedding,requires_grad = True)
        self.aspect_embed.weight.data.copy_(aspect_field_weight)

        self.convs1 = nn.ModuleList([nn.Conv1d(D, Co, K) for K in Ks])  # in_channels,  out_channels, kernel_size
        self.convs2 = nn.ModuleList([nn.Conv1d(D, Co, K) for K in Ks]) 
        self.fc1 = nn.Linear(len(Ks)*Co, C)
        self.fc_aspect = nn.Linear(args.aspect_embed_dim, Co)

    def forward(self,feature,aspect):
        feature = self.embed(feature) # (len,batch_size,Dim)
        aspect_v = self.aspect_embed(aspect) # (batch_size,Dim)
        aspect_v = aspect_v.sum(1) / aspect_v.size(1)  # (batch_size,)

        x = [torch.tanh(conv(feature.transpose(1, 2))) for conv in self.convs1]  # [(N,Co,L), ...]*len(Ks)
        y = [torch.relu(conv(feature.transpose(1, 2)) + self.fc_aspect(aspect_v).unsqueeze(2)) for conv in self.convs2]

        x = [i*j for i, j in zip(x, y)]

        # pooling method
        x0 = [torch.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N,Co), ...]*len(Ks)
        x0 = [i.view(i.size(0), -1) for i in x0]

        x0 = torch.cat(x0, 1)
        logit = self.fc1(x0)  # (N,C)
        return logit, x, y


