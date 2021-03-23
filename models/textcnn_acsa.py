import torch
import torch.nn as nn


class TextCNN_ACSA(nn.Module):
    def __init__(self,args,text_field_weight,aspect_field_weight,text_pad_idx,aspect_pad_idx):
        super().__init__()
        self.args = args

        V = args.embed_num  # length of vocab
        D = args.embed_dim  # embedding dim
        C = args.polarities_dim 
        # A = args.aspect_num  # length of aspect vocab
        Co = args.kernel_num
        Ks = [int(k) for k in args.kernel_sizes.split(',')]
        
        self.embedding = nn.Embedding(V,D,padding_idx = text_pad_idx)
        self.embedding.weight.data.copy_(text_field_weight)
        self.embedding.weight.requires_grad = True

        self.convs = nn.ModuleList([nn.Conv1d(D, Co, K) for K in Ks])

        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(len(Ks)*Co, C)

    def forward(self,feature):

        feature = self.embedding(feature) # [batch,len,D]

        featured = feature.permute(0,2,1)

        convsed = [torch.relu(conv(featured)) for conv in self.convs]

        pooled = [torch.max_pooled(conved,conved.shape[2]).squeeze(2) for conved in convsed]

        cat = self.dropout(torch.cat(pooled,dim=1))

        out = self.fc(cat)

        return out

        


