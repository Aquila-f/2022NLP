import torch
from torch import nn
from transformers import AutoModel
from einops.layers.torch import Reduce



# gradient checkpoint
# test time augmentation 1. random mask 2. translate 3. replace


class BertClassifier(nn.Module):
    def __init__(self, model_name, dropout=0.5):

        super(BertClassifier, self).__init__()

        self.bert = AutoModel.from_pretrained(model_name)

        self.drop = nn.Dropout(dropout)
        
        self.fc1 = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(768),
            nn.Linear(768, 32),
            # nn.Dropout(),
            nn.LeakyReLU() # Tanh / Leakyrelu
        )
        

    def forward(self, input_id, mask):
        # with torch.no_grad():
        outs = self.bert(
            input_ids= input_id, 
            # token_type_ids=seg,
            attention_mask=mask,
            # return_dict=False
        )
        
        # print(outs)
        # print(outs[0].size())
        # print(outs[0].shape)
        # print(outs[1].shape)
        # print(outs[2].shape)
        # print(outs[3].shape)

        
        outs = self.drop(outs[0])
        outs = self.fc1(outs)
        

        return outs

# class BertClassifier(nn.Module):
#     def __init__(self, dropout=0.5):

#         super(BertClassifier, self).__init__()

#         self.bert = BertModel.from_pretrained('bert-base-cased')
#         self.fc1 = nn.Sequential(
#             nn.Dropout(dropout),
#             nn.Linear(768, 32),
#             nn.ReLU()
#         )

#     def forward(self, input_id, mask):
#         # bert 這個 model 的 output 還要再看一下
#         _, outs = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
#         outs = self.fc1(outs)

#         return outs

class DoubleBertClassifier(nn.Module):
    def __init__(self, dropout=0.5):

        super(BertClassifier, self).__init__()

        self.bert1 = AutoModel.from_pretrained('bert-base-cased')
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.fc1 = nn.Sequential(
            nn.Linear(768, 256),
            nn.Dropout(0.5),
            nn.Sigmoid()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(768, 256),
            nn.Dropout(dropout),
            nn.Sigmoid()
        )
        self.finalfc = nn.Sequential(
            nn.Linear(512, 32),
        )

    def forward(self, input_id1, mask1, input_id2, mask2):
        _, outs = self.bert1(input_ids= input_id1, attention_mask=mask1,return_dict=False)
        _, outs = self.bert2(input_ids= input_id2, attention_mask=mask2,return_dict=False)
        outs = torch.cat((outs, outs), 0)


        outs = self.drop(outs)
        outs = self.fc1(outs)
        outs = self.fc2(outs)

        return outs