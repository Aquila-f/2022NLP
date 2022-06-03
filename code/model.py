import torch
from torch import nn
from transformers import AutoModel
from einops.layers.torch import Reduce



# gradient checkpoint
# test time augmentation 1. random mask 2. translate 3. replace


class BertClassifier(nn.Module):
    def __init__(self, model_name, classifermethod=1, dropout=0.5):
        super(BertClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.classifer = classifermethod

        self.pureclassifer = nn.Sequential(
            nn.Dropout(dropout),
            nn.LayerNorm(768),
            nn.Linear(768,32),
            nn.LeakyReLU()
        )
        
        self.fc1 = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.Dropout(dropout),
            nn.LayerNorm(768),
            nn.Linear(768, 32),
            nn.LeakyReLU() # Tanh / Leakyrelu
        )
    def forward(self, input_id, mask):
        # with torch.no_grad():
        outs = self.bert(
            input_ids= input_id, 
            # token_type_ids=seg,
            attention_mask=mask,
            return_dict=False
        )
        # print(type(outs))
        # print(len(outs))

        if len(outs) == 1:
            outs = self.fc1(outs[0])  
        elif self.classifer == 1:
            outs = self.pureclassifer(outs[1])
        else:
            outs = self.fc1(outs[0])  
        
        return outs

class BiBertClassifier(nn.Module):
    def __init__(self, model_name, classifermethod=1, dropout=0.5):
        super(BiBertClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.classifer = classifermethod

        self.pureclassifer = nn.Sequential(
            nn.Dropout(dropout),
            nn.LayerNorm(768),
            nn.Linear(768,32),
            nn.Sigmoid()
        )
        self.r = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
        )
        
        self.fc1 = nn.Sequential(
            # Reduce('b n e -> b e', reduction='mean'),
            nn.Dropout(dropout),
            nn.LayerNorm(768),
            nn.Linear(768, 32),
            nn.LeakyReLU() # Tanh / Leakyrelu
        )
    def forward(self, input_id, mask, input_id2, mask2, mode = ''):
        # with torch.no_grad():
        outs = self.bert(
            input_ids= input_id, 
            # token_type_ids=seg,
            attention_mask=mask,
            # return_dict=True
        )
        
        if mode == 'train':
            outs2 = self.bert(
                input_ids= input_id2, 
                # token_type_ids=seg,
                attention_mask=mask2,
                # return_dict=True
            )
            outs2 = self.r(outs2[self.classifer])
        

        if self.classifer == 1:
            outs = outs[self.classifer]*0.5 + outs2[self.classifer]*0.5 if mode == 'train' else outs[self.classifer]
            outs = self.pureclassifer(outs)
        else:
            outs = self.r(outs[self.classifer])
            
            outs = outs*0.5 + outs2*0.5 if mode == 'train' else outs
            outs = self.fc1(outs)  
        
        return outs


class SpBertClassifier(nn.Module):
    def __init__(self, model_name, classifermethod=1, dropout=0.5):
        super(SpBertClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.classifer = classifermethod

        self.reduce1 = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
        )
        self.reduce2 = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
        )
        self.drop1 = nn.Dropout(dropout)
        # self.drop2 = nn.Dropout(dropout)

        self.pureclassifer = nn.Sequential(
            # nn.LayerNorm(768*2),
            nn.Linear(768*2, 768),
            nn.Linear(768*2, 32),
            nn.Softmax()
        )
        
        self.fc1 = nn.Sequential(
            # nn.LayerNorm(768*2),
            nn.Linear(768*2, 32),
            nn.Softmax() # Tanh / Leakyrelu
        )
    def forward(self, input_id, mask, input_id2, mask2):
        # with torch.no_grad():
        outs = self.bert(
            input_ids= input_id, 
            # token_type_ids=seg,
            attention_mask=mask,
            # return_dict=True
        )[self.classifer]

        outs2 = self.bert(
            input_ids= input_id2, 
            # token_type_ids=seg,
            attention_mask=mask2,
            # return_dict=True
        )[self.classifer]
        

        if self.classifer == 1:
            
            # outs2 = self.drop2(outs2)
            outs = torch.cat((outs, outs2), 1)
            outs = self.drop1(outs)
            outs = self.pureclassifer(outs)
        else:
            outs = self.reduce1(outs)
            # outs = self.drop1(outs)

            outs2 = self.reduce1(outs2)
            # outs2 = self.drop2(outs2)
            outs = torch.cat((outs, outs2), 1)
            outs = self.drop1(outs)

            outs = self.fc1(outs)
        
        return outs

# class BiBertClassifier(nn.Module):
#     def __init__(self, model_name, classifermethod=1, dropout=0.5):
#         super(BertClassifier, self).__init__()
#         self.bert = AutoModel.from_pretrained(model_name)
#         self.classifer = classifermethod
#         self.pureclassifer = nn.Sequential(
#             nn.Dropout(dropout),
#             nn.LayerNorm(768),
#             nn.Linear(768,32),
#             nn.LeakyReLU()
#         )
        
#         self.fc1 = nn.Sequential(
#             Reduce('b n e -> b e', reduction='mean'),
#             nn.Dropout(dropout),
#             nn.LayerNorm(768),
#             nn.Linear(768, 32),
#             nn.LeakyReLU() # Tanh / Leakyrelu
#         )
#     def forward(self, input_id, mask):
#         # with torch.no_grad():
#         outs = self.bert(
#             input_ids= input_id, 
#             # token_type_ids=seg,
#             attention_mask=mask,
#             # return_dict=True
#         )

#         if self.classifer == 1:
#             outs = self.pureclassifer(outs[1])
#         else:
#             outs = self.fc1(outs[0])        
        
#         return outs

