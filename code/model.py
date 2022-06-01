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
            # return_dict=True
        )

        if self.classifer == 1:
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
            nn.LeakyReLU()
        )
        
        self.fc1 = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
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

        outs = outs[self.classifer]*0.5 + outs2[self.classifer]*0.5 if mode == 'train' else outs[self.classifer]
        

        if self.classifer == 1:
            outs = self.pureclassifer(outs)
        else:
            outs = self.fc1(outs)  
        
        return outs


class SpBertClassifier(nn.Module):
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
            # return_dict=True
        )

        if self.classifer == 1:
            outs = self.pureclassifer(outs[1])
        else:
            outs = self.fc1(outs[0])  
        
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

