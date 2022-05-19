from torch import nn
from transformers import BertModel


class BertClassifier(nn.Module):
    def __init__(self, dropout=0.5):

        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.fc1 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(768, 32),
            nn.ReLU()
        )

    def forward(self, input_id, mask):
        # bert 這個 model 的 output 還要再看一下
        _, outs = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        outs = self.fc1(outs)

        return outs