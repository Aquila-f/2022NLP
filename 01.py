import pandas as pd

path = 'dataset/fixed_{}.csv'
df_train = pd.read_csv(path.format("train"))
df_valid = pd.read_csv(path.format("valid"))
df_test = pd.read_csv(path.format("test"))
# print(df_train.head())

column_names = ['conv_id']
df_train.drop_duplicates(subset=column_names, keep='first', inplace=True)
df_valid.drop_duplicates(subset=column_names, keep='first', inplace=True)

# df_train.head()

from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

# example_text = df_train['prompt'][0]
# print(example_text)
# bert_input = tokenizer(example_text,padding='max_length', max_length = 50, 
#                        truncation=True, return_tensors="pt")

# print(bert_input['input_ids'])
# print(bert_input['token_type_ids'])
# print(bert_input['attention_mask'])

import torch
import torch.utils.data as data
import numpy as np
from transformers import BertTokenizer

print(torch.cuda.device_count())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.__version__, device)

tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
labels = {
    'sad': 0, 
    'trusting': 1, 
    'terrified': 2, 
    'caring': 3, 
    'disappointed': 4,
    'faithful': 5,
    'joyful': 6,
    'jealous': 7,
    'disgusted': 8,
    'surprised': 9,
    'ashamed': 10,
    'afraid': 11,
    'impressed': 12,
    'sentimental': 13, 
    'devastated': 14,
    'excited': 15,
    'anticipating': 16,
    'annoyed': 17,
    'anxious': 18,
    'furious': 19,
    'content': 20,
    'lonely': 21,
    'angry': 22,
    'confident': 23,
    'apprehensive': 24,
    'guilty': 25,
    'embarrassed': 26,
    'grateful': 27,
    'hopeful': 28,
    'proud': 29,
    'prepared': 30,
    'nostalgic': 31
}

class Dataset(data.Dataset):
    def __init__(self, df):
        
        self.labels = [label for label in df['label']]
        self.texts = [tokenizer(text, 
                                truncation=True, return_tensors="pt")
                      for text in df['prompt']
                     ]
    def classes(self):
        return self.labels
    
    def __len__(self):
        return len(self.labels)
    
    def get_batch_labels(self, idx):
        return np.array(self.labels[idx])
    
    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):

        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

#         batch_texts = self.labels[idx]
#         batch_y = self.texts[idx]

        return batch_texts, batch_y

from torch import nn
from transformers import BertModel


class BertClassifier(nn.Module):

    def __init__(self, dropout=0.5):

        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 32)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):

        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)

        return final_layer

from torch.optim import Adam
from tqdm import tqdm

def train(model, train_data, val_data, learning_rate, epochs):

    train, val = Dataset(train_data), Dataset(val_data)

    train_dataloader = torch.utils.data.DataLoader(train, batch_size=1, shuffle=True, num_workers = 4)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=1, num_workers = 4)

    use_cuda = torch.cuda.is_available()
    # device = torch.device("cuda" if use_cuda else "cpu")

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr= learning_rate)

    if use_cuda:
            model = model.cuda()
            criterion = criterion.cuda()

    for epoch_num in range(epochs):

            total_acc_train = 0
            total_loss_train = 0

            for train_input, train_label in tqdm(train_dataloader):

                train_label = train_label.to(device)
                mask = train_input['attention_mask'].to(device)
                input_id = train_input['input_ids'].squeeze(1).to(device)

                output = model(input_id, mask)
                # print(output)
                # print(train_label)
                
                batch_loss = criterion(output, train_label)
                total_loss_train += batch_loss.item()
                
                acc = (output.argmax(dim=1) == train_label).sum().item()
                total_acc_train += acc

                model.zero_grad()
                batch_loss.backward()
                optimizer.step()
            
            total_acc_val = 0
            total_loss_val = 0

            with torch.no_grad():

                for val_input, val_label in val_dataloader:

                    val_label = val_label.to(device)
                    mask = val_input['attention_mask'].to(device)
                    input_id = val_input['input_ids'].squeeze(1).to(device)

                    output = model(input_id, mask)

                    batch_loss = criterion(output, val_label)
                    total_loss_val += batch_loss.item()
                    
                    acc = (output.argmax(dim=1) == val_label).sum().item()
                    total_acc_val += acc
            
            print(
                f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_data): .3f} \
                | Train Accuracy: {total_acc_train / len(train_data): .3f} \
                | Val Loss: {total_loss_val / len(val_data): .3f} \
                | Val Accuracy: {total_acc_val / len(val_data): .3f}')
                  
EPOCHS = 5
model = BertClassifier()
# model= nn.DataParallel(model)
model.to(device)
LR = 1e-6
              
train(model, df_train, df_valid, LR, EPOCHS)