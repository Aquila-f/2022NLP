import pandas as pd
import numpy as np
import torch.utils.data as data
# from nltk.tokenize import word_tokenize 戰實用步道
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

class Dataset_loader():
    def __init__(self, root, batch_size):
        self.df_train_pp, self.df_valid_pp = Dataset_Preprocessing(root).preprocessing()
        self.train_dataset = Dataset_Handler(self.df_train_pp, 'train')
        self.valid_dataset = Dataset_Handler(self.df_valid_pp, 'valid')
        self.batch_size = batch_size

    # 必須要使用的 loader
    def get_loader(self):
        train_dataloader = data.DataLoader(
            self.train_dataset,
            batch_size = self.batch_size,   # 太大你的 gpu mem 可能會爆炸 (但你 batchsize 如果大於 1 一定要用 padding > tokenize 那邊)
            shuffle = True,                 # shuffle
            num_workers = 1                 # 平行處理數量
        )

        valid_dataloader = data.DataLoader(
            self.valid_dataset,
            batch_size = self.batch_size,
            shuffle = False,
            num_workers = 1
        )

        return train_dataloader, valid_dataloader


class Dataset_Preprocessing():
    def __init__(self, root):
        self.root = root
        self.path = '{}fixed_{}.csv'
        self.df_train = pd.read_csv(self.path.format(self.root, 'train'))
        self.df_valid = pd.read_csv(self.path.format(self.root, 'valid'))
        print("> Loading Datas from path {} ...".format(root))
        print("Train datas : {}".format(len(self.df_train['conv_id'])))
        print("Valid datas : {}".format(len(self.df_valid['conv_id'])))
        
        
        # self.df_test = pd.read_csv(self.path.format(self.root, 'test'))


    # 從這裡設計 文本資料的預先處理
    def preprocessing(self):
        
        column_names = ['conv_id']
        self.df_train.drop_duplicates(subset=column_names, keep='first', inplace=True)
        self.df_valid.drop_duplicates(subset=column_names, keep='first', inplace=True)

        self.df_train['prompt'].replace("_comma_", ",")
        self.df_valid['prompt'].replace("_comma_", ",")
        print("> Data Preprocessing ...")


        return self.df_train, self.df_valid

        # token set or something



class Dataset_Handler(data.Dataset):
    # 目前只考慮 prompt 與 label 之後要用 concate 可以從這裡下手
    def __init__(self, df, mode):
        self.labels = [label for label in df['label']]
        self.texts = [tokenizer(text, # padding='max_length', max_length=512, # 不用padding 可以簡少訓練時間 tensor較小 (但你 batchsize 如果大於 1 一定要用 padding)
                                truncation=True, return_tensors="pt")
                      for text in df['prompt']
                     ]
        print("{} datas : {}".format(mode, len(self.labels)))
    
    def __len__(self):
        return len(self.labels)
    

    def __getitem__(self, idx):

        batch_texts = self.texts[idx]
        batch_y = np.array(self.labels[idx])

#         batch_texts = self.labels[idx]
#         batch_y = self.texts[idx]

        return batch_texts, batch_y


# train_dataloader, valid_dataloader = Dataset_loader('../dataset/', 1).get_loader()

# for txt in train_dataloader:
#     print(txt)
#     break




