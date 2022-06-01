import pandas as pd
import numpy as np
import torch
import torch.utils.data as data
from torch.nn.utils.rnn import pad_sequence
# from nltk.tokenize import word_tokenize 暫時用不到
from transformers import AutoTokenizer
import random


class Dataset_loader():
    def __init__(self, root, batch_size, tokenizer_name, dropp=0.2):
        self.dtokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        print("> Data Preprocessing ...")
        self.df_train_pp = Dataset_Preprocessing(root, 'train', 1-dropp).preprocessing()
        self.df_valid_pp = Dataset_Preprocessing(root, 'valid', 1-dropp).preprocessing()
        print("> Data tokenizing ...")
        self.train_dataset = Dataset_Handler(self.df_train_pp, 'train', self.dtokenizer)
        self.valid_dataset = Dataset_Handler(self.df_valid_pp, 'Valid', self.dtokenizer)
        self.batch_size = batch_size

    def create_mini_batch(self, samples):
        # print(samples, flush=True)
        
        tokens_tensors = [s[0] for s in samples]
        # segments_tensors = [s[1] for s in samples]
        
        if samples[0][1] is not None:
            label_ids = torch.stack([s[1] for s in samples])
        else:
            label_ids = None

        tokens_tensors = pad_sequence(tokens_tensors, batch_first=True)
        # segments_tensors = pad_sequence(segments_tensors, batch_first=True)

        masks_tensors = torch.zeros(tokens_tensors.shape, dtype=torch.long)
        masks_tensors = masks_tensors.masked_fill(tokens_tensors != 0, 1)
        # segments_tensors

        return tokens_tensors, masks_tensors, label_ids

    # 必須要使用的 loader
    def get_loaders(self):
        train_dataloader = data.DataLoader(
            self.train_dataset,
            batch_size = self.batch_size,   # 太大你的 gpu mem 可能會爆炸 (但你 batchsize 如果大於 1 一定要用 padding > tokenize 那邊)
            shuffle = True,                 # shuffle
            num_workers = 2,                # 平行處理數量
            collate_fn = self.create_mini_batch
        )

        valid_dataloader = data.DataLoader(
            self.valid_dataset,
            batch_size = self.batch_size,
            shuffle = False,
            num_workers = 2,
            collate_fn = self.create_mini_batch
        )

        return train_dataloader, valid_dataloader
    

class Dataset_Preprocessing():
    def __init__(self, root, mode, dropp):
        self.path = '{}fixed_{}.csv'
        self.mode = mode
        self.dropdf = dropp
        self.df = pd.read_csv(self.path.format(root, self.mode))
        print("{} datas : {}".format(self.mode,len(self.df['conv_id'])))
        
        
        # self.df_test = pd.read_csv(self.path.format(self.root, 'test'))


    # 從這裡設計 文本資料的預先處理
    def preprocessing(self):

        ### v2 ###
        seglist = []
        i = 0
        while i < len(self.df):
            if self.df.loc[i, 'utterance_idx'] == 1:
                tmp = i
                i+=1
                s = 0
                while not self.df.loc[i, 'utterance_idx'] == 1 :
                    self.df.loc[tmp, 'utterance'] += " [SEP] " + self.df.loc[i, 'utterance'] 
                    i+=1
                    s+=1
                    if i > len(self.df)-1: break
                for ss in range(s):
                    self.df.loc[tmp+ss+1, 'utterance'] = self.df.loc[tmp, 'utterance']
        
        # column_names = ['conv_id']
        # self.df.drop_duplicates(subset=column_names, keep='first', inplace=True)
        self.df['prompt'] = self.df['prompt'].str.replace("_comma_", ",")
        self.df['utterance'] = self.df['utterance'].str.replace("_comma_", ",")
        # for sss in range(5):
        #     print(self.df.loc[sss, 'utterance'])

        # print(len(self.df))
        # d = -len(self.df)*self.dropdf
        # self.df = self.df.drop(self.df.index[int(d):])
        # print(len(self.df))

        return self.df
        # token set or something



class Dataset_Handler(data.Dataset):
    # 目前只考慮 prompt 與 label 之後要用 concate 可以從這裡下手
    def __init__(self, df, mode, dtokenizer):
        self.mode = mode
        self.df = df
        self.tokenizer = dtokenizer
        print("{} datas : {}".format(mode, len(self.df)))
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        text_a, text_b = self.df.iloc[idx, 2:4].values

        if self.mode == "test":
            label_tensor = None  
        else: 
            label_tensor = torch.tensor(self.df.iloc[idx, 4:5])

        tokens_a = ["[CLS]"] + self.tokenizer.tokenize(text_a) + ["[SEP]"]
        # lena = len(tokens_a)

        tokens_b = self.tokenizer.tokenize(text_b) + ["[SEP]"]
        # lenb = len(tokens_b)

        merge_token = tokens_a + tokens_b


        while len(merge_token) > 512:
            n = ''
            s = 0
            while n=='' or n=='[CLS]' or n=='[SEP]':
                s = random.randrange(len(merge_token))
                n = merge_token[s]
            merge_token.pop(s)
        

        if(self.mode == 'train'):
            randmasknum = random.randrange(3)+1
            rn = 0
            while rn < randmasknum:
                n = ''
                s = 0
                while n=='' or n=='[CLS]' or n=='[SEP]':
                    s = random.randrange(len(merge_token))
                    n = merge_token[s]
                merge_token[s] = '[MASK]'
                rn += 1
        # print(merge_token)

        ids = self.tokenizer.convert_tokens_to_ids(merge_token)
        tokens_tensor = torch.tensor(ids)

            
        # segments_tensor = torch.tensor(lena*[0] + lenb*[1], dtype=torch.long)

        return (tokens_tensor, label_tensor) # segments_tensor

############################### Test ###############################
class Test_loader():
    def __init__(self, root, batch_size):
        self.root = root
        self.path = '{}fixed_{}.csv'
        self.batch_size = batch_size
        self.df_test = pd.read_csv(self.path.format(self.root, 'test'))
        print("> Loading Datas from path {} ...".format(root))
        print("Test datas : {}".format(len(self.df_test['conv_id'])))
        self.preprocessing()


    def preprocessing(self):
        self.df_test['prompt'] = self.df_test['prompt'].str.replace("_comma_", ",")
        print("> Data Preprocessing ...")


    def get_loader(self):
        test_dataset = Test_Dataset_Handler(self.df_test)
        test_dataloader = data.DataLoader(
            test_dataset,
            batch_size = self.batch_size,
            shuffle = False,
            num_workers = 1
        )
        return test_dataloader


class Test_Dataset_Handler(data.Dataset):
    def __init__(self, df):
        self.texts = [tokenizer(text,
                                truncation=True, return_tensors="pt")
                      for text in df['prompt']
                     ]
    
    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        batch_text = self.texts[idx]

        return batch_text

# df_valid_pp = Dataset_Preprocessing('../dataset/', 'valid').preprocessing()
# # valid_dataset = Dataset_Handler(df_valid_pp, 'Valid')
# # print(valid_dataset)

# train_dataloader , valid_dataloader = Dataset_loader('../dataset/', 2).get_loaders()
# _, valid_dataloader = Dataset_loader('../dataset/', 2, 'distilbert-base-uncased-finetuned-sst-2-english').get_loaders()
# # # # print(len(train_dataloader))
# s = 0
# for txt in train_dataloader:
#     # print("--")
#     print(txt)
#     print(txt[0].squeeze(1))
#     print(txt[1])
#     print(txt[2])
#     # print(txt[3])
#     print(txt[0].shape[1])
#     break



# test_loader = Test_loader('../dataset/', 1).get_loader()

# for txt in test_loader:
#     print(txt)
#     break

