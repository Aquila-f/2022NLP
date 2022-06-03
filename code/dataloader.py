import pandas as pd
import numpy as np
import torch
import torch.utils.data as data
from torch.nn.utils.rnn import pad_sequence
# from nltk.tokenize import word_tokenize 暫時用不到
from transformers import AutoTokenizer
import random


class Dataset_loader():
    def __init__(self, config, mode=""):
        self.dtokenizer = AutoTokenizer.from_pretrained(config['Model_name'])
        self.bsn = config['Batch_fn']
        self.batch_size = config['Batch_size']
        self.mini_batch_fn = [self.create_mini_batch, self.bi_create_mini_batch, self.sp_create_mini_batch]

        if mode == "test":
            self.df_test_pp = Dataset_Preprocessing(config['Root'], 'test', config['Drop_dup']).preprocessing()
            self.test_dataset = Dataset_Handler(self.df_test_pp, 'test', self.dtokenizer)

        else:
            print("> Data Preprocessing ...")
            self.df_train_pp = Dataset_Preprocessing(config['Root'], 'train', config['Drop_dup'], config['Datadroppercent']).preprocessing() 
            self.df_valid_pp = Dataset_Preprocessing(config['Root'], 'valid', config['Drop_dup'], config['Datadroppercent']).preprocessing()
            print("> Data tokenizing ...")

            if config['Batch_fn'] == 1:
                self.train_dataset = BiDataset_Handler(self.df_train_pp, 'train', self.dtokenizer, config['Random_maskn'])
            elif config['Batch_fn'] == 2:
                self.train_dataset = SpDataset_Handler(self.df_train_pp, 'train', self.dtokenizer, config['Random_maskn'])
            else:
                self.train_dataset = Dataset_Handler(self.df_train_pp, 'train', self.dtokenizer, config['Random_maskn'])

            if config['Batch_fn'] == 2:
                self.valid_dataset = SpDataset_Handler(self.df_valid_pp, 'valid', self.dtokenizer)
            else:
                self.valid_dataset = Dataset_Handler(self.df_valid_pp, 'valid', self.dtokenizer)

        

    def create_mini_batch(self, samples):
        tokens_tensors = [s['tokens'] for s in samples]
        
        # segments_tensors = [s[1] for s in samples]
        
        if samples[0]['labels'] is not None:
            label_ids = torch.stack([s['labels'] for s in samples])
        else:
            label_ids = None

        tokens_tensors = pad_sequence(tokens_tensors, batch_first=True)

        masks_tensors = torch.zeros(tokens_tensors.shape, dtype=torch.long)
        masks_tensors = masks_tensors.masked_fill(tokens_tensors != 0, 1)
    
        return {"tokens":tokens_tensors, "masks":masks_tensors, "labels":label_ids}

    def bi_create_mini_batch(self, samples):
        tokens_tensors = [s['tokens'] for s in samples]
        
        # segments_tensors = [s[1] for s in samples]
        
        if samples[0]['labels'] is not None:
            label_ids = torch.stack([s['labels'] for s in samples])
        else:
            label_ids = None

        tokens_tensors = pad_sequence(tokens_tensors, batch_first=True)

        masks_tensors = torch.zeros(tokens_tensors.shape, dtype=torch.long)
        masks_tensors = masks_tensors.masked_fill(tokens_tensors != 0, 1)

        if self.bsn == 1:
            tokens_tensors2 = [s['tokens2'] for s in samples]
            tokens_tensors2 = pad_sequence(tokens_tensors2, batch_first=True)
            masks_tensors2 = torch.zeros(tokens_tensors2.shape, dtype=torch.long)
            masks_tensors2 = masks_tensors2.masked_fill(tokens_tensors2 != 0, 1)
            return {"tokens":tokens_tensors, "masks":masks_tensors, "tokens2" : tokens_tensors2, "masks2":masks_tensors2 ,"labels":label_ids}
    
        return {"tokens":tokens_tensors, "masks":masks_tensors, "labels":label_ids}
    
    def sp_create_mini_batch(self, samples):
        
        tokens_a_tensors = [s['prompt'] for s in samples]
        tokens_b_tensors = [s['utterance'] for s in samples]
        
        # segments_tensors = [s[1] for s in samples]
        
        if samples[0]['labels'] is not None:
            label_ids = torch.stack([s['labels'] for s in samples])
        else:
            label_ids = None

        tokens_a_tensors = pad_sequence(tokens_a_tensors, batch_first=True)
        tokens_b_tensors = pad_sequence(tokens_b_tensors, batch_first=True)

        masks_a_tensors = torch.zeros(tokens_a_tensors.shape, dtype=torch.long)
        masks_a_tensors = masks_a_tensors.masked_fill(masks_a_tensors != 0, 1)

        masks_b_tensors = torch.zeros(tokens_b_tensors.shape, dtype=torch.long)
        masks_b_tensors = masks_b_tensors.masked_fill(masks_b_tensors != 0, 1)
    
        return {"tokens":tokens_a_tensors, "masks":masks_a_tensors, "tokens2":tokens_b_tensors, "masks2":masks_b_tensors, "labels":label_ids}

    # 必須要使用的 loader
    def get_loaders(self):
        train_dataloader = data.DataLoader(
            self.train_dataset,
            batch_size = self.batch_size,   # 太大你的 gpu mem 可能會爆炸 (但你 batchsize 如果大於 1 一定要用 padding > tokenize 那邊)
            shuffle = True,                 # shuffle
            num_workers = 2,                # 平行處理數量
            collate_fn = self.mini_batch_fn[self.bsn]
        )

        valid_dataloader = data.DataLoader(
            self.valid_dataset,
            batch_size = self.batch_size,
            shuffle = False,
            num_workers = 2,
            collate_fn = self.create_mini_batch if self.bsn != 2 else self.sp_create_mini_batch
        )

        return train_dataloader, valid_dataloader
    
    def get_test_loader(self):
        test_dataloader = data.DataLoader(
            self.test_dataset,
            batch_size = self.batch_size,
            shuffle = False,
            num_workers = 2,
            collate_fn = self.create_mini_batch if self.bsn != 2 else self.sp_create_mini_batch
        )
        return test_dataloader
    

class Dataset_Preprocessing():
    def __init__(self, root, mode, dropdup ,dropp=0.):
        self.path = '{}fixed_{}.csv'
        self.mode = mode
        self.dropdf = dropp
        self.dropdup = dropdup
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
                if self.mode == 'test':
                    for ss in range(s):
                        self.df.loc[tmp+ss+1, 'utterance'] = self.df.loc[tmp, 'utterance']
        
        if self.mode != 'test' and self.dropdup:
            column_names = ['conv_id']
            self.df.drop_duplicates(subset=column_names, keep='first', inplace=True)

        self.df['prompt'] = self.df['prompt'].str.replace("_comma_", ",")
        self.df['utterance'] = self.df['utterance'].str.replace("_comma_", ",")

        

        # print(len(self.df))
        if self.dropdf > 0:
            d = -len(self.df)*self.dropdf
            self.df = self.df.drop(self.df.index[int(d):])
        # print(len(self.df))

        return self.df
        # token set or something



class Dataset_Handler(data.Dataset):
    # 目前只考慮 prompt 與 label 之後要用 concate 可以從這裡下手
    def __init__(self, df, mode, dtokenizer, nmask=5):
        self.mode = mode
        self.df = df
        self.tokenizer = dtokenizer
        self.number_mask = nmask
        print("{} datas : {}".format(mode, len(self.df)))
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        text_a, text_b = self.df.iloc[idx, 2:4].values
        
        # print(type(self.df.iloc[idx, 4:5]))

        if self.mode == "test":
            label_tensor = None  
        else: 
            label_tensor = torch.tensor(self.transform(self.df.iloc[idx, 4:5].item()), dtype=torch.float)


        tokens_a = ["[CLS]"] + self.tokenizer.tokenize(text_a) + ["[SEP]"]
        # lena = len(tokens_a)

        tokens_b = self.tokenizer.tokenize(text_b) + ["[SEP]"]
        # lenb = len(tokens_b)

        merge_token = tokens_a + tokens_b

        # random delete if more than 512
        while len(merge_token) > 512:
            n = ''
            s = 0
            while n=='' or n=='[CLS]' or n=='[SEP]':
                s = random.randrange(len(merge_token))
                n = merge_token[s]
            merge_token.pop(s)
        

        # Argumentataion
        if self.mode == 'train' and self.number_mask > 0:
            # randmasknum = random.randrange(self.number_mask)+1
            rn = 0
            while rn < self.number_mask:
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
        return {"tokens":tokens_tensor, "labels":label_tensor} # segments_tensor
    def transform(self, value):
        rv = [0] * 32
        rv[value] = 1
        return [rv]





class BiDataset_Handler(data.Dataset):
    # 目前只考慮 prompt 與 label 之後要用 concate 可以從這裡下手
    def __init__(self, df, mode, dtokenizer, nmask):
        self.mode = mode
        self.df = df
        self.tokenizer = dtokenizer
        self.number_mask = nmask
        print("{} datas : {}".format(mode, len(self.df)))
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        randn = random.randrange(len(self.df))
        text_a, text_b = self.df.iloc[idx, 2:4].values

        text_c, text_d = self.df.iloc[randn, 2:4].values

        label_tensor = torch.tensor(self.transform(self.df.iloc[idx, 4:5].item()), dtype=torch.float)
        label_tensor2 = torch.tensor(self.transform(self.df.iloc[randn, 4:5].item()), dtype=torch.float)

        label_tensor = label_tensor * 0.5 + label_tensor2 * 0.5
        # print(label_tensor)

        tokens_a = ["[CLS]"] + self.tokenizer.tokenize(text_a) + ["[SEP]"]
        tokens_b = self.tokenizer.tokenize(text_b) + ["[SEP]"]
        merge_token1 = tokens_a + tokens_b
        tokens_c = ["[CLS]"] + self.tokenizer.tokenize(text_c) + ["[SEP]"]
        tokens_d = self.tokenizer.tokenize(text_d) + ["[SEP]"]
        merge_token2 = tokens_c + tokens_d


        # random delete if more than 512
        while len(merge_token1) > 512:
            n = ''
            s = 0
            while n=='' or n=='[CLS]' or n=='[SEP]':
                s = random.randrange(len(merge_token1))
                n = merge_token1[s]
            merge_token1.pop(s)

        while len(merge_token2) > 512:
            n = ''
            s = 0
            while n=='' or n=='[CLS]' or n=='[SEP]':
                s = random.randrange(len(merge_token2))
                n = merge_token2[s]
            merge_token2.pop(s)

        # Argumentataion
        if self.mode == 'train' and self.number_mask>0:
            # randmasknum = random.randrange(self.number_mask)+1
            rn = 0
            while rn < self.number_mask:
                n = ''
                s = 0
                while n=='' or n=='[CLS]' or n=='[SEP]':
                    s = random.randrange(len(merge_token1))
                    n = merge_token1[s]
                merge_token1[s] = '[MASK]'
                rn += 1

            # randmasknum = random.randrange(self.number_mask)+1
            rn = 0
            while rn < self.number_mask:
                n = ''
                s = 0
                while n=='' or n=='[CLS]' or n=='[SEP]':
                    s = random.randrange(len(merge_token2))
                    n = merge_token2[s]
                merge_token2[s] = '[MASK]'
                rn += 1
        # print(merge_token)

        idss = self.tokenizer.convert_tokens_to_ids(merge_token2)
        ids = self.tokenizer.convert_tokens_to_ids(merge_token1)
        tokens_tensor = torch.tensor(ids)
        tokens_tensor2 = torch.tensor(idss)

        return {"tokens":tokens_tensor, "tokens2":tokens_tensor2, "labels":label_tensor} # segments_tensor

    def transform(self, value):
        rv = [0] * 32
        rv[value] = 1
        return [rv]


class SpDataset_Handler(data.Dataset):
    # 目前只考慮 prompt 與 label 之後要用 concate 可以從這裡下手
    def __init__(self, df, mode, dtokenizer, nmask=5):
        self.mode = mode
        self.df = df
        self.tokenizer = dtokenizer
        self.number_mask = nmask
        print("{} datas : {}".format(mode, len(self.df)))
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        text_a, text_b = self.df.iloc[idx, 2:4].values
        
        # print(type(self.df.iloc[idx, 4:5]))

        if self.mode == "test":
            label_tensor = None  
        else: 
            label_tensor = torch.tensor(self.transform(self.df.iloc[idx, 4:5].item()), dtype=torch.float)
            

        tokens_a = ["[CLS]"] + self.tokenizer.tokenize(text_a) + ["[SEP]"]
        # lena = len(tokens_a)

        tokens_b = ["[CLS]"] + self.tokenizer.tokenize(text_b) + ["[SEP]"]
        # lenb = len(tokens_b)

        

        # random delete if more than 512
        while len(tokens_b) > 512:
            n = ''
            s = 0
            while n=='' or n=='[CLS]' or n=='[SEP]':
                s = random.randrange(len(tokens_b))
                n = tokens_b[s]
            tokens_b.pop(s)
        

        # Argumentataion
        if self.mode == 'train' and self.number_mask > 0:
            # randmasknum = random.randrange(self.number_mask)+1
            rn = 0
            while rn < self.number_mask:
                n = ''
                s = 0
                while n=='' or n=='[CLS]' or n=='[SEP]':
                    s = random.randrange(len(tokens_b))
                    n = tokens_b[s]
                tokens_b[s] = '[MASK]'
                rn += 1
        # print(merge_token)

        ids_a = self.tokenizer.convert_tokens_to_ids(tokens_a)
        tokens_a_tensor = torch.tensor(ids_a)

        ids_b = self.tokenizer.convert_tokens_to_ids(tokens_b)
        tokens_b_tensor = torch.tensor(ids_b)

            
        # segments_tensor = torch.tensor(lena*[0] + lenb*[1], dtype=torch.long)
        return {"prompt":tokens_a_tensor, "utterance":tokens_b_tensor, "labels":label_tensor} # segments_tensor
    def transform(self, value):
        rv = [0] * 32
        rv[value] = 1
        return [rv]