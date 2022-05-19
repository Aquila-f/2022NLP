import torch
import argparse
from transformers import BertModel

from dataloader import Dataset_loader
from model import BertClassifier
from train import Model_Process

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.__version__, device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "None")
    parser.add_argument('--bert_model', '-bm', default='bert-base-cased', type=str)
    parser.add_argument('--mode', default="train", type=str)
    parser.add_argument('--load', default="", type=str)
    parser.add_argument('--vision', default="v1", type=str)
    args = parser.parse_args()
    
    print(args.mode)
    print(args.load)
    # model = BertClassifier()
    model = torch.load(args.load)
    print(model)
    exit()

    config = {
        'Batch_size' : 1,
        'Epochs' : 10,
        'Optimizer' : 'Adam',
        'Optim_hparas':{
            'lr' : 1e-6,
            # 'momentum' : 0.9,
            # 'weight_decay' : 5e-4
        },
        'Loss_function' : torch.nn.CrossEntropyLoss(),
        'Bert_model' : 'bert-base-cased',
        'Root' : '../dataset/'
    }
    
    print(args)
    train_dataloader, valid_dataloader = Dataset_loader(config['Root'], 1).get_loader()

    model = BertClassifier()
    model = model.cuda()

    p = Model_Process(model, config, args)
    p.train(train_dataloader, valid_dataloader, device)





    
    
