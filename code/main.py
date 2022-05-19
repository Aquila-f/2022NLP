import torch
import argparse
from transformers import BertModel

from dataloader import Dataset_loader, Test_loader
from model import BertClassifier
from train import Model_Process

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.__version__, device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "None")
    parser.add_argument('--bert_model', '-bm', default='bert-base-cased', type=str)
    parser.add_argument('--mode', default="train", type=str)
    parser.add_argument('--loadname', '-ln', default="", type=str)
    parser.add_argument('--path', '-sp', default="../netweight/", type=str)
    parser.add_argument('--vision', default="v1", type=str)
    parser.add_argument('--resultpath', '-rp', default="../testresult/", type=str)
    # parser.add
    args = parser.parse_args()
    
    

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

    print(args.mode)
    print(args.path+args.loadname)
    # exit()
    
    test_dataloader = Test_loader(config['Root'], config['Batch_size']).get_loader()

    model = BertClassifier()
    model = model.cuda()
    model.load_state_dict(torch.load(args.path + args.loadname))
    model.eval()
    p = Model_Process(model, config, args)
    p.test(test_dataloader, device)
    
    exit()
    
    print(args)
    train_dataloader, valid_dataloader = Dataset_loader(config['Root'], config['Batch_size']).get_loaders()

    model = BertClassifier()
    model = model.cuda()

    p = Model_Process(model, config, args)
    p.train(train_dataloader, valid_dataloader, device)





    
    
