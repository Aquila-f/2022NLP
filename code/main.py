from numpy import float64
import torch
from torch import nn
import argparse

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
    parser.add_argument('--vision', '-v', default="v1", type=str)
    parser.add_argument('--resultpath', '-rp', default="../testresult/", type=str)
    parser.add_argument('--message', '-m', default="", type=str)
    parser.add_argument('--cudadev', '-cd', default=1, type=int)
    parser.add_argument('--learningrate', '-lr', default=1e-6, type=float)
    parser.add_argument('--epoch', '-ep', default=10, type=int)
    parser.add_argument('--batchsize', '-bs', default=1, type=int)
    parser.add_argument('--datapercent', '-dps', default=0.2, type=float)
    # parser.add
    args = parser.parse_args()
    
    

    config = {
        'Batch_size' : args.batchsize,
        'Epochs' : args.epoch,
        'Optimizer' : 'Adam',
        'Optim_hparas':{
            'lr' : args.learningrate,
            # 'momentum' : 0.9,
            # 'weight_decay' : 5e-4
        },
        'Loss_function' : torch.nn.CrossEntropyLoss(),
        # 'Model_name' : 'distilbert-base-uncased-finetuned-sst-2-english',
        'Model_name' : args.bert_model,
        'Root' : '../dataset/',
        'Datapercent' : args.datapercent,
        'Device' : [0,1]

    }
    print(config)

    print("mode : {}".format(args.mode))
    torch.cuda.set_device(args.cudadev)


    model = BertClassifier(config['Model_name'])
    # if torch.cuda.device_count() > 1:
    #     print(torch.cuda.device_count(), "GPUs!")
    #     model = nn.DataParallel(model, device_ids=config['Device'])
    
    model = model.cuda()
    if args.loadname != "": model.load_state_dict(torch.load(args.path + args.loadname))

    if args.mode == "test":
        
        print("result csv save in : {}{}".format(args.path, args.loadname))
        
        test_dataloader = Test_loader(config['Root'], config['Batch_size'], config['Model_name']).get_loader()

        # model.load_state_dict(torch.load(args.path + args.loadname))
        model.eval()

        p = Model_Process(model, config, args)
        p.test(test_dataloader, device)
        
    else:
        train_dataloader, valid_dataloader = Dataset_loader(config['Root'], config['Batch_size'], config['Model_name'], config['Datapercent']).get_loaders()

        model.train()
        p = Model_Process(model, config, args)
        p.train(train_dataloader, valid_dataloader, device)
    
    print(args)
    print(config)




    
    
