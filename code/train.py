import pandas as pd
import torch
from tqdm import tqdm



class Model_Process():
    def __init__(self, model, config, args):
        self.model = model
        self.config = config
        self.criterion = config['Loss_function'].cuda()
        self.epoch = config['Epochs']
        self.optimizer = getattr(torch.optim, config['Optimizer'])(model.parameters(), **config['Optim_hparas'])
        self.acc = []
        self.loss = []
        self.max_acc = 0
        self.args = args
        


    def train(self, train_dataloader, valid_dataloader, device):

        for epoch_num in range(self.epoch):
            
            total_acc_train = 0 
            total_loss_train = 0

            for train_input in tqdm(train_dataloader):
                
                self.optimizer.zero_grad()
                train_label = train_input["labels"].squeeze(1).to(device)
                mask = train_input['masks'].squeeze(1).to(device)
                # seg = train_input[1].squeeze(1).to(device)
                input_id = train_input['tokens'].squeeze(1).to(device)

                output = self.model(input_id, mask)
                # output = output.to(torch.float32)
                # print(output)
                # print(train_label)
                
                batch_loss = self.criterion(output, train_label)
                total_loss_train += batch_loss.item()


                
                acc = (output.argmax(dim=1) == train_label.argmax(dim=1)).sum().item()
                # print(acc)
                
                
                total_acc_train += acc

                batch_loss.backward()

                self.optimizer.step()
            
            total_acc_valid = 0
            total_loss_valid = 0

            with torch.no_grad():
                for val_input in valid_dataloader:


                    val_label = val_input['labels'].squeeze(1).to(device)
                    mask = val_input['masks'].squeeze(1).to(device)
                    # seg = val_input[1].squeeze(1).to(device)
                    input_id = val_input['tokens'].squeeze(1).to(device)

                    output = self.model(input_id, mask)

                    batch_loss = self.criterion(output, val_label)
                    total_loss_valid += batch_loss.item()
                    
                    acc = (output.argmax(dim=1) == val_label.argmax(dim=1)).sum().item()
                    total_acc_valid += acc
            
            avg_train_acc = total_acc_train / len(train_dataloader)
            avg_train_los = total_loss_train / len(train_dataloader)
            avg_valid_acc = total_acc_valid / len(valid_dataloader)
            avg_valid_los = total_loss_valid / len(valid_dataloader)
            
            print(
                f'Epochs: {epoch_num + 1} | Train Loss: {avg_train_los: .3f} \
                | Train Accuracy: {avg_train_acc: .3f} \
                | Val Loss: {avg_valid_los: .3f} \
                | Val Accuracy: {avg_valid_acc: .3f}')
            
            if self.max_acc < avg_valid_acc:
                self.max_acc = avg_valid_acc
                torch.save(self.model.state_dict(), '{}classnlp_{}_{}'.format(self.args.path, self.args.vision, round(avg_valid_acc*100, 2)))

    def test(self, test_loader, device):
        ans = []
        df = pd.DataFrame()
        with torch.no_grad():
            for test_input in tqdm(test_loader):

                mask = test_input['masks'].to(device)
                input_id = test_input['tokens'].squeeze(1).to(device)

                output = self.model(input_id, mask)

                # print(output.argmax(dim=1).item())

                ans.append(output.argmax(dim=1).item())
                # break

        # print(ans)
        df['pred'] = ans
        df.to_csv('{}{}_out.csv'.format(self.args.resultpath, self.args.loadname))

    def Bitrain(self, train_dataloader, valid_dataloader, device):

        for epoch_num in range(self.epoch):
            
            total_acc_train = 0 
            total_loss_train = 0

            for train_input in tqdm(train_dataloader):
                
                self.optimizer.zero_grad()
                train_label = train_input["labels"].squeeze(1).to(device)
                mask = train_input['masks'].squeeze(1).to(device)
                mask2 = train_input['masks2'].squeeze(1).to(device)
                
                # seg = train_input[1].squeeze(1).to(device)
                input_id = train_input['tokens'].squeeze(1).to(device)
                input_id2 = train_input['tokens2'].squeeze(1).to(device)

                output = self.model(input_id, mask, input_id2, mask2, 'train')
                
                batch_loss = self.criterion(output, train_label)
                total_loss_train += batch_loss.item()

                
                # acc = (output.argmax(dim=1) == train_label).sum().item()
                acc = 0
                
                total_acc_train += acc

                batch_loss.backward()

                self.optimizer.step()
            
            total_acc_valid = 0
            total_loss_valid = 0

            with torch.no_grad():
                for val_input in valid_dataloader:


                    val_label = val_input['labels'].squeeze(1).to(device)
                    mask = val_input['masks'].squeeze(1).to(device)
                    # seg = val_input[1].squeeze(1).to(device)
                    input_id = val_input['tokens'].squeeze(1).to(device)

                    output = self.model(input_id, mask, None, None)

                    batch_loss = self.criterion(output, val_label)
                    total_loss_valid += batch_loss.item()
                    
                    acc = (output.argmax(dim=1) == val_label.argmax(dim=1)).sum().item()
                    total_acc_valid += acc
            
            avg_train_acc = total_acc_train / len(train_dataloader)
            avg_train_los = total_loss_train / len(train_dataloader)
            avg_valid_acc = total_acc_valid / len(valid_dataloader)
            avg_valid_los = total_loss_valid / len(valid_dataloader)
            
            print(
                f'Epochs: {epoch_num + 1} | Train Loss: {avg_train_los: .3f} \
                | Train Accuracy: {avg_train_acc: .3f} \
                | Val Loss: {avg_valid_los: .3f} \
                | Val Accuracy: {avg_valid_acc: .3f}')
            
            if self.max_acc < avg_valid_acc:
                self.max_acc = avg_valid_acc
                torch.save(self.model.state_dict(), '{}classnlp_{}_{}'.format(self.args.path, self.args.vision, round(avg_valid_acc*100, 2)))
        




                
            