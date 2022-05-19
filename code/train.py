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
        self.vision = args.vision
        self.max_acc = 0


    def train(self, train_dataloader, valid_dataloader, device):

        for epoch_num in range(self.epoch):
            
            total_acc_train = 0 
            total_loss_train = 0

            for train_input, train_label in tqdm(train_dataloader):

                train_label = train_label.to(device)
                mask = train_input['attention_mask'].to(device)
                input_id = train_input['input_ids'].squeeze(1).to(device)

                output = self.model(input_id, mask)
    #                 print(output)
    #                 print(train_label)
                
                batch_loss = self.criterion(output, train_label)
                total_loss_train += batch_loss.item()
                
                acc = (output.argmax(dim=1) == train_label).sum().item()
                total_acc_train += acc

                self.model.zero_grad()
                batch_loss.backward()
                self.optimizer.step()
            
            total_acc_valid = 0
            total_loss_valid = 0

            with torch.no_grad():
                for val_input, val_label in valid_dataloader:

                    val_label = val_label.to(device)
                    mask = val_input['attention_mask'].to(device)
                    input_id = val_input['input_ids'].squeeze(1).to(device)

                    output = self.model(input_id, mask)

                    batch_loss = self.criterion(output, val_label)
                    total_loss_valid += batch_loss.item()
                    
                    acc = (output.argmax(dim=1) == val_label).sum().item()
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
                torch.save(self.model, 'netweight/classnlp_{}_{}'.format(self.vision, round(avg_valid_acc*100, 2)))
                
            