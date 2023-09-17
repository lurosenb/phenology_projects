import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from datasets import load_dataset
from transformers import ViTImageProcessor, ViTForImageClassification

seed=816
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

class BuffelMulti(nn.Module):
    def __init__(self, input_dim, hidden_dim, embed_dim, vit_model):
        super(BuffelMulti, self).__init__()
        self.vit_model = vit_model
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True)
        self.lr = nn.Linear(embed_dim, hidden_dim)
        self.hidden2out = nn.Linear(hidden_dim*2, 1)
        self.leaky = nn.LeakyReLU(0.1)
        
    def forward(self, x, img):
        ## lstm out
        lstm_out, _ = self.lstm(x)
        lstm_out = self.leaky(lstm_out)[:, -1, :]
        
        ## linear out
        vit_out = self.vit_model(img)
        vit_out = torch.mean(vit_out.last_hidden_state,dim=1)
        lr_out = self.lr(vit_out)
        lr_out = self.leaky(lr_out)

        ## concatenation
        concat = torch.concat([lstm_out, lr_out],dim=-1)
        output = self.hidden2out(concat)
        return output

def main():

    #-------------------------
    # arguments
    #-------------------------
    input_dim = 16
    lr = 2e-5
    batch_size = 2
    hidden_dim = 128
    embed_dim = 768
    num_epochs = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = 'google/vit-base-patch16-224-in21k'
    feature_extractor = ViTImageProcessor.from_pretrained(model_name)
    vit_model = ViTForImageClassification.from_pretrained(model_name,num_labels=2)
    vit_model = vit_model.vit
    for param in vit_model.parameters():
        param.requires_grad = False
        
    def transform(example_batch):
        inputs = feature_extractor([x for x in example_batch['image']], return_tensors='pt')
        return inputs

    # ----------------
    # load data
    # ----------------
    train_data_path = '../buffelgrass-onetime-train.csv'
    test_data_path = '../buffelgrass-onetime-test.csv'
    train_feature_path = '../buffelgrass-onetime-train.npy'
    test_feature_path = '../buffelgrass-onetime-test.npy'
    variable_path = '../buffelgrass-onetime-variables.npy'
    
    train_data = pd.read_csv(train_data_path)
    test_data = pd.read_csv(test_data_path)
    train_features = np.load(train_feature_path, allow_pickle=True).astype('float')
    test_features = np.load(test_feature_path, allow_pickle=True).astype('float')
    
    ## labels
    train_labels = torch.FloatTensor(train_data.Abundance_Binary.values)
    test_labels = torch.from_numpy(test_data.Abundance_Binary.values)
    
    ## precipitation feature
    train_features = torch.FloatTensor(train_features)
    test_features = torch.FloatTensor(test_features)
    
    ## normalization
    for i in range(train_features.size(-1)):
        mean, std = torch.mean(train_features[:,:,i]), torch.std(train_features[:,:,i])
        train_features[:,:,i] = ((train_features[:,:,i]-mean)/std)
        test_features[:,:,i] = (test_features[:,:,i]-mean)/std    
    
    # ----------------
    # 5-fold cv
    # ----------------    
    kf = KFold(n_splits=5)
    test_acc = []
    test_fp = []
    test_fn = []
    
    for i, (train_index, val_index) in enumerate(kf.split(train_features)):
        
        ## split
        print('---------------------')
        print(f'Split: {i+1}...')
        val_features = train_features[val_index]
        val_labels = train_labels[val_index]
        train_sub_features = train_features[train_index]
        train_sub_labels = train_labels[train_index]
            
        ## image data
        dataset = load_dataset("imagefolder", data_dir=f"planet-imgs-original/split{i+1}/")
        dataset = dataset.with_transform(transform)
        train_img_features = torch.concat([image['pixel_values'].unsqueeze_(0) for image in dataset['train']])
        val_img_features = torch.concat([image['pixel_values'].unsqueeze_(0) for image in dataset['validation']])
        test_img_features = torch.concat([image['pixel_values'].unsqueeze_(0) for image in dataset['test']])
    
        ## torch datasets
        train_dataset = TensorDataset(train_sub_features, train_img_features, train_sub_labels)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataset = TensorDataset(val_features, val_img_features, val_labels)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_dataset = TensorDataset(test_features, test_img_features, test_labels)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
        ## load model
        model = BuffelMulti(input_dim, hidden_dim, embed_dim, vit_model)
        model.to(device)
        pos_weight = torch.tensor([(1 / torch.cat([train_labels, test_labels]).float().mean()) + 0.1]) 
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        ## training
        lowest_loss = np.inf
        for epoch in tqdm(range(num_epochs)):
            model.train()
            for batch_idx, (data, img, target) in enumerate(train_loader):
                optimizer.zero_grad()
                output = model(data.to(device), img.to(device))
                loss = criterion(output.squeeze_(-1), target.to(device))
                loss.backward()
                optimizer.step()
        
            ## evaluation
            model.eval()
            all_preds = []
            all_labels = []
            with torch.no_grad():
                for data, img, target in val_loader:
                    output = model(data.to(device), img.to(device))
                    all_preds.extend(output.squeeze_(-1))
                    all_labels.extend(target.to(device))
            all_preds = torch.stack(all_preds)
            all_labels = torch.stack(all_labels)
            loss = criterion(all_preds, all_labels).item()  
            if loss<lowest_loss:
                lowest_loss = loss
                torch.save(model.state_dict(), "lowest_loss_model.pth")
                print(f"New lowest loss: {lowest_loss}. Model saved.")
        
        # load highest performing model
        best_model = BuffelMulti(input_dim, hidden_dim, embed_dim, vit_model)
        best_model.load_state_dict(torch.load("lowest_loss_model.pth"))    
        best_model.to(device)
        best_model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for data, img, target in test_loader:
                output = best_model(data.to(device), img.to(device))
                preds = (output.detach().cpu().squeeze_(-1) > 0.5).float()
                all_preds.extend(preds.numpy())
                all_labels.extend(target.numpy())
        
        tn, fp, fn, tp = confusion_matrix(all_preds, all_labels).ravel()
        acc = 100 * (tn + tp) / len(all_labels)
        fp_rate = 100 * fp / len(all_labels)
        fn_rate = 100 * fn / len(all_labels)
        
        print(f'Accuracy: {acc}%')
        print(f'FP: {fp_rate}%')
        print(f'FN: {fn_rate}%')
        test_acc.append(acc)
        test_fp.append(fp_rate)
        test_fn.append(fn_rate)
        
    np.save('results/multi-modal.npy', np.stack([test_acc, test_fp, test_fn]))
    print(f'Avg Acc: {np.mean(test_acc), np.std(test_acc)}')
    print(f'Avg FP: {np.mean(test_fp), np.std(test_fp)}')
    print(f'Avg FN: {np.mean(test_fn), np.std(test_fn)}')
    
if __name__ == "__main__":
    main()