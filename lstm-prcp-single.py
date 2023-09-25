import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
import argparse
from sklearn.metrics import confusion_matrix, f1_score
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold

seed=816
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
        
class BuffelLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(BuffelLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True)
        self.hidden2out = nn.Linear(hidden_dim, 1)
        self.leaky = nn.LeakyReLU(0.1)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = self.leaky(lstm_out)
        output = self.hidden2out(lstm_out[:, -1, :])
        return output

def main():

    #-------------------------
    # hyperparameters
    #-------------------------
    input_dim = 1
    batch_size = 4
    hidden_dim = 128
    num_epochs = 500
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #-------------------------
    # arguments
    #-------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', required=True, help='filename of test data')
    parser.add_argument('--lr', required=True, help='filename of test data')
    args = parser.parse_args()    
    data_name = args.data_name
    lr = float(args.lr)
    
    # ----------------
    # load data
    # ----------------
    train_path = '../datasets/'+data_name+'-train.csv'
    test_path = '../datasets/'+data_name+'-test.csv'
    train_feature_path = '../datasets/'+data_name+'-train-features.npy'
    test_feature_path = '../datasets/'+data_name+'-test-features.npy'
    variable_path = '../datasets/variables.npy'
    
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    train_features = np.load(train_feature_path, allow_pickle=True).astype('float')
    test_features = np.load(test_feature_path, allow_pickle=True).astype('float')
    variables = np.load(variable_path, allow_pickle=True)
    
    ## labels
    train_labels = torch.FloatTensor(train_data.Abundance_Binary.values)
    test_labels = torch.from_numpy(test_data.Abundance_Binary.values)
    
    ## precipitation feature
    index = np.where(variables == 'total_precipitation_sum')[0][0]
    train_features = torch.FloatTensor(train_features[:,:,index][:,:,None])
    test_features = torch.FloatTensor(test_features[:,:,index][:,:,None])
    
    ## normalization
    mean, std = torch.mean(train_features), torch.std(train_features)
    train_features = ((train_features-mean)/std)
    test_features = (test_features-mean)/std
    
    ## torch datasets
    test_dataset = TensorDataset(test_features, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # ----------------
    # 5-fold cv
    # ----------------    
    kf = KFold(n_splits=5)
    test_acc = []
    test_f1 = []
    test_fp = []
    test_fn = []
    best_loss = np.inf
    
    for i, (train_index, val_index) in enumerate(kf.split(train_features)):
        
        ## split
        print('---------------------')
        print(f'Split: {i+1}...')
        val_features = train_features[val_index]
        val_labels = train_labels[val_index]
        train_sub_features = train_features[train_index]
        train_sub_labels = train_labels[train_index]
            
        ## torch datasets
        train_dataset = TensorDataset(train_sub_features, train_sub_labels)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataset = TensorDataset(val_features, val_labels)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        ## load model
        model = BuffelLSTM(input_dim, hidden_dim)
        model.to(device)
        pos_weight = torch.tensor([(1 / torch.cat([train_labels, test_labels]).float().mean()) + 0.1]) 
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        ## training
        lowest_loss = np.inf
        for epoch in tqdm(range(num_epochs)):
            model.train()
            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                output = model(data.to(device))
                loss = criterion(output.squeeze_(-1), target.to(device))
                loss.backward()
                optimizer.step()
        
            ## evaluation
            model.eval()
            all_preds = []
            all_labels = []
            with torch.no_grad():
                for data, target in val_loader:
                    output = model(data.to(device))
                    all_preds.extend(output.squeeze_(-1))
                    all_labels.extend(target.to(device))
            all_preds = torch.stack(all_preds)
            all_labels = torch.stack(all_labels)
            loss = criterion(all_preds, all_labels).item()   
            if loss<lowest_loss:
                lowest_loss = loss
                torch.save(model.state_dict(), "lowest_loss_model.pth")
                print(f"New lowest loss: {lowest_loss}. Model saved.")
        
        # save best model from cv
        if lowest_loss<best_loss:
            torch.save(model.state_dict(), f"{data_name}_best_model_lstm_prcp_single.pth")
        
        # load highest performing model
        best_model = BuffelLSTM(input_dim, hidden_dim)
        best_model.load_state_dict(torch.load("lowest_loss_model.pth"))    
        best_model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for data, target in test_loader:
                output = best_model(data)
                preds = (output.squeeze_(-1) > 0.5).float() 
                all_preds.extend(preds.numpy())
                all_labels.extend(target.numpy())
        
        tn, fp, fn, tp = confusion_matrix(all_preds, all_labels).ravel()
        f1 = f1_score(all_preds, all_labels)
        acc = 100 * (tn + tp) / len(all_labels)
        fp_rate = 100 * fp / len(all_labels)
        fn_rate = 100 * fn / len(all_labels)
        
        print(f'Accuracy: {acc}%')
        print(f'F1: {f1}%')
        print(f'FP: {fp_rate}%')
        print(f'FN: {fn_rate}%')
        test_acc.append(acc)
        test_f1.append(f1)
        test_fp.append(fp_rate)
        test_fn.append(fn_rate)
        
    np.save(f'results/{data_name}-lstm-prcp-single.npy', np.stack([test_acc, test_f1, test_fp, test_fn]))
    print(f'Avg Acc: {np.mean(test_acc), np.std(test_acc)}')
    print(f'Avg F1: {np.mean(test_f1), np.std(test_f1)}')
    print(f'Avg FP: {np.mean(test_fp), np.std(test_fp)}')
    print(f'Avg FN: {np.mean(test_fn), np.std(test_fn)}')
    
if __name__ == "__main__":
    main()