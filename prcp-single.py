import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

def main():

    #-------------------------
    # arguments
    #-------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', required=True, help='filename of test data')
    args = parser.parse_args()    
    data_name = args.data_name
    
    #-------------------------
    # data
    #-------------------------
    test_path = '../datasets/'+data_name+'-test.csv'
    test_feature_path = '../datasets/'+data_name+'-test-features.npy'
    variable_path = '../datasets/variables.npy'
    test_data = pd.read_csv(test_path)
    test_features = np.load(test_feature_path, allow_pickle=True)
    variables = np.load(variable_path, allow_pickle=True)
    label = test_data.Abundance_Binary.values ## labels
    index = np.where(variables == 'total_precipitation_sum')[0][0]
    
    #-------------------------
    # prediction
    #-------------------------    
    pred = 1*(np.sum(test_features[:,:,index]*39.3701, axis=1)>1.7) # 1 meter = 39.3701 inches
    tn, fp, fn, tp = confusion_matrix(pred, label).ravel()            
    acc = 100*(tn+tp)/len(label)
    fp_rate = 100*(fp)/len(label)
    fn_rate = 100*(fn)/len(label)
    np.save(f'results/{data_name}-prcp-single.npy', np.array([acc, fp_rate, fn_rate]))
    
    print('-------------------')
    print(f'Accuracy: {acc}%')
    print(f'FP: {fp_rate}%')
    print(f'FN: {fn_rate}%')    
        
if __name__ == "__main__":
    main()