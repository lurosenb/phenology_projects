import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

def main():

    #-------------------------
    # data
    #-------------------------
    test_data_path = '../buffelgrass-onetime-test.csv'
    test_feature_path = '../buffelgrass-onetime-test.npy'
    variable_path = '../buffelgrass-onetime-variables.npy'
    test_data = pd.read_csv(test_data_path)
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
    np.save('results/prcp-single.npy', np.array([acc, fp_rate, fn_rate]))
    
    print('-------------------')
    print(f'Accuracy: {acc}%')
    print(f'FP: {fp_rate}%')
    print(f'FN: {fn_rate}%')    
        
if __name__ == "__main__":
    main()