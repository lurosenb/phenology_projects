from lstm_multiple_variables import BuffelLSTM as BuffelLSTMMulti 
from lstm_prcp_single import BuffelLSTM as BuffelLSTMSingle

import torch
from torch.utils.data import DataLoader, TensorDataset

import pandas as pd
import numpy as np

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from matplotlib_venn import venn2

from PIL import Image

import os

seed=816
torch.manual_seed(seed)

input_dim = 16
hidden_dim = 128
batch_size = 4

multi_model = BuffelLSTMMulti(input_dim, hidden_dim)
multi_model.load_state_dict(torch.load("../models/comb_best_model_lstm_multiple_variables.pth", map_location=torch.device('cpu')))

data_name = 'comb'
train_path = '../datasets/'+data_name+'-train.csv'
test_path = '../datasets/'+data_name+'-test.csv'
train_feature_path = '../datasets/'+data_name+'-train-features.npy'
test_feature_path = '../datasets/'+data_name+'-test-features.npy'
variable_path = '../datasets/variables.npy'

train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)
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

## torch datasets
test_dataset = TensorDataset(test_features, test_labels)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

train_dataset = TensorDataset(train_features, train_labels)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

model = multi_model

model.eval()
all_preds = []
all_labels = []
true_positives = []
true_negatives = []
false_positives = []
false_negatives = []

with torch.no_grad():
    for data, target in test_loader:
        output = model(data)
        preds = (output.squeeze() > 0.5).float()
        all_preds.extend(preds.numpy())
        all_labels.extend(target.numpy())

        correct_indices = (preds == target).nonzero(as_tuple=True)[0]
        for idx in correct_indices:
            if target[idx] == 1:
                true_positives.append((data[idx], target[idx]))
            else:
                true_negatives.append((data[idx], target[idx]))

        incorrect_indices = (preds != target).nonzero(as_tuple=True)[0]
        for idx in incorrect_indices:
            if target[idx] == 1:
                false_negatives.append((data[idx], target[idx]))
            else:
                false_positives.append((data[idx], target[idx]))

tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
acc = 100 * (tn + tp) / len(all_labels)
fp_rate = 100 * fp / len(all_labels)
fn_rate = 100 * fn / len(all_labels)

print(f'Accuracy: {acc}%')
print(f'FP: {fp_rate}%')
print(f'FN: {fn_rate}%')
print("True Positive Samples:", len(true_positives))
print("True Negative Samples:", len(true_negatives))
print("False Positive Sample:", len(false_positives))
print("False Negative Sample:", len(false_negatives))

mean_true_positives = np.array([t[0].numpy() for t in true_positives]).mean(axis=0).mean(axis=0)
mean_true_negatives = np.array([t[0].numpy() for t in true_negatives]).mean(axis=0).mean(axis=0)
mean_false_positives = np.array([t[0].numpy() for t in false_positives]).mean(axis=0).mean(axis=0)
mean_false_negatives = np.array([t[0].numpy() for t in false_negatives]).mean(axis=0).mean(axis=0)

print("True Positive Mean:", mean_true_positives)
print("True Negative Mean:", mean_true_negatives)
print("False Positive Mean:", mean_false_positives)
print("False Negative Mean:", mean_false_negatives)

df = pd.read_csv('vit-best-prediction-test.csv')

true_positives_vit = []
true_negatives_vit = []
false_positives_vit = []
false_negatives_vit = []

for index, row in df.iterrows():
    file_id = int(row['file_name'].split('.')[0])
    
    if row['labels'] == 1 and row['preds'] == 1:
        true_positives_vit.append(file_id)
    elif row['labels'] == 0 and row['preds'] == 0:
        true_negatives_vit.append(file_id)
    elif row['labels'] == 0 and row['preds'] == 1:
        false_positives_vit.append(file_id)
    elif row['labels'] == 1 and row['preds'] == 0:
        false_negatives_vit.append(file_id)

print("True Positives:", true_positives_vit)
print("True Negatives:", true_negatives_vit)
print("False Positives:", false_positives_vit)
print("False Negatives:", false_negatives_vit)

true_positives_id = []
true_negatives_id = []
false_positives_id = []
false_negatives_id = []

model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for batch_idx, (data, target) in enumerate(test_loader):
        output = model(data)

        preds = (output.squeeze() > 0.5).float()

        all_preds.extend(preds.numpy())
        all_labels.extend(target.numpy())

        start_idx = batch_idx * batch_size

        correct_indices = (preds == target).nonzero(as_tuple=True)[0]
        incorrect_indices = (preds != target).nonzero(as_tuple=True)[0]

        for idx in correct_indices:
            original_idx = start_idx + idx.item()
            observation_id = test_data.iloc[original_idx].Observation_ID

            if target[idx] == 1:
                true_positives_id.append(observation_id)
            else:
                true_negatives_id.append(observation_id)

        for idx in incorrect_indices:
            original_idx = start_idx + idx.item()
            observation_id = test_data.iloc[original_idx].Observation_ID

            if target[idx] == 1:
                false_negatives_id.append(observation_id)
            else:
                false_positives_id.append(observation_id)

print("True Positive IDs:", true_positives_id)
print("True Negative IDs:", true_negatives_id)
print("False Positive IDs:", false_positives_id)
print("False Negative IDs:", false_negatives_id)

true_positives_set1 = set(true_positives_id)
true_negatives_set1 = set(true_negatives_id)
false_positives_set1 = set(false_positives_id)
false_negatives_set1 = set(false_negatives_id)

true_positives_set2 = set(true_positives_vit)
true_negatives_set2 = set(true_negatives_vit)
false_positives_set2 = set(false_positives_vit)
false_negatives_set2 = set(false_negatives_vit)

plt.figure(figsize=(12, 12))

plt.subplot(2, 2, 1)
venn2([true_positives_set1, true_positives_set2], set_labels=('LSTM_multi', 'ViT'))
plt.title('True Positives')

plt.subplot(2, 2, 2)
venn2([true_negatives_set1, true_negatives_set2], set_labels=('LSTM_multi', 'ViT'))
plt.title('True Negatives')

plt.subplot(2, 2, 3)
venn2([false_positives_set1, false_positives_set2], set_labels=('LSTM_multi', 'ViT'))
plt.title('False Positives')

plt.subplot(2, 2, 4)
venn2([false_negatives_set1, false_negatives_set2], set_labels=('LSTM_multi', 'ViT'))
plt.title('False Negatives')


plt.show()

image_folder = 'all_observation_date_images'

def create_collage(ids, folder, output_filename):
    images = [Image.open(os.path.join(folder, f"{id}.png")) for id in ids]
    images = [image.resize((100, 100)) for image in images] 
    
    image_size = images[0].size
    collage_width = image_size[0] * min(len(images), 5)
    collage_height = image_size[1] * ((len(images) - 1) // 5 + 1) 
    
    collage = Image.new('RGB', (collage_width, collage_height))
    
    for index, image in enumerate(images):
        row = index // 5
        col = index % 5
        collage.paste(image, (col * image_size[0], row * image_size[1]))
    
    collage.save(output_filename)

unique_tp1 = true_positives_set1 - true_positives_set2
create_collage(unique_tp1, image_folder, "collages/unique_tp1.png")

unique_tp2 = true_positives_set2 - true_positives_set1
create_collage(unique_tp2, image_folder, "collages/unique_tp2.png")

unique_tn1 = true_negatives_set1 - true_negatives_set2
create_collage(unique_tn1, image_folder, "collages/unique_tn1.png")

unique_tn2 = true_negatives_set2 - true_negatives_set1
create_collage(unique_tn2, image_folder, "collages/unique_tn2.png")

unique_fp1 = false_positives_set1 - false_positives_set2
create_collage(unique_fp1, image_folder, "collages/unique_fp1.png")

unique_fp2 = false_positives_set2 - false_positives_set1
create_collage(unique_fp2, image_folder, "collages/unique_fp2.png")

unique_fn1 = false_negatives_set1 - false_negatives_set2
create_collage(unique_fn1, image_folder, "collages/unique_fn1.png")

unique_fn2 = false_negatives_set2 - false_negatives_set1
create_collage(unique_fn2, image_folder, "collages/unique_fn2.png")

create_collage(true_positives_set2, image_folder, "collages/tps_vit.png")



