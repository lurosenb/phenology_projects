
from lstm_multiple_variables import BuffelLSTM as BuffelLSTMMulti 
from lstm_prcp_single import BuffelLSTM as BuffelLSTMSingle

import torch
from torch.utils.data import DataLoader, TensorDataset

import pandas as pd
import numpy as np

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import os

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
    for data, target in train_loader:
        output = model(data)
        preds = (output.squeeze() > 0.5).float()
        if preds.dim() == 0:
            preds = preds.unsqueeze(0)
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

def z_score_normalize(sensitivity_scores):
    mean_val = np.mean(sensitivity_scores)
    std_val = np.std(sensitivity_scores)
    normalized_scores = (sensitivity_scores - mean_val) / std_val
    return normalized_scores

def sensitivity_analysis(class_samples):
    mean_feats = np.array([t[0].numpy() for t in class_samples]).mean(axis=0).mean(axis=0)
    sensitivities = []
    original_predictions = []
    for sample in class_samples:
        sequence_to_explain = sample[0]

        length, num_features = sequence_to_explain.shape

        sensitivity_scores = np.zeros(num_features)

        with torch.no_grad():
            seq = sequence_to_explain.unsqueeze(0)
            original_prediction = model(seq.float()).item()
            original_predictions.append(original_prediction)

        for feature_idx in range(num_features):
            perturbed_sequence = sequence_to_explain.clone()
            perturbed_sequence[:, feature_idx] = torch.tensor(mean_feats[feature_idx])
            
            with torch.no_grad():
                seq = perturbed_sequence.unsqueeze(0)
                perturbed_prediction = model(seq.float()).item()
            
            sensitivity_scores[feature_idx] = np.abs(original_prediction - perturbed_prediction)
        
        sensitivity_scores = z_score_normalize(sensitivity_scores)
        sensitivities.append(sensitivity_scores)
        
    return sensitivities, original_predictions

sensitivities_true_positives, predictions_true_positives = sensitivity_analysis(true_positives)
sensitivities_true_negatives, predictions_true_negatives = sensitivity_analysis(true_negatives)
sensitivities_false_positives, predictions_false_positives = sensitivity_analysis(false_positives)
sensitivities_false_negatives, predictions_false_negatives = sensitivity_analysis(false_negatives)

colors = {
    "true_positives": "blue",
    "true_negatives": "green",
    "false_positives": "red",
    "false_negatives": "orange",
}

f_names = ['total_precipitation_sum', 
    'temperature_2m', 'temperature_2m_min', 'temperature_2m_max',
    'soil_temperature_level_1', 'soil_temperature_level_1_min', 'soil_temperature_level_1_max',
    'volumetric_soil_water_layer_1', 'volumetric_soil_water_layer_1_min', 'volumetric_soil_water_layer_1_max',
    'surface_solar_radiation_downwards_sum', 'surface_solar_radiation_downwards_min', 'surface_solar_radiation_downwards_max',
    'surface_pressure', 'surface_pressure_min', 'surface_pressure_max'
]

fontsize=14

feature_names = {i:n for i,n in enumerate(f_names)}

output_dir = 'output_plots'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

num_features = 16
for feature_idx in range(num_features):
    plt.figure(figsize=(10,6))
    
    feature_name = feature_names.get(feature_idx, f"Feature {feature_idx+1}")
    
    plt.scatter(predictions_true_positives, [s[feature_idx] for s in sensitivities_true_positives], c=colors["true_positives"], label="True Positives")
    plt.scatter(predictions_true_negatives, [s[feature_idx] for s in sensitivities_true_negatives], c=colors["true_negatives"], label="True Negatives")
    plt.scatter(predictions_false_positives, [s[feature_idx] for s in sensitivities_false_positives], c=colors["false_positives"], label="False Positives")
    plt.scatter(predictions_false_negatives, [s[feature_idx] for s in sensitivities_false_negatives], c=colors["false_negatives"], label="False Negatives")

    plt.title(f'Sensitivity of {feature_name} vs Model Prediction', fontsize=fontsize)
    plt.xlabel('Model Prediction', fontsize=fontsize)
    plt.ylabel(f'Sensitivity of {feature_name}', fontsize=fontsize)

    plt.axvline(x=0.5, linestyle='--', color='grey', linewidth=1)
    
    plt.legend(fontsize=fontsize)

    output_file = os.path.join(output_dir, f'Sensitivity_of_{feature_name}_vs_Model_Prediction.pdf')
    plt.savefig(output_file, format='pdf')
    plt.show()

abbreviations = {
    'total_precipitation_sum': 'TP_Sum',
    'temperature_2m': 'Temp2m',
    'temperature_2m_min': 'Temp2m_Min',
    'temperature_2m_max': 'Temp2m_Max',
    'soil_temperature_level_1': 'SoilTemp_L1',
    'soil_temperature_level_1_min': 'SoilTemp_L1_Min',
    'soil_temperature_level_1_max': 'SoilTemp_L1_Max',
    'volumetric_soil_water_layer_1': 'VolSoilWater_L1',
    'volumetric_soil_water_layer_1_min': 'VolSoilWater_L1_Min',
    'volumetric_soil_water_layer_1_max': 'VolSoilWater_L1_Max',
    'surface_solar_radiation_downwards_sum': 'SSR_Down_Sum',
    'surface_solar_radiation_downwards_min': 'SSR_Down_Min',
    'surface_solar_radiation_downwards_max': 'SSR_Down_Max',
    'surface_pressure': 'Surf_Press',
    'surface_pressure_min': 'Surf_Press_Min',
    'surface_pressure_max': 'Surf_Press_Max'
}

feature_names_abbr = {i: abbreviations[n] for i, n in feature_names.items()}

sensitivity_values = {
    "true_positives": sensitivities_true_positives,
    "true_negatives": sensitivities_true_negatives,
    "false_positives": sensitivities_false_positives,
    "false_negatives": sensitivities_false_negatives,
}

box_data_sensitivity = []

num_features = len(feature_names)
for i in range(num_features):
    feature_sensitivities = []
    for label in sensitivity_values.keys():
        class_sensitivities = [sensitivity_array[i] for sensitivity_array in sensitivity_values[label]]
        feature_sensitivities.append(class_sensitivities)
    box_data_sensitivity.append(feature_sensitivities)

num_classes = len(sensitivity_values.keys())
group_width = num_classes + 1
positions = [i * group_width + j for i in range(num_features) for j in range(num_classes)]

flat_box_data = [item for sublist in box_data_sensitivity for item in sublist]

plt.figure(figsize=(15,4))
bp = plt.boxplot(flat_box_data, showfliers=False, vert=True, patch_artist=True, positions=positions,widths=0.8)

colors_list = list(colors.values())
for patch, color in zip(bp['boxes'], colors_list * num_features):
    patch.set_facecolor(color)

plt.ylabel('Sensitivity', fontsize=fontsize)
plt.title('Distribution of Sensitivity of Each Feature by Class Label', fontsize=fontsize)
plt.xticks([i * group_width + (group_width - 1) / 2 for i in range(num_features)], [feature_names_abbr[idx] for idx in range(num_features)], rotation=25, ha="right", fontsize=fontsize)

legend_elements = [Line2D([0], [0], color=c, lw=2, label=l) for l, c in colors.items()]
plt.legend(handles=legend_elements, loc='upper right', fontsize=fontsize)

plt.tight_layout()
plt.savefig('Sensitivity_Box_Plot_By_Class.pdf', format='pdf')
plt.show()

box_data_sensitivity = [[] for _ in range(num_features)]
for i in range(num_features):
    for label in sensitivity_values.keys():
        feature_sensitivities = [sensitivity_array[i] for sensitivity_array in sensitivity_values[label]]
        box_data_sensitivity[i].append(feature_sensitivities)

num_classes = len(sensitivity_values.keys())
group_width = num_classes + 1
positions = [[i * group_width + j for j in range(num_classes)] for i in range(num_features)]

tick_positions = [i * group_width + (group_width - 1) / 2 for i in range(num_features)]
tick_labels = [feature_names[idx] for idx in range(num_features)]

fontsize=14

plt.figure(figsize=(12,12))

colors_list = list(colors.values())

for i in range(num_features):
    bp = plt.boxplot(box_data_sensitivity[i], positions=positions[i], vert=False, patch_artist=True, showfliers=False)
    for patch, color in zip(bp['boxes'], colors_list):
        patch.set_facecolor(color)

plt.yticks(tick_positions, tick_labels, fontsize=fontsize)
plt.ylabel('Features', fontsize=fontsize)
plt.xlabel('Sensitivity', fontsize=fontsize)
plt.title('Distribution of Sensitivity of Each Feature by Class Label', fontsize=fontsize)

legend_elements = [Line2D([0], [0], color=c, lw=2, label=l) for l, c in colors.items()]
plt.legend(handles=legend_elements, loc='lower right', fontsize=fontsize)

plt.tight_layout()
plt.savefig('Sensitivity_Box_Plot_By_Class_Vertical.pdf', format='pdf')
plt.show()