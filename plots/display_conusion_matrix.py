import argsparse
import os
import yaml
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np




# Get the configuration file
parser = argparse.ArgumentParser()
parser.add_argument('--config', help='yaml configuration file path') # EXAMPLE content/config/test.yaml
args = parser.parse_args()
config_file = args.config

with open(config_file) as f:
    configuration = yaml.load(f, Loader=yaml.FullLoader)
print('Loaded configuration file: ', config_file)
print(configuration)

# get file
if ((configuration['test_corpora'] is not None) and (__name__ == '__main__')):
    file_name = (configuration['output_dir'].split('/')[-1]
            + '-evaluated-on-'
            + configuration['test_corpora']
            + '_clsf_report.csv')

    cm_file_name = (configuration['output_dir'].split('/')[-1]
            + '-evaluated-on-'
            + configuration['test_corpora']
            + '_conf_matrix.csv')
else:
    file_name = 'clsf_report.csv'
    cm_file_name = 'conf_matrix.csv'

cm_path = configuration['output_dir'] + '/results' + cm_file_name
cr_path = configuration['output_dir'] + '/results' + file_name

# read file as pd DataFrame
cm_df = pd.read_csv(cm_path, sep='\t', header=0, index_col=0)
labels = list(cm_df.columns.values)
cm_values = cm_df.to_numpy()
cm = ConfusionMatrixDisplay(cm_values, display_labels=labels)

# Dispaly conf_matrix
cm.plot()
plt.title('Confusion Matrix')
plt.savefig(results_path + '/' + cm_file_name.split('.')[0] + '.png')
plt.show()
plt.close()


#test
# cr_path = "/Users/jia2025/Desktop/XLSER-ar/evaluation_results/wav2vec2-xlsr-53-arabic-ksuemotions-finetuned-3ep_clsf_report.csv"
# Classification Report
cr_df = pd.read_csv(cr_path, sep='\t', header=0, index_col=0)
metrics = list(cr_df.columns.values)
metrics = metrics[:-1]
classes = list(cr_df.index.values)
cr_values = cr_df.to_numpy()
# remove the support column
cr_values = np.delete(cr_values, -1, axis=1)
#
cr_values = cr_values * 100
cr_values = cr_values.round(decimals=1)

# plot
fig, ax = plt.subplots()
# im = ax.imshow(cr_values)
plt.imshow(cr_values, interpolation = 'nearest' )

ax.set_xticks(np.arange(len(metrics)))
ax.set_yticks(np.arange(len(classes)))
ax.set_xticklabels(metrics)
ax.set_yticklabels(classes)

for i in range(len(classes)):
    for j in range(len(metrics)):
        text = ax.text(j, i, cr_values[i, j], ha="center", va="center", color="w")


plt.title('Classification Report')
plt.rcParams['figure.figsize'] = [6, 4]
# plt.figure(figsize=(3, 4))
# plt.xlabel('Metrics')
# plt.ylabel('Classes')
# plt.savefig(results_path + '/' + cr_file_name.split('.')[0] + '.png')
plt.show()
plt.close()
