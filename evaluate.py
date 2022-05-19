import argparse
import yaml
import os

import prepare_data
from datasets import load_dataset

import pandas as pd
import torch
import torch.nn as nn
import torchaudio
import numpy as np
import librosa
from sklearn.metrics import classification_report
from transformers import AutoConfig, Wav2Vec2Processor

from nested_array_catcher import nested_array_catcher



# Get the configuration file
parser = argparse.ArgumentParser()
parser.add_argument('--config', help='yaml configuration file path') # EXAMPLE content/config/test.yaml
args = parser.parse_args()
config_file = args.config

with open(config_file) as f:
    configuration = yaml.load(f, Loader=yaml.FullLoader)


# Get test data
test_filepath = configuration['output_dir'] + "/splits/test.csv"

if not os.path.exists(test_filepath):
    df = prepare_data.df(configuration['corpora'], configuration['data_path'])
    prepare_data.prepare_splits(df, configuration)

if configuration['test_corpora'] is not None:
    """
    In case a different dataset is used to evaluate the model, this 'other'
    dataset is loaded, then the additional labels the model is not trained on
    are removed.
    """
    test_filepath_ = configuration['output_dir'] + "/evaluation-splits/test.csv"

    if not os.path.exists(test_filepath_):
        # The original dataset is first loaded to be used for label comparasion
        df = pd.read_csv(test_filepath, delimiter='\t')

        eval_df = prepare_data.df(configuration['test_corpora'], configuration['test_corpora_path'])
        eval_df = prepare_data.remove_additional_labels(df, eval_df)
        prepare_data.prepare_splits(eval_df, configuration, evaluation=True)

    test_filepath = test_filepath_

# Load test data
test_dataset = load_dataset("csv", data_files={"test": test_filepath}


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


# Load pretrained checkpoint, model configuration, and processor
processor_name_or_path = configuration['processor_name_or_path']
model_name_or_path = configuration['output_dir']

config = AutoConfig.from_pretrained(model_name_or_path)
processor = Wav2Vec2Processor.from_pretrained(processor_name_or_path) 
model = Wav2Vec2ForSpeechClassification.from_pretrained(model_name_or_path).to(device)


def speech_file_to_array_fn(batch):
    # Resample the audio files and extract features using the processor
    speech_array, sampling_rate = torchaudio.load(batch["path"])
    speech_array = speech_array.squeeze().numpy()
    speech_array = librosa.resample(np.asarray(speech_array), sampling_rate, processor.feature_extractor.sampling_rate)

    speech_array = nested_array_catcher(speech_array)

    batch["speech"] = speech_array
    return batch

def predict(batch):
    features = processor(batch["speech"], sampling_rate=processor.feature_extractor.sampling_rate, return_tensors="pt", padding=True)

    input_values = features.input_values.to(device)
    attention_mask = features.attention_mask.to(device)

    with torch.no_grad():
        logits = model(input_values, attention_mask=attention_mask).logits

    pred_ids = torch.argmax(logits, dim=-1).detach().cpu().numpy()
    batch["predicted"] = pred_ids
    return batch

test_dataset = test_dataset.map(speech_file_to_array_fn)

result = test_dataset.map(predict, batched=True, batch_size=8)

label_names = [config.id2label[i] for i in range(config.num_labels)]
label_names

y_true = [config.label2id[name] for name in result["emotion"]]
y_pred = result["predicted"]

print(y_true[:5])
print(y_pred[:5])

print(classification_report(y_true, y_pred, target_names=label_names))
