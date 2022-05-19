import argparse
import yaml
import os

import prepare_data
import build_model

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


# Load test data
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


test_dataset = load_dataset("csv", data_files={"test": test_filepath}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


# Make a model trained on one dataset be evaluated on another: use the emotions present in the original dataset
