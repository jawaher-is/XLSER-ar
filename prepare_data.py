"""
This file contains functions to process each dataset used accordingly.
Saves train, test, valid data splits in ./content/<modelname>/splits/
"""
import numpy as np
import pandas as pd

from pathlib import Path
from tqdm import tqdm

import torchaudio
from sklearn.model_selection import train_test_split

import os
# import sys


class CorpusDataFrame():
    """
    Used to organize and create a unified dataframe for all datasets.
    """
    def __init__(self):
        self.data = []
        self.exceptions = 0

    def append_file(self, path, name, label):
        # Append filename, filepath, and emotion label to the data list.
        try:
            # avoid broken files
            s = torchaudio.load(path)
            self.data.append({
                "name": name,
                "path": path,
                "emotion": label
            })
        except Exception as e:
            print('Could not load ', str(path), e)
            exceptions+=1
            pass

    def data_frame(self):
        if self.exceptions > 0: print(f'{exceptions} files could not be loaded')

        # Create the dataframe from the organized data list
        df = pd.DataFrame(self.data)
        return df


def KSUEmotions(data_path):
    print('PREPARING KSUEmotions DATA PATHS')

    cdf = CorpusDataFrame()

    # Iterate through all file paths.
    for path in tqdm(Path(data_path).glob("**/**/*.flac")):
        name = str(path).split('/')[-1].split('.')[0]
        label = int(name.split('E')[-1].split('P')[0])

        # Use common labels for each emotion for all datasets
        if label == 0:
            label = 'neutral'# 'Neutral'
        elif label == 1:
            label = 'happy' # 'Happiness'
        elif label == 2:
            label = 'sad' # Sadness'
        elif label == 3:
            label = 'surprise' #'Surprise'
        elif label == 4:
            label = 'questioning' # 'Questioning'
        elif label == 5:
            label = 'angry' # 'Anger'

        cdf.append_file(path, name, label)

    df = cdf.data_frame()

    return df


def RAVDESS(data_path):
    print('PREPARING RAVDESS DATA PATHS')

    cdf = CorpusDataFrame()

    for path in tqdm(Path(data_path).glob("**/*.wav")):
        name = str(path).split('/')[-1].split('.')[0]
        label = name.split('-')[2]

        cdf.append_file(path, name, label)

    df = cdf.data_frame()
    df.emotion.replace({'01':'neutral', '02':'calm', '03':'happy', '04':'sad', '05':'angry', '06':'fear', '07':'disgust', '08':'surprise'}, inplace=True)

    return df


def CREMA(data_path):
    print('PREPARING CREMA DATA PATHS')

    cdf = CorpusDataFrame()

    for path in tqdm(Path(data_path).glob("*.wav")):
        name = str(path).split('/')[-1].split('.')[0]
        label = name.split('_')[2]

        if label == 'SAD':
            label = 'sad'
        elif label == 'ANG':
            label = 'angry'
        elif label == 'DIS':
            label = 'disgust'
        elif label == 'FEA':
            label = 'fear'
        elif label == 'HAP':
            label = 'happy'
        elif label == 'NEU':
            label = 'neutral'
        else:
            label = 'unknown'
            print('Unknown label detected in ', path)

        cdf.append_file(path, name, label)

    df = cdf.data_frame()

    return df


def TESS(data_path):
    print('PREPARING TESS DATA PATHS')

    cdf = CorpusDataFrame()

    for path in tqdm(Path(data_path).glob("**/*.wav")):
        name = str(path).split('/')[-1].split('.')[0]
        label = name.split('_')[-1]
        if label =='ps': label = 'surprise'

        cdf.append_file(path, name, label)

    df = cdf.data_frame()
    return df


def SAVEE(data_path):
    print('PREPARING SAVEE DATA PATHS')

    cdf = CorpusDataFrame()

    for path in tqdm(Path(data_path).glob("*.wav")):
        name = str(path).split('/')[-1].split('.')[0]
        label = name.split('_')[1][:-2]

        if label == 'a':
            label = 'angry'
        elif label == 'd':
            label = 'disgust'
        elif label == 'f':
            label = 'fear'
        elif label == 'h':
            label = 'happy'
        elif label == 'n':
            label = 'neutral'
        elif label == 'sa':
            label = 'sad'
        elif label == 'su':
            label = 'surprise'
        else:
            label = 'unknown'
            print('Unknown label detected in ', path)

        cdf.append_file(path, name, label)

    df = cdf.data_frame()
    return df



def get_df(corpus, data_path, i=None):
    # Use the correct function to iterate through the named dataset.
    if corpus == 'ksuemotions':
        df = KSUEmotions(data_path)
    elif corpus == 'ravess':
        df = RAVDESS(data_path)
    elif corpus == 'crema':
        df = CREMA(data_path)
    elif corpus == 'tess':
        df = TESS(data_path)
    elif corpus == 'savee':
        df = SAVEE(data_path)
    return df


def df(config):
    '''
    This function uses the configurations file to get the datasets names and their file paths
    It creates a dataframe containing every filename, path, and emotion.
    '''

    # In case more than one dataset is used.
    if type(config['corpora']) == list:
        df = pd.DataFrame()
        for i, corpus in enumerate(config['corpora']):
            data_path = config['data_path'][i]
            df_ = get_df(corpus, data_path)

            df = pd.concat([df, df_], axis = 0)

    else:
        corpus = config['corpora']
        data_path = config['data_path']
        df = get_df(corpus, data_path)

    print(f"Step 0: {len(df)}")

    # Filter out non-existing files.
    df["status"] = df["path"].apply(lambda path: True if os.path.exists(path) else None)
    df = df.dropna(subset=["path"])
    df = df.drop("status", 1)
    print(f"Step 1: {len(df)}")

    df = df.sample(frac=1)
    df = df.reset_index(drop=True)

    # Explore the number of emotion lables in the dataset with what distribution.
    print("Labels: ", df["emotion"].unique())
    print()
    df.groupby("emotion").count()[["path"]]

    return df


def prepare_splits(df, config):
    output_dir = config['output_dir']
    save_path = output_dir + "/splits/"

    # Create splits directory
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    # Create train, test, and validation splits.
    random_state = config['seed'] # 101
    train_df, test_df = train_test_split(df, test_size=0.1, train_size=0.9, random_state=random_state, stratify=df["emotion"])
    train_df, valid_df = train_test_split(train_df, test_size=0.11, train_size=0.89, random_state=random_state, stratify=train_df["emotion"])

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    valid_df = valid_df.reset_index(drop=True)

    # Save each to file
    train_df.to_csv(f"{save_path}/train.csv", sep="\t", encoding="utf-8", index=False)
    test_df.to_csv(f"{save_path}/test.csv", sep="\t", encoding="utf-8", index=False)
    valid_df.to_csv(f"{save_path}/valid.csv", sep="\t", encoding="utf-8", index=False)

    print(train_df.shape)
    print(test_df.shape)
    print(valid_df.shape)


if __name__ == '__main__':
    import argparse
    import yaml

    # Get the configuration file containing dataset name, path, and other configurations
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='yaml configuration file path')
    args = parser.parse_args()
    config_file = args.config

    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    train_filepath = config['output_dir'] + "/splits/train.csv"
    test_filepath = config['output_dir'] + "/splits/test.csv"
    valid_filepath = config['output_dir'] + "/splits/valid.csv"

    # Create a dataframe
    df = df(config)

    # Create train, test, and validation splits and save them to file
    prepare_splits(df, config)
