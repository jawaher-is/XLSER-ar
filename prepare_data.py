"""
This file contains functions to process each dataset used accordingly.
"""

import numpy as np
import pandas as pd

from pathlib import Path
from tqdm import tqdm

import torchaudio
from sklearn.model_selection import train_test_split

import os
# import sys
import argparse
import yaml

class PrepareDataPaths():
    def __init__(self):
        self.data = []
        self.exceptions = 0

    def append_file(self, path, name, label):
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
        df = pd.DataFrame(self.data)
        return df

# class KSUTest(PrepareDataPaths):
# data_path = './content/data/ksu_emotions/data/SPEECH/'
def KSUTest(data_path):
    print('TESTING CLASS')

    Class = PrepareDataPaths()

    # def __init__(self, data_path):
    #     super().__init__(data_path)

    for path in tqdm(Path(data_path).glob("**/**/*.flac")): # self.data_path
        name = str(path).split('/')[-1].split('.')[0]
        label = int(name.split('E')[-1].split('P')[0])

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

        Class.append_file(path, name, label)

    df = Class.data_frame()

    return df

df = KSUTest('./content/data/ksu_emotions/data/SPEECH/')

def KSUEmotions(data_path):
    print('PREPARING KSUEmotions DATA PATHS')

    data = []
    exceptions = 0

    for path in tqdm(Path(data_path).glob("**/**/*.flac")):
        name = str(path).split('/')[-1].split('.')[0]
        label = int(name.split('E')[-1].split('P')[0])

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

        try:
            # avoid broken files
            s = torchaudio.load(path)
            data.append({
                "name": name,
                "path": path,
                "emotion": label
            })
        except Exception as e:
            print('Could not load ', str(path), e)
            exceptions+=1
            pass

    if exceptions > 0: print(f'{exceptions} files could not be loaded')

    df = pd.DataFrame(data)

    return df


def RAVDESS(data_path):
    print('PREPARING RAVDESS DATA PATHS')

    data = []
    exceptions = 0

    for path in tqdm(Path(data_path).glob("**/*.wav")):
        name = str(path).split('/')[-1].split('.')[0]
        label = name.split('-')[2]

        try:
            # avoid broken files
            s = torchaudio.load(path)
            data.append({
                "name": name,
                "path": path,
                "emotion": label
            })
        except Exception as e:
            print('Could not load ', str(path), e)
            exceptions+=1
            pass

    if exceptions > 0: print(f'{exceptions} files could not be loaded')

    df = pd.DataFrame(data)
    df.emotion.replace({'01':'neutral', '02':'calm', '03':'happy', '04':'sad', '05':'angry', '06':'fear', '07':'disgust', '08':'surprise'}, inplace=True)
    # ravdess_df.head() # do not display on hpc
    return df


def CREMA(data_path):
    print('PREPARING CREMA DATA PATHS')

    data = []
    exceptions = 0

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

        try:
            # avoid broken files
            s = torchaudio.load(path)
            crema_data.append({
                "name": name,
                "path": path,
                "emotion": label
            })
        except Exception as e:
            print('Could not load ', str(path), e)
            exceptions+=1
            pass

    if exceptions > 0: print(f'{exceptions} files could not be loaded')

    df = pd.DataFrame(data)
    return df


def TESS(data_path):
    print('PREPARING TESS DATA PATHS')

    data = []
    exceptions = 0

    for path in tqdm(Path(data_path).glob("**/*.wav")):
        name = str(path).split('/')[-1].split('.')[0]
        label = name.split('_')[-1]
        if label =='ps': label = 'surprise'

        try:
            # avoid broken files
            s = torchaudio.load(path)
            data.append({
                "name": name,
                "path": path,
                "emotion": label
            })
        except Exception as e:
            print('Could not load ', str(path), e)
            exceptions+=1
            pass

    if exceptions > 0: print(f'{exceptions} files could not be loaded')

    df = pd.DataFrame(data)
    return df


def SAVEE(data_path):
    print('PREPARING SAVEE DATA PATHS')

    data = []
    exceptions = 0

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

        try:
            # avoid broken files
            s = torchaudio.load(path)
            data.append({
                "name": name,
                "path": path,
                "emotion": label
            })
        except Exception as e:
            print('Could not load ', str(path), e)
            exceptions+=1
            pass

    if exceptions > 0: print(f'{exceptions} files could not be loaded')

    df = pd.DataFrame(data)
    return df



def get_df(corpus, config, i=None):
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
    if type(config.corpora) == list:
        df = pd.DataFrame()
        for i, corpus in enumerate(config.corpora):
            data_path = config.data_path[i]
            df_ = get_df(corpus, data_path)

            df = pd.concat([df, df_], axis = 0)

    else:
        corpus = config.corpora
        data_path = config.data_path
        df = get_df(corpus, data_path)

    print(f"Step 0: {len(df)}")

    df["status"] = df["path"].apply(lambda path: True if os.path.exists(path) else None)
    df = df.dropna(subset=["path"])
    df = df.drop("status", 1)
    print(f"Step 1: {len(df)}")

    df = df.sample(frac=1)
    df = df.reset_index(drop=True)

    """Let's explore how many labels (emotions) are in the dataset with what distribution."""
    print("Labels: ", df["emotion"].unique())
    print()
    df.groupby("emotion").count()[["path"]]

    return df


def prepare_splits(df, config):
    save_path = "./content/data"

    random_state = config.seed # 101
    train_df, test_df = train_test_split(df, test_size=0.1, train_size=0.9, random_state=random_state, stratify=df["emotion"])
    train_df, valid_df = train_test_split(train_df, test_size=0.11, train_size=0.89, random_state=random_state, stratify=train_df["emotion"])


    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    valid_df = valid_df.reset_index(drop=True)

    train_df.to_csv(f"{save_path}/train.csv", sep="\t", encoding="utf-8", index=False)
    test_df.to_csv(f"{save_path}/test.csv", sep="\t", encoding="utf-8", index=False)
    valid_df.to_csv(f"{save_path}/valid.csv", sep="\t", encoding="utf-8", index=False)

    print(train_df.shape)
    print(test_df.shape)
    print(valid_df.shape)


if __name__ == '__main__':
    # config
    parser = agrparse.ArgumentParser()
    parser.add_argument('--config', help='yaml configuration file path')
    args = parser.parse_arguments()
    config_file = args.config

    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    df = df(config)
    prepare_splits(df, config)
