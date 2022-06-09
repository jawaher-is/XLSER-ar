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
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoConfig, Wav2Vec2Processor

import build_model

from nested_array_catcher import nested_array_catcher



def get_test_data(configuration):
    test_filepath = configuration['output_dir'] + "/splits/test.csv"

    if not os.path.exists(test_filepath):
        df = prepare_data.df(configuration['corpora'],
            configuration['data_path'])
        prepare_data.prepare_splits(df, configuration)

    if ((configuration['test_corpora'] is not None) and (__name__ == '__main__')):
        """
        In case a different dataset is used to evaluate the model, this 'other'
        dataset is loaded, then the additional labels the model is not trained on
        are removed.
        """
        test_filepath_ = (configuration['output_dir']
            + "/splits/evaluation-splits/test.csv")

        # The original dataset is first loaded to be used for label comparasion
        df = pd.read_csv(test_filepath, delimiter='\t')

        eval_df = prepare_data.df(configuration['test_corpora'],
                    configuration['test_corpora_path']
                    )
        eval_df = prepare_data.remove_additional_labels(df, eval_df)

        prepare_data.prepare_splits(eval_df, configuration, evaluation=True)

        test_filepath = test_filepath_

    # Load test data
    test_dataset = load_dataset("csv",
                    data_files={"test": test_filepath},
                    delimiter="\t",
                    cache_dir=configuration['cache_dir']
                    )["test"]

    return test_dataset


def load_model(configuration, device):
    # Load model configuration, processor, and pretrained checkpoint
    processor_name_or_path = configuration['processor_name_or_path']
    model_name_or_path = configuration['output_dir'] + configuration['checkpoint']

    config = AutoConfig.from_pretrained(model_name_or_path,
                cache_dir=configuration['cache_dir']
                )
    processor = Wav2Vec2Processor.from_pretrained(processor_name_or_path,
                    cache_dir=configuration['cache_dir']
                    )
    model = build_model.Wav2Vec2ForSpeechClassification.from_pretrained(
                model_name_or_path,
                cache_dir=configuration['cache_dir']
                ).to(device)

    return config, processor, model


def speech_file_to_array_fn(batch, processor):
    # Resample the audio files
    speech_array, sampling_rate = torchaudio.load(batch["path"])
    speech_array = speech_array.squeeze().numpy()
    speech_array = librosa.resample(np.asarray(speech_array),
                    sampling_rate,
                    processor.feature_extractor.sampling_rate
                    )

    speech_array = nested_array_catcher(speech_array)

    batch["speech"] = speech_array
    return batch


def predict(batch, configuration, processor, model, device):
    # Extract features using the processor
    features = processor(batch["speech"],
                    sampling_rate=processor.feature_extractor.sampling_rate,
                    return_tensors="pt",
                    padding=True
                    )

    input_values = features.input_values.to(device)

    if configuration['return_attention_mask'] is not False:
        attention_mask = features.attention_mask.to(device)
    else:
        attention_mask = None

    # Pass input values through the model to get predictions
    with torch.no_grad():
        logits = model(input_values, attention_mask=attention_mask).logits

    pred_ids = torch.argmax(logits, dim=-1).detach().cpu().numpy()
    batch["predicted"] = pred_ids
    return batch


def report(configuration, y_true, y_pred, label_names, labels=None):
    clsf_report = classification_report(y_true,
                    y_pred,
                    labels=labels,
                    target_names=label_names,
                    zero_division=0,
                    output_dict=True
                    )

    clsf_report_df = pd.DataFrame(clsf_report).transpose()

    print(clsf_report_df)

    if ((configuration['test_corpora'] is not None) and (__name__ == '__main__')):
        file_name = (configuration['output_dir'].split('/')[-1]
                + '-evaluated-on-'
                + configuration['test_corpora']
                + '_clsf_report.csv')
    else:
        file_name = 'clsf_report.csv'

    clsf_report_df.to_csv(configuration['output_dir']
                    + '/'
                    + file_name, sep ='\t')

    return clsf_report_df



if __name__ == '__main__':
    # Get the configuration file
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='yaml configuration file path')
    args = parser.parse_args()
    config_file = args.config

    with open(config_file) as f:
        configuration = yaml.load(f, Loader=yaml.FullLoader)
    print('Loaded configuration file: ', config_file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    config, processor, model = load_model(configuration, device)

    test_dataset = get_test_data(configuration)

    test_dataset = test_dataset.map(speech_file_to_array_fn,
        fn_kwargs=dict(processor=processor)
        )

    result = test_dataset.map(predict,
        batched=True,
        batch_size=8,
        fn_kwargs=dict(configuration=configuration,
                    processor=processor,
                    model=model,
                    device=device
                    )
        )

    label_names = [config.id2label[i] for i in range(config.num_labels)]
    labels = list(config.id2label.keys())

    y_true = [config.label2id[name] for name in result["emotion"]]
    y_pred = result["predicted"]

    print("Sample true values: \t", y_true[:5])
    print("Sample predicted values: \t", y_pred[:5])

    print(classification_report(y_true, y_pred, labels=labels, target_names=label_names)) # , zero_division=0
    print(confusion_matrix(y_true, y_pred))
    clsf_report_df = report(configuration, y_true, y_pred, label_names, labels)
