"""
Preprocess training and validation dataset splits and save the features in ./content/<modelname>/features/
"""

import os

from transformers import AutoConfig, Wav2Vec2Processor

def training_data(configuration):
    """
    Prepare data splits for Training
    return: train_dataset, eval_dataset, input_column, output_column, label_list, num_labels
    """

    # Load the created dataset splits using datasets
    from datasets import load_dataset

    train_filepath = configuration['output_dir'] + "/splits/train.csv"
    valid_filepath = configuration['output_dir'] + "/splits/valid.csv"

    data_files = {
        "train": train_filepath,
        "validation": valid_filepath,
    }

    dataset = load_dataset("csv", data_files=data_files,
                delimiter="\t",
                cache_dir=configuration['cache_dir']
                )
    train_dataset = dataset["train"]
    eval_dataset = dataset["validation"]

    print('train_dataset: ', train_dataset)
    print('eval_dataset: ', eval_dataset)

    # Specify the input and output columns
    input_column = "path"
    output_column = "emotion"

    # Distinguish the unique labels in the dataset
    label_list = train_dataset.unique(output_column)
    label_list.sort()  # Sort it for determinism
    num_labels = len(label_list)
    print(f"A classification problem with {num_labels} classes: {label_list}")

    return train_dataset, eval_dataset, input_column, output_column, label_list, num_labels


def load_processor(configuration, label_list, num_labels):
    """
    Load the processor of the underlying pretrained wav2vec model
    return: config, processor, target_sampling_rate
    """

    processor_name_or_path = configuration['processor_name_or_path']
    pooling_mode = configuration['pooling_mode']

    # Load model configuration
    config = AutoConfig.from_pretrained(
        processor_name_or_path,
        num_labels=num_labels,
        label2id={label: i for i, label in enumerate(label_list)},
        id2label={i: label for i, label in enumerate(label_list)},
        finetuning_task="wav2vec2_clf",
        cache_dir=configuration['cache_dir']
        )
    setattr(config, 'pooling_mode', pooling_mode)

    # Load processor
    processor = Wav2Vec2Processor.from_pretrained(processor_name_or_path,
                                    cache_dir=configuration['cache_dir']
                                    )
    target_sampling_rate = processor.feature_extractor.sampling_rate
    print(f"The target sampling rate: {target_sampling_rate}")

    return config, processor, target_sampling_rate


def preprocess_data(configuration,
        processor,
        target_sampling_rate,
        train_dataset,
        eval_dataset,
        input_column,
        output_column,
        label_list
        ):
    """
    Preprocess the datasets:
    Resample the audio files, extract features using the processor, and save them to file.
    return: train_dataset, eval_dataset splits with the features
    """
    import torchaudio
    import numpy as np
    from tqdm import tqdm
    from datasets import concatenate_datasets
    from nested_array_catcher import nested_array_catcher

    def speech_file_to_array_fn(path):
        speech_array, sampling_rate = torchaudio.load(path)
        resampler = torchaudio.transforms.Resample(sampling_rate,
                                            target_sampling_rate
                                            )
        speech = resampler(speech_array).squeeze().numpy()
        return speech

    def label_to_id(label, label_list):
        if len(label_list) > 0:
            return label_list.index(label) if label in label_list else -1
        return label

    def preprocess_function(examples):
        # Read all the audio files and resample them to 16kHz
        speech_list = [speech_file_to_array_fn(path) for path in examples[input_column]]
        # Map each audio file to the corresponding label
        target_list = [label_to_id(label, label_list) for label in examples[output_column]]

        result = processor(speech_list, sampling_rate=target_sampling_rate)
        result["labels"] = list(target_list)

        print('\nASSERTING dtype')
        for i in tqdm(range(len(result['input_values']))):
            result['input_values'][i] = nested_array_catcher(result['input_values'][i])

        return result

    features_path = configuration['output_dir'] + '/features'

    if os.path.exists(features_path + "/train_dataset") and os.path.exists(features_path + "/eval_dataset"):

        # Load preprocessed datasets from file
        from datasets import load_from_disk
        train_dataset = load_from_disk(features_path + "/train_dataset")
        eval_dataset = load_from_disk((features_path + "/eval_dataset"))
        print("Loaded preprocessed dataset from file")

    else:
        # Preprocess features using a multiprocess map function
        train_dataset = train_dataset.map(
            preprocess_function,
            batch_size=100,
            batched=True,
            num_proc=4
        )

        # Save preprocessed dataset to file
        train_dataset.save_to_disk(features_path + "/train_dataset")

        eval_dataset = eval_dataset.map(
            preprocess_function,
            batch_size=100,
            batched=True,
            num_proc=4
        )

        eval_dataset.save_to_disk(features_path + "/eval_dataset")

        print(f"\n train and eval features saved to {features_path}")

    print('train_dataset: ', train_dataset)
    idx = 0
    # print(f"Training input_values: {train_dataset[idx]['input_values']}")
    print('return_attention_mask:\t', configuration['return_attention_mask'])
    if configuration['return_attention_mask'] is not False:
        print(f"Training attention_mask: {train_dataset[idx]['attention_mask'][0]}")
    # print(f"Training label: {train_dataset[idx]['labels']} - {train_dataset[idx]['emotion']}")

    return train_dataset, eval_dataset



if __name__ == '__main__':
    import argparse
    import yaml

    # Get the configuration file
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='yaml configuration file path')
    args = parser.parse_args()
    config_file = args.config

    with open(config_file) as f:
        configuration = yaml.load(f, Loader=yaml.FullLoader)


    # prepare_data
    train_filepath = configuration['output_dir'] + "/splits/train.csv"
    test_filepath = configuration['output_dir'] + "/splits/test.csv"
    valid_filepath = configuration['output_dir'] + "/splits/valid.csv"

    if not os.path.exists(train_filepath) or not os.path.exists(test_filepath) or not os.path.exists(valid_filepath):
        import prepare_data
        # prepare datasplits
        df = prepare_data.df(configuration['corpora'], configuration['data_path'])
        prepare_data.prepare_splits(df, configuration)


    # Preprocess data

    train_dataset, eval_dataset, input_column, output_column, label_list, num_labels = training_data(configuration)

    config, processor, target_sampling_rate = load_processor(configuration, label_list)

    train_dataset, eval_dataset = preprocess_data(configuration, processor, target_sampling_rate, train_dataset, eval_dataset, input_column, output_column, label_list)
