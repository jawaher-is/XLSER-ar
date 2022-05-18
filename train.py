if __name__ == '__main__':

    # Get the configuration file
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='yaml configuration file path') # EXAMPLE content/config/test.yaml
    args = parser.parse_args()
    config_file = args.config

    with open(config_file) as f:
        configurations = yaml.load(f, Loader=yaml.FullLoader)


    # load processed dataset .... or later???
    train_dataset =
    eval_dataset = load_from_disk

    # create model
    import preprocess_data

    train_dataset, eval_dataset, input_column, output_column, label_list, num_labels = preprocess_data.training_data(configurations)

    config, processor, target_sampling_rate = preprocess_data.load_processor(configurations, lablel_list)

    train_dataset, eval_dataset = preprocess_data.preprocess_data(configurations, processor, target_sampling_rate, train_dataset, eval_dataset, input_column, output_column, label_list)


    import build_model

    """
    Define a data collator. (https://github.com/huggingface/transformers/blob/9a06b6b11bdfc42eea08fa91d0c737d1863c99e3/examples/research_projects/wav2vec2/run_asr.py#L81)
        # Unlike most NLP models, XLSR-Wav2Vec2 has a much larger input length than output length. E.g., a sample of input length 50000 has an output length of no more than 100. Given the large input sizes, it is much more efficient to pad the training batches dynamically meaning that all training samples should only be padded to the longest sample in their batch and not the overall longest sample. Therefore, fine-tuning XLSR-Wav2Vec2 requires a special padding data collator, which is defined below.
        # Without going into too many details, in contrast to the common data collators, this data collator treats the `input_values` and `labels` differently and thus applies to separate padding functions on them (again making use of XLSR-Wav2Vec2's context manager). This is necessary because in speech input and output are of different modalities meaning that they should not be treated by the same padding function.
        # Analogous to the common data collators, the padding tokens in the labels with `-100` so that those tokens are **not** taken into account when computing the loss.
    """

    data_collator = build_model.set_up_Trainer(config, processor)

    """
    Now, we can load the pretrained XLSR-Wav2Vec2 checkpoint into our classification model with a pooling strategy.
    We need to load a pretrained checkpoint and configure it correctly for training.
    """
    
    model = build_model.load_pretrained_checkpoint(config, configurations.model_name_or_path)



    # train


    """The first component of XLSR-Wav2Vec2 consists of a stack of CNN layers that are used to extract acoustically meaningful - but contextually independent - features from the raw speech signal. This part of the model has already been sufficiently trained during pretraining and as stated in the [paper](https://arxiv.org/pdf/2006.13979.pdf) does not need to be fine-tuned anymore.
    Thus, we can set the `requires_grad` to `False` for all parameters of the *feature extraction* part.
    """

    model.freeze_feature_extractor()

    """
    - Define the training configuration.
    """
