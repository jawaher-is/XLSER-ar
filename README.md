# XLSER-ar

This projects evaluates the performance of Wav2vec2 models pre-trained on different languages in recognizing speech emotions recorded in Arabic.

## Requirements
- Pytorch (torch, torchaudio)
- Huggingface (transformers, datasets)
- Pandas
- Librosa
- tqdm
- Scikit-learn


## Configuration
Configuration files are named as follows:  `pretrained-model-name-datasets-names.yaml`

For the structure of the config file, see [template.yaml](https://github.com/jawaher-is/XLSER/blob/main/content/config/template.yaml)


## Execution
...

#### 1. Prepare the data splits
```bash
python3 prepare_data.py --config content/config/<config-file>.yaml
```

#### 2. Preprocess the data
```bash
python3 preprocess_data.py --config content/config/<config-file>.yaml
```
#### 3. Train
```bash
python3 train.py --config content/config/<config-file>.yaml
```

> Or run the training (Step 3) directly, which executes the above steps.

#### 4. Evaluate
The model is evaluated after training in `train.py`. Further, to test the trained model on different data, change `test_corpora` and `test_corpora_path` values in the configuration file.

```bash
python3 evaluate.py --config <config-file-path>
```
#### 5. Predict
Demonstrate the model
```
```

## Datasets
The datasets used contains speech emotions in different languages.
> currently compatible with: KSUemotions, RAVDESS, CREMAD, TESS, SAVEE.
>
> to be added: AESDD, IEMOCAP, JLCorpus

## File Structure
```
XLSER
├── build_model.py
├── content
│   ├── cache
│   ├── config
│   │   ├── template.yaml
│   │   └── ...
│   ├── data
│   └── models
│       ├── model_name
│       │   ├── checkpoint-###
│       │   ├── clsf_report.csv
│       │   ├── features
│       │   │   ├── train_dataset
│       │   │   └── eval_dataset
│       │   ├── splits
│       │   │   ├── test.csv
│       │   │   ├── train.csv
│       │   │   └── valid.csv
│       │   └── ...
│       └── ...
├── evaluate.py
├── nested_array_catcher.py
├── prepare_data.py
├── preprocess_data.py
├── README.md
└── train.py

```

## Results
After evaluation, the classification report of each trained models is saved in the corresponding folder as a csv file. In this repo, they are collected in the `evaluation_results` folder.

## References
1. [Emotion Recognition in Greek Speech Using Wav2Vec 2.0](https://colab.research.google.com/github/m3hrdadfi/soxan/blob/main/notebooks/Emotion_recognition_in_Greek_speech_using_Wav2Vec2.ipynb)
