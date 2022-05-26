# XLSER

Cross Lingual Speech Emotions Recognition using Wav2vec2

This projects evaluates the performance of models pre-trained on different languages in recognizing emotions.

The datasets used contains speech emotions in different languages.

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
python3 evaluate.py --config content/config/<config-file>.yaml
```
#### 5. Predict
Demonstrate the model
```
```

## Datasets
> currently compatible with: KSUemotions, RAVDESS, CREMAD, TESS, SAVEE
> to be added: AESDD, IEMOCAP, JLCorpus

## File Structure
```
XLSER
├── build_model.py
├── content
│   ├── config
│   │   ├── testing.yaml
│   │   └── ...
│   ├── data
│   └── models
│      └── model 1
│       │   └───features
│       │   │   ├── train_dataset
│       │   │   └── eval_dataset
│       │   └── splits
│       │       ├── test.csv
│       │       ├── train.csv
│       │       └── valid.csv
│       │
│       └── model 2
├── evaluate.py
├── nested_array_catcher.py
├── prepare_data.py
├── preprocess_data.py
├── README.md
└── train.py

```

## References
1. [Emotion Recognition in Greek Speech Using Wav2Vec 2.0](https://colab.research.google.com/github/m3hrdadfi/soxan/blob/main/notebooks/Emotion_recognition_in_Greek_speech_using_Wav2Vec2.ipynb)
