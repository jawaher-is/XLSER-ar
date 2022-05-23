# XLSER

Cross Lingual Speech Emotions Recognition

This projects evaluates the performance of models pre-trained on different languages in recognizing emotions.

The datasets used contains speech emotions in different languages.

## Requirements
- Pytorch (torch, torchaudio)
- Huggingface (transformers, datasets)
- Pandas
- Librosa
- tqdm


## Configuration
Configuration files are named as follows:  `pretrained-model-name-datasets-names`

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
```bash
python3 evaluate.py --config content/config/<config-file>.yaml
```
#### 5. Predict
```
```


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
