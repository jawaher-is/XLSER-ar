# XLSER

Cross Lingual Speech Emotions Recognition

This projects evaluates the performance of models pre-trained on different languages in recognizing emotions.

The datasets used contains speech emotions in different languages.

### Requirements
- Pytorch (torch, torchaudio)
...

#### 1. Prepare the data splits
```bash
python3 prepare_data.py --config content/config/<config-file>.yaml
```

After preparing the data splits, if set to null, update the splits' data paths in the configuration file.

#### 2. Preprocess the data
```bash
python3 preprocess_data.py --config content/config/<config-file>.yaml
```
#### 3. Train
```
```
#### 4. Evaluate
```
```
...

For the structure of the config file, see [template.yaml](https://github.com/jawaher-is/XLSER/blob/main/content/config/template.yaml)


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
├── nested_array_catcher.py
├── prepare_data.py
├── preprocess_data.py
├── README.md
└── train.py

```
