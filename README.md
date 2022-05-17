# XLSER

Cross Lingual Speech Emotions Recognition

This projects evaluates the performance of models pre-trained on different languages in recognizing emotions.

The datasets used contains speech emotions in different languages.

### Requirements

...

#### 1. Prepare the data splits
```bash
python3 prepare_data.py --config content/config/<config-file>.yaml
```

After preparing the data splits, update the splits' data paths in the configuration file.

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
