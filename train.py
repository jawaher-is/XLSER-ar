"""
For training:
    #   We preprocess the audio files, and prepare the training splits,
    #   Build the classification model based on the merge strategy,
    #   Use Huggingface's Trainer to set up the training pipline. For that, we need the follwing:
        -   Define a data collator. https://github.com/huggingface/transformers/blob/9a06b6b11bdfc42eea08fa91d0c737d1863c99e3/examples/research_projects/wav2vec2/run_asr.py#L81
        -   Evaluation metric.
        -   Load a pretrained checkpoint.
        -   Define the training parameters.
"""

import argparse
import yaml
import os

import preprocess_data
import build_model
import evaluate

from transformers import (
    Trainer,
    is_apex_available,
    TrainingArguments,
)

from typing import Any, Dict, Union

import torch
from packaging import version
from torch import nn



# Get the configuration file
parser = argparse.ArgumentParser()
parser.add_argument('--config', help='yaml configuration file path') # EXAMPLE content/config/test.yaml
args = parser.parse_args()
config_file = args.config

with open(config_file) as f:
    configuration = yaml.load(f, Loader=yaml.FullLoader)


# Prepare data
train_filepath = configuration['output_dir'] + "/splits/train.csv"
test_filepath = configuration['output_dir'] + "/splits/test.csv"
valid_filepath = configuration['output_dir'] + "/splits/valid.csv"

if not os.path.exists(train_filepath) or not os.path.exists(test_filepath) or not os.path.exists(valid_filepath):
    import prepare_data
    # prepare datasplits
    df = prepare_data.df(configuration['corpora'], configuration['data_path'])
    prepare_data.prepare_splits(df, configuration)


# Prepare data splits for Training
train_dataset, eval_dataset, input_column, output_column, label_list, num_labels = preprocess_data.training_data(configuration)
# Load the processor of the underlying pretrained wav2vec model
config, processor, target_sampling_rate = preprocess_data.load_processor(configuration, label_list, num_labels)
# Get the preprocessed data splits
train_dataset, eval_dataset = preprocess_data.preprocess_data(configuration, processor, target_sampling_rate, train_dataset, eval_dataset, input_column, output_column, label_list)


# Define the data collator
data_collator = build_model.data_collator(processor)

# Define evaluation metrics
compute_metrics = build_model.compute_metrics

# Load the pretrained XLSR-Wav2Vec2 checkpoint into our classification model with a pooling strategy.
model = build_model.load_pretrained_checkpoint(config, configuration['processor_name_or_path'])

"""
The first component of XLSR-Wav2Vec2 consists of a stack of CNN layers that are used to extract acoustically meaningful - but contextually independent - features from the raw speech signal. This part of the model has already been sufficiently trained during pretraining and as stated in the [paper](https://arxiv.org/pdf/2006.13979.pdf) does not need to be fine-tuned anymore.
Thus, we can set the `requires_grad` to `False` for all parameters of the *feature extraction* part.
"""
if configuration['freeze_feature_extractor']:
    model.freeze_feature_extractor()


"""
Define the training parameters
    - `learning_rate` and `weight_decay` were heuristically tuned until fine-tuning has become stable. Note that those parameters strongly depend on the Common Voice dataset and might be suboptimal for other speech datasets.
For more explanations on other parameters, one can take a look at the [docs](https://huggingface.co/transformers/master/main_classes/trainer.html?highlight=trainer#trainingarguments).
"""

training_args = TrainingArguments(
    output_dir=configuration['output_dir'],
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,
    evaluation_strategy="steps",
    num_train_epochs=1.0,
    fp16=True,
    save_steps=10,
    eval_steps=10,
    logging_steps=10,
    learning_rate=1e-4,
    save_total_limit=2,
)


"""For future use we can create our training script, we do it in a simple way. You can add more on you own."""
if is_apex_available():
    from apex import amp

if version.parse(torch.__version__) >= version.parse("1.6"):
    _is_native_amp_available = True
    from torch.cuda.amp import autocast

class CTCTrainer(Trainer):
    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to train.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.

        Return:
            :obj:`torch.Tensor`: The tensor with training loss on this batch.
        """

        model.train()
        inputs = self._prepare_inputs(inputs)

        if self.use_amp:
            with autocast():
                loss = self.compute_loss(model, inputs)
        else:
            loss = self.compute_loss(model, inputs)

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        if self.use_amp:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            self.deepspeed.backward(loss)
        else:
            loss.backward()

        return loss.detach()


# Pass all instances to Trainer
trainer = CTCTrainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=processor.feature_extractor,
)

# Train
trainer.train()



# Evaluate

# Load test data
test_dataset = load_dataset("csv",
    data_files={"test": test_filepath},
    delimiter="\t",
    cache_dir=configuration['cache_dir'])["test"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Load model configuration, processor, and pretrained checkpoint
config, processor, model = evaluate.load_model(configuration, device)

# Resample the audio files
test_dataset = test_dataset.map(evaluate.speech_file_to_array_fn,
    fn_kwargs=dict(processor=processor)
    )

# get predictions
result = test_dataset.map(evaluate.predict,
    batched=True,
    batch_size=8,
    fn_kwargs=dict(configuration=configuration, processor=processor, model=model, device=device)
    )

label_names = [config.id2label[i] for i in range(config.num_labels)]
label_names

y_true = [config.label2id[name] for name in result["emotion"]]
y_pred = result["predicted"]

print("Sample true values: \t", y_true[:5])
print("Sample predicted values: \t", y_pred[:5])

print(classification_report(y_true, y_pred, target_names=label_names))

# TODO: save evaluation results to file
