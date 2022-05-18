"""
For training:
    #   We preprocess the audio files, and prepare the training splits,
    #   Build the classification model based on the merge strategy,
    #   Use Huggingface's Trainer to set up the training pipline. For that, we need the follwing:
        -   Define a data collator. https://github.com/huggingface/transformers/blob/9a06b6b11bdfc42eea08fa91d0c737d1863c99e3/examples/research_projects/wav2vec2/run_asr.py#L81
            +   Unlike most NLP models, XLSR-Wav2Vec2 has a much larger input length than output length. Given the large input sizes, it is much more efficient to pad the training batches dynamically meaning that all training samples should only be padded to the longest sample in their batch and not the overall longest sample. Therefore, fine-tuning XLSR-Wav2Vec2 requires a special padding data collator, which is defined below.
            +   Without going into too many details, in contrast to the common data collators, this data collator treats the `input_values` and `labels` differently and thus applies to separate padding functions on them (again making use of XLSR-Wav2Vec2's context manager). This is necessary because in speech input and output are of different modalities meaning that they should not be treated by the same padding function.
            +   Analogous to the common data collators, the padding tokens in the labels with `-100` so that those tokens are **not** taken into account when computing the loss.
        -   Evaluation metric.
        -   Load a pretrained checkpoint.
        -   Define the training parameters.
"""

import argparse
import yaml

import preprocess_data
import build_model

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
    configurations = yaml.load(f, Loader=yaml.FullLoader)


# Prepare data
train_filepath = configurations['output_dir'] + "/splits/train.csv"
test_filepath = configurations['output_dir'] + "/splits/test.csv"
valid_filepath = configurations['output_dir'] + "/splits/valid.csv"

if not os.path.exists(train_filepath) or not os.path.exists(test_filepath) or not os.path.exists(valid_filepath):
    import prepare_data
    # prepare datasplits
    df = df(configurations)
    prepare_splits(df, configurations)


# Prepare data splits for Training
train_dataset, eval_dataset, input_column, output_column, label_list, num_labels = preprocess_data.training_data(configurations)
# Load the processor of the underlying pretrained wav2vec model
config, processor, target_sampling_rate = preprocess_data.load_processor(configurations, lablel_list)
# Get the preprocessed data splits
train_dataset, eval_dataset = preprocess_data.preprocess_data(configurations, processor, target_sampling_rate, train_dataset, eval_dataset, input_column, output_column, label_list)


# Define the data collator
data_collator = build_model.data_collator(processor)

# Define evaluation metrics
compute_metrics = build_model.compute_metrics()

# Load the pretrained XLSR-Wav2Vec2 checkpoint into our classification model with a pooling strategy.
model = build_model.load_pretrained_checkpoint(config, configurations['model_name_or_path'])

"""
The first component of XLSR-Wav2Vec2 consists of a stack of CNN layers that are used to extract acoustically meaningful - but contextually independent - features from the raw speech signal. This part of the model has already been sufficiently trained during pretraining and as stated in the [paper](https://arxiv.org/pdf/2006.13979.pdf) does not need to be fine-tuned anymore.
Thus, we can set the `requires_grad` to `False` for all parameters of the *feature extraction* part.
"""
model.freeze_feature_extractor()


"""
Define the training parameters
    - `learning_rate` and `weight_decay` were heuristically tuned until fine-tuning has become stable. Note that those parameters strongly depend on the Common Voice dataset and might be suboptimal for other speech datasets.
For more explanations on other parameters, one can take a look at the [docs](https://huggingface.co/transformers/master/main_classes/trainer.html?highlight=trainer#trainingarguments).
"""

training_args = TrainingArguments(
    output_dir=configurations['output_dir'],
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,
    evaluation_strategy="steps",
    num_train_epochs=1.0
    fp16=True,
    save_steps=10
    eval_steps=10
    logging_steps=10
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


"""Now, all instances can be passed to Trainer and we are ready to start training!"""

trainer = CTCTrainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=processor.feature_extractor,
)


""" Training """
trainer.train()
