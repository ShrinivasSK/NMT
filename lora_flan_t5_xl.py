#Load the FLAN-T5-XL model

import os
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
model_name = "google/flan-t5-xl"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, load_in_8bit=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)

#prepare the model for 8-bit training

from peft import prepare_model_for_int8_training
model = prepare_model_for_int8_training(model)

# load the model for the training

from peft import LoraConfig, get_peft_model, TaskType


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


lora_config = LoraConfig(
    r=16, lora_alpha=32, target_modules=["q", "v"], lora_dropout=0.05, bias="none", task_type="SEQ_2_SEQ_LM"
)

model = get_peft_model(model, lora_config)
model = nn.DataParallel(model,device_ids = [0,1,2])
print_trainable_parameters(model)

#Load the dataset and pre-process

dataset_id = "Pretam/hi-kn" ##"cnn_dailymail" # Hugging Face Dataset Id
dataset_config = "3.0.0" # config/verison of the dataset
#save_dataset_path = sys.argv[2] # local path to save processed dataset
text_column = "source" # column of input text is
summary_column = "target" # column of the output text
# custom instruct prompt start
# prompt_template = f"Summarize the following news article:\n{{input}}\nSummary:\n"
prompt_template = f"Translate Hindi to Kannada:\n {{input}}\nTranslation:\n"


# Load dataset from the hub
dataset = load_dataset(dataset_id,name=dataset_config)
# Load tokenizer of FLAN-t5-base
tokenizer = AutoTokenizer.from_pretrained(model_name)


print(f"Train dataset size: {len(dataset['train'])}")
print(f"Test dataset size: {len(dataset['test'])}")

prompt_length = len(tokenizer(prompt_template.format(input=""))["input_ids"])
max_sample_length = tokenizer.model_max_length - prompt_length
print(f"Prompt length: {prompt_length}")
print(f"Max input length: {max_sample_length}")


from datasets import concatenate_datasets
import numpy as np
# The maximum total input sequence length after tokenization. 
# Sequences longer than this will be truncated, sequences shorter will be padded.
tokenized_inputs = concatenate_datasets([dataset["train"], dataset["test"]]).map(lambda x: tokenizer(x[text_column], truncation=True), batched=True, remove_columns=[text_column, summary_column])
max_source_length = max([len(x) for x in tokenized_inputs["input_ids"]])
max_source_length = min(max_source_length, max_sample_length)
print(f"Max source length: {max_source_length}")
 
# The maximum total sequence length for target text after tokenization. 
# Sequences longer than this will be truncated, sequences shorter will be padded."
tokenized_targets = concatenate_datasets([dataset["train"], dataset["test"]]).map(lambda x: tokenizer(x[summary_column], truncation=True), batched=True, remove_columns=[text_column, summary_column])
target_lenghts = [len(x) for x in tokenized_targets["input_ids"]]
# use 95th percentile as max target length
max_target_length = int(np.percentile(target_lenghts, 95))
print(f"Max target length: {max_target_length}")



def preprocess_function(sample, padding="max_length"):

    inputs = [prompt_template.format(input=item) for item in sample[text_column]]
    model_inputs = tokenizer(inputs, max_length=tokenizer.model_max_length, padding=padding, truncation=True)

    labels = tokenizer(text_target=sample[summary_column], max_length=max_target_length, padding=padding, truncation=True)

    if padding == "max_length":
        labels["input_ids"] = [[(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]]
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=list(dataset["train"].features))
train_data=tokenized_dataset["train"]
test_data=tokenized_dataset["test"]
print("generation_max_length : ",max_target_length)



#load the model and train


from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    "temp",
    evaluation_strategy="epoch",
    learning_rate=1e-3,
    gradient_accumulation_steps=1,
    auto_find_batch_size=True,
    num_train_epochs=3,
    save_steps=1000,
    save_total_limit=8,
    per_device_train_batch_size=64
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=test_data,
)
# model.config.use_cache = False

trainer.train()

lora_model_id = "results"
trainer.model.save_pretrained(lora_model_id)
tokenizer.save_pretrained(lora_model_id)
