from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer
import numpy as np 
import sys

# experiment config
model_id = "Enoch/llama-7b-hf" # Hugging Face Model Id
dataset_id = sys.argv[1] ##"cnn_dailymail" # Hugging Face Dataset Id
dataset_config = "3.0.0" # config/verison of the dataset
save_dataset_path = sys.argv[2] # local path to save processed dataset
text_column = "source" # column of input text is
summary_column = "target" # column of the output text 
# custom instruct prompt start
# prompt_template = f"Summarize the following news article:\n{{input}}\nSummary:\n"
prompt_template = f"Translate Hindi to Kannada:\n {{input}}\nTranslation:\n"

# Load dataset from the hub
dataset = load_dataset(dataset_id,name=dataset_config)
# Load tokenizer of model
tokenizer = AutoTokenizer.from_pretrained(model_id)

if 'llama' in model_id: ## set tokeniser params for LLama
    MODEL_MAX_LENGTH = 512
    tokenizer.padding_side = 'right'
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.model_max_length = MODEL_MAX_LENGTH

print(f"Train dataset size: {len(dataset['train'])}")
print(f"Test dataset size: {len(dataset['test'])}")

# Train dataset size: 287113
# Test dataset size: 11490

prompt_length = len(tokenizer(prompt_template.format(input=""))["input_ids"])
max_sample_length = tokenizer.model_max_length - prompt_length
print(f"Prompt length: {prompt_length}")
print(f"Max input length: {max_sample_length}")

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

import os

def preprocess_function(sample, padding="max_length"):
    # created prompted input
    inputs = [prompt_template.format(input=item) for item in sample[text_column]]

    # tokenize inputs
    model_inputs = tokenizer(inputs, max_length=tokenizer.model_max_length, padding=padding, truncation=True)

    # Tokenize targets with the `text_target` keyword argument
    labels = tokenizer(text_target=sample[summary_column], max_length=max_target_length, padding=padding, truncation=True)

    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    if padding == "max_length": 
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# process dataset
tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=list(dataset["train"].features))

# save dataset to disk
tokenized_dataset["train"].save_to_disk(os.path.join(save_dataset_path,"train"))
tokenized_dataset["test"].save_to_disk(os.path.join(save_dataset_path,"eval"))

## save generation max length in a file for later use
with open(os.path.join(save_dataset_path,"metadata.txt"),'w') as f:
    f.write("Generation Max Length: " + str(max_target_length) + "\n")