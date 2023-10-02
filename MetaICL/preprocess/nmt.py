import os
import datasets
import numpy as np
import argparse

from _all_tasks import TASK_TO_PROMPT_TEMPLATE
from fewshot_gym_dataset import FewshotGymDataset, FewshotGymTextToTextDataset

class NMT(FewshotGymTextToTextDataset):

    def __init__(self, hf_identifier) -> None:
        self.hf_identifier = hf_identifier
        self.task_type = "text to text"
        self.license = "unknown"
        self.prompt_template = TASK_TO_PROMPT_TEMPLATE[self.hf_identifier]
        ## Example : f"Translate Hindi to Kannada:\n {{input}}\nTranslation:\n"

    def map_hf_dataset_to_list(self, hf_dataset, split_name):
        lines = []
        for datapoint in hf_dataset[split_name]:
            input = datapoint["source"]
            target = datapoint["target"]
            # if self.prompt_template: # Already in dataset
            #    input = self.prompt_template.format(input=input)
            lines.append((input, target))
        return lines
    
    def load_dataset(self):
        return datasets.load_dataset(self.hf_identifier)
    
def parse_args():
    argparser = argparse.ArgumentParser()
    ## arguments for hf identifier and prompt template
    argparser.add_argument("--hf_identifier", type=str)
    argparser.add_argument("--prompt_template", default = None, type=str)
    args, _  = argparser.parse_known_args()
    return args

def main():
    ## define argparser for hf identifier and prompt template
    args = parse_args()
    dataset = NMT(args.hf_identifier)

    for seed in [100, 13, 21, 42, 87]:
        train, dev, test = dataset.generate_k_shot_data(k=32, seed=seed, path="../../data/")

if __name__ == "__main__":
    main()