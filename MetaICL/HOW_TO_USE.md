## Usage Instructions

Here are the usage instructions for training Meta ICL on a custom set of datasets.

CONDA ENVIRONMENT : meta

### Data Downloading and Preprocessing

Code given in [_build_gym.py](preprocess/_build_gym.py). 

For each dataset we need to define a class that inherits from [`FewshotGymClassificationDataset`](preprocess/fewshot_gym_dataset.py) or [`FewshotGymTextToTextDataset`](preprocess/fewshot_gym_dataset.py) (Later in our case).

We need to define a few functions:
1. __init__ : define self.hf_identifier (hugging face identifier) and self.task_type = "text to text" or "classification"
2. map_hf_dataset_to_list : that iterates over the dataset and returns a list as (input, output) tuple.  
3. load_dataset : that just calls hugging face load dataset function

Look at [`example`](preprocess/aqua_rat.py) for better understanding. 

We can also use prompts by changing the [`_apply_prompts` function](preprocess/fewshot_gym_dataset.py) following [`apply_prompts`](preprocess/utils.py) to a simpler function that just applies a given prompt sent to it (example : prompt template from [here](../generate_dataset_test.py)) OR simply apply prompt on the input in the `map_hf_dataset_to_list` function. Prompts already there in our data. So need to apply them here.  

#### Internal Working

The classes `FewShotGymClassificationDataset` and `FewShotGymTextToTextDataset` generate k shot data that shuffles the generated list of data, preprocess it and saves as a json file in the `/data` folder. 

#### To run

Modify the task list in [_build_gym.py](preprocess/_build_gym.py) for each task. Updated file is [_build_gym_nmt.py](preprocess/mtp/_build_gym_nmt.py)

```
cd preprocess/mtp/
# preprocess from crossfit
python _build_gym_nmt.py --n_proc=40 --do_test
python _build_gym_nmt.py --n_proc=40 --do_train # skip if you won't run training yourself
```
The data will be saved in `/data`.

### Training

Define a config for your task ([`example`](config/class_to_class.json)) with hugging face identifiers of train and test tasks in `config/` folder . Use that task in the commands below.

You might need to change --k 16384 (train size) if your dataset sizes are small. 

I have created a task called nmt. Have set k to 64 as our dataset size for testing was small. So just run the below commands setting task to nmt and k to 64. 

1. Tensorise
```
python train.py --task $task --k 16384 --test_k 16 --seed 100 --use_demonstrations --method channel --do_tensorize --n_gpu 8 --n_process 40
```
2. Train
```
python -m torch.distributed.launch --nproc_per_node=8 train.py --task $task --k 16384 --test_k 16 --seed 100 --train_seed 1 --use_demonstrations --method channel --n_gpu 8 --batch_size 1 --lr 1e-05 --fp16 --optimization 8bit-adam --out_dir checkpoints/channel-metaicl/$task
```