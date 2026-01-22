import argparse, json
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from train.visualize import visualize_attention_bias_blocks
from train.data_process import build_dataset_rank, DataCollatorWithPaddingV2
from utils import AttrDict

SEED = 0

parser = argparse.ArgumentParser()
parser.add_argument("--trainpath", type=str, default="nvidia/Llama-Nemotron-Post-Training-Dataset")
parser.add_argument("--testpath", type=str, default="nvidia/Llama-Nemotron-Post-Training-Dataset")
parser.add_argument("--savedir", type=str, default="0")
parser.add_argument("--training_config", type=str, default="train/train_config_2.json")
parser.add_argument("--model_config", type=str, default="model/config.json")
parser.add_argument("--think_device", type=str, default="cuda:3")
parser.add_argument("--talk_device", type=str, default="cuda:2")
args = parser.parse_args()

with open(args.training_config) as f:
    train_config = json.load(f)
training_parameters = AttrDict(train_config["training_parameters"])
with open(args.model_config) as f:
    model_config = json.load(f)

tokenizer = AutoTokenizer.from_pretrained(model_config["pretrained_model_name_or_path"], trust_remote_code=True)

testdataset = build_dataset_rank(
    tokenizer, args.testpath, training_parameters["max_len"], target_len=train_config["data"]["block_size"]*train_config["data"]["block_num"],
    get_test_subset=True, seed=SEED
)

test_loader = DataLoader(
    testdataset,
    batch_size=training_parameters["bs"],
    num_workers=1,
    pin_memory=True,
    collate_fn=DataCollatorWithPaddingV2(block_size=train_config["data"]["block_size"], block_num=train_config["data"]["block_num"])
)

for i, data in enumerate(test_loader):
    if i == 3:
        visualize_attention_bias_blocks(
            tokenizer=tokenizer,
            data=data,
            bidx=0,
            block_size=train_config["data"]["block_size"],
            block_num=train_config["data"]["block_num"],
            show_last=128,
            query_mode="last",   # or "first_last"
            max_ctx_tokens=80,
        )
        break
