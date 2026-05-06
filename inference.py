"""Interactive T3 inference and quick-debug entry point.

Loads a checkpoint, takes a prompt from stdin (or applies a chat template),
runs the shared `T3DecodeEngine`, prints the generated answer + decode time.
For the lm_eval harness path see `eval_t3.py` — both files wrap the same
engine so behaviour is aligned.
"""
import argparse
import json
import time

import torch
from transformers import AutoTokenizer

from model.inference_engine import T3DecodeEngine
from model.modeling_t3 import T3Model
from utils import load_ckpt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive T3 inference.")
    parser.add_argument("--ckpt_path", type=str, required=True,
                        help="Path to a saved T3 checkpoint dir (contains config.json + ckpt.pt).")
    parser.add_argument("--gen_length", type=int, default=768,
                        help="Number of tokens to generate. Must be a multiple of --block_size.")
    parser.add_argument("--block_size", type=int, default=6,
                        help="Tokens decoded per think+talk iteration.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--prompt_mode", choices=["chat", "raw"], default="chat",
                        help="`chat` applies the tokenizer's chat template; `raw` uses the prompt verbatim.")
    return parser.parse_args()


def load_model_and_tokenizer(ckpt_path: str, device: str):
    """Load the T3 model + matching tokenizer from a checkpoint dir."""
    with open(f"{ckpt_path}/config.json") as f:
        model_config = json.load(f)
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["pretrained_model_name_or_path"],
        trust_remote_code=True,
    )
    model = T3Model(model_config, train=False, device=device)
    model.eval()
    load_ckpt(ckpt_path, model, optimizer=None, scheduler=None, map_location="cpu")
    return model, tokenizer, model_config


def build_prompt_tokens(
    tokenizer,
    user_input: str,
    prompt_mode: str,
    device: str,
) -> torch.Tensor:
    """Tokenize the user's prompt under the chosen template. Returns `[1, S]`."""
    if prompt_mode == "raw":
        prompt_text = user_input
    else:
        messages = [{"role": "user", "content": user_input}]
        prompt_text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False,
        )
    input_ids = tokenizer(
        prompt_text, return_tensors="pt", add_special_tokens=False,
    ).input_ids
    return input_ids.to(device)


def main() -> None:
    args = parse_args()

    model, tokenizer, model_config = load_model_and_tokenizer(args.ckpt_path, args.device)
    engine = T3DecodeEngine(
        model=model,
        model_config=model_config,
        block_size=args.block_size,
        gen_length=args.gen_length,
    )

    user_input = input("Enter your question: ")
    input_ids = build_prompt_tokens(tokenizer, user_input, args.prompt_mode, args.device)
    prompt_len = input_ids.shape[1]

    start = time.time()
    x, nfe = engine.generate(input_ids)
    elapsed = time.time() - start

    print(f"Decode time: {elapsed:.3f}s   nfe: {nfe}")
    print("AI Reply:")
    print(tokenizer.decode(x[0, prompt_len:].tolist(), skip_special_tokens=True))


if __name__ == "__main__":
    main()
