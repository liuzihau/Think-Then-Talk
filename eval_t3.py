# eval_t3.py
# Evaluation harness for Think-Then-Talk (T3) model
# Based on Fast-dLLM's eval_llada.py structure

import os
import json
import time
import random
import torch

import numpy as np
import torch.nn.functional as F
from pathlib import Path
from datasets import Dataset
from tqdm import tqdm

from lm_eval.__main__ import cli_evaluate
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from transformers import AutoTokenizer

from model.modeling_t3 import T3Model
from utils import load_ckpt, denoise_k_step_soft_embed_v2, select_reveal_positions


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@register_model("t3_model")
class T3EvalHarness(LM):
    def __init__(
        self,
        ckpt_path='',
        mask_id=126336,
        max_length=4096,
        batch_size=1,
        mc_num=128,
        is_check_greedy=False,
        steps=None,
        gen_length=256,
        block_size=8,
        device="cuda",
        think_device1="cuda:0",
        think_device2="cuda:0",
        talk_device="cuda:0",
        save_dir=None,
        show_speed=False,
        **kwargs,
    ):
        super().__init__()

        self.mask_id = mask_id
        self.batch_size = int(batch_size)
        self.mc_num = mc_num
        self.max_length = max_length
        self.is_check_greedy = is_check_greedy
        self.gen_length = int(gen_length)
        self.block_size = int(block_size)
        self.show_speed = show_speed
        self.save_dir = save_dir

        self.think_device1 = think_device1
        self.think_device2 = think_device2
        self.talk_device = talk_device

        # Load model config from checkpoint
        config_path = os.path.join(ckpt_path, "config.json")
        with open(config_path) as f:
            self.model_config = json.load(f)

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config["pretrained_model_name_or_path"],
            trust_remote_code=True,
        )

        # Load T3 model
        self.model = T3Model(
            self.model_config,
            think_dev1=think_device1,
            think_dev2=think_device2,
            talk_dev=talk_device,
        )
        self.model.eval()
        load_ckpt(ckpt_path, self.model, None, None, map_location="cpu")

        # steps = number of denoise iterations per block
        # In T3, each block gets model.length denoise steps
        self.steps_per_block = self.model.length
        self.num_blocks = self.gen_length // self.block_size
        denoise_cfg = self.model_config.get("denoise", {})
        self.reveal_mode = str(denoise_cfg.get("reveal_strategy", "ar_force"))
        self.reveal_k = int(denoise_cfg.get("reveal_k", 1))

        print(
            f"[T3Eval] steps_per_block={self.steps_per_block}, num_blocks={self.num_blocks}, "
            f"reveal_mode={self.reveal_mode}, reveal_k={self.reveal_k}"
        )

        self._rank = 0
        self._world_size = 1

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    @torch.no_grad()
    def t3_generate(self, input_ids):
        """
        Block-wise generation using T3 model (single-sample path, aligned with inference.py).
        input_ids: [1, seq_len] prompt token ids on think_device1
        Returns: ([1, seq_len + gen_length] full sequence, nfe count)
        """
        B = input_ids.shape[0]
        if B != 1:
            raise ValueError(f"t3_generate expects batch size 1, got {B}")
        seq_len = input_ids.shape[1]
        max_len = seq_len + self.gen_length

        x = torch.full(
            (B, max_len), self.mask_id, dtype=torch.long, device=self.think_device1
        )
        x[:, :seq_len] = input_ids

        # Position ids
        position_ids = torch.arange(0, max_len, device=self.think_device1).unsqueeze(0).expand(B, -1)

        # Attention mask (all ones)
        attention_mask = torch.ones(B, max_len, dtype=torch.bool, device=self.think_device1)

        # Build block attention bias (bool, [max_len, max_len])
        attention_bias = torch.zeros(
            (max_len, max_len), dtype=torch.bool, device=self.think_device1
        )
        # Prompt attends to prompt
        attention_bias[:seq_len, :seq_len] = True
        # Each mask block attends to prompt + all preceding blocks + itself
        for block_idx in range(self.num_blocks):
            r0 = seq_len + block_idx * self.block_size
            r1 = seq_len + (block_idx + 1) * self.block_size
            attention_bias[r0:r1, :r1] = True

        past_key_values = None

        # First think input: prompt + first mask block
        x0 = x[:, :seq_len + self.block_size]
        p0 = position_ids[:, :seq_len + self.block_size]
        attn_mask = attention_mask[:, :seq_len + self.block_size]
        attn_bias = attention_bias[:seq_len + self.block_size, :seq_len + self.block_size]
        attn_bias = attn_bias.unsqueeze(0).unsqueeze(0)  # [1, 1, S, S]

        nfe = 0

        for block_idx in range(self.num_blocks):
            # Think forward
            think_outputs = self.model(
                input_ids=x0,
                attention_mask=attn_mask,
                attention_bias=attn_bias,
                position_ids=p0,
                use_cache=True,
                past_key_values=past_key_values,
                output_hidden_states=True,
            )
            nfe += 1

            past_key_values = think_outputs.past_key_values
            # Trim KV cache: remove the mask block tokens
            new_past_key_values = []
            for i in range(len(past_key_values)):
                new_past_key_values.append(())
                for j in range(len(past_key_values[i])):
                    new_past_key_values[i] += (
                        past_key_values[i][j][:, :, :-self.block_size],
                    )
            past_key_values = new_past_key_values

            think_rps = think_outputs.hidden_states

            # Extract talk inputs for current block
            s = seq_len + self.block_size * block_idx
            e = seq_len + self.block_size * (block_idx + 1)

            if x0.shape[-1] > 2 * self.block_size:
                talk_input_ids = x0[:, s:e]
                talk_rps = think_rps[:, s:e, :]
            else:
                talk_input_ids = x0[:, -self.block_size:]
                talk_rps = think_rps[:, -self.block_size:, :]

            if self.think_device2 != self.talk_device:
                talk_input_ids = talk_input_ids.to(self.talk_device)
                talk_rps = talk_rps.to(self.talk_device)

            talk_attn_mask = torch.ones_like(
                talk_input_ids, dtype=torch.long, device=self.talk_device
            )
            talk_attn_bias = torch.zeros(
                (1, 1, self.block_size, self.block_size),
                device=self.talk_device,
                dtype=torch.float32,
            )
            loss_mask = torch.ones_like(
                talk_attn_mask, dtype=torch.float32, device=self.talk_device
            )

            talk_input_embeds = F.embedding(talk_input_ids, self.model.talk_embed_weight)

            # Denoise loop
            for idx in range(self.steps_per_block):
                talk_outputs = self.model(
                    input_ids=None,
                    inputs_embeds=talk_input_embeds,
                    inputs_repres=talk_rps,
                    attention_mask=talk_attn_mask,
                    attention_bias=talk_attn_bias,
                    use_cache=False,
                    output_hidden_states=True,
                )
                nfe += 1
                logits = talk_outputs.logits.float()
                talk_rps = talk_outputs.hidden_states

                if self.model_config["soft_inputs"]["enabled"]:
                    talk_input_ids, talk_input_embeds, loss_mask = denoise_k_step_soft_embed_v2(
                        input_ids=talk_input_ids,
                        target=None,
                        loss_mask=loss_mask,
                        logits=logits,
                        emb_weight=self.model.talk_embed_weight,
                        k_reveal=self.reveal_k,
                        soft_topk=self.model_config["soft_inputs"]["top_k"],
                        soft_temp=self.model_config["soft_inputs"]["temperature"],
                        mode=self.reveal_mode,
                        sample_tokens=False,
                    )
                else:
                    # Hard iterative reveal fallback (same as inference.py)
                    idx = select_reveal_positions(
                        loss_mask=loss_mask,
                        logits=logits,
                        k_reveal=self.reveal_k,
                        mode=self.reveal_mode,
                    )
                    rows = torch.arange(
                        talk_input_ids.size(0), device=self.talk_device
                    ).unsqueeze(1).expand_as(idx)
                    chosen_active = loss_mask.bool().gather(1, idx)
                    rows = rows[chosen_active]
                    cols = idx[chosen_active]
                    pred = logits.argmax(dim=-1)
                    talk_input_ids = talk_input_ids.clone()
                    loss_mask = loss_mask.clone()
                    talk_input_ids[rows, cols] = pred[rows, cols]
                    loss_mask[rows, cols] = 0
                    talk_input_embeds = F.embedding(talk_input_ids, self.model.talk_embed_weight)

            # After denoise loop: update x and prepare next block
            if self.talk_device != self.think_device1:
                talk_input_ids = talk_input_ids.to(self.think_device1)

            x[:, s:e] = talk_input_ids

            # Prepare next block input
            x0 = x[:, s:e + self.block_size]
            p0 = position_ids[:, s:e + self.block_size]
            attn_mask = attention_mask[:, :e + self.block_size]
            attn_bias = attention_bias[:e + self.block_size, :e + self.block_size]
            attn_bias = attn_bias.unsqueeze(0).unsqueeze(0)

        return x, nfe

    def _encode_pair(self, context, continuation):
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]

        whole_enc = self.tokenizer(context + continuation)["input_ids"]
        context_enc = self.tokenizer(context)["input_ids"]

        context_enc_len = len(context_enc)
        continuation_enc = whole_enc[context_enc_len:]

        return context_enc, continuation_enc

    def _forward_process(self, batch, prompt_index):
        """Monte Carlo forward diffusion for log-likelihood estimation."""
        b, l = batch.shape
        target_len = (l - prompt_index.sum()).item()
        k = torch.randint(1, target_len + 1, (), device=batch.device)

        x = torch.round(
            torch.linspace(
                float(k), k + (b - 1) * (target_len / b), steps=b, device=batch.device
            )
        ).long()
        x = ((x - 1) % target_len) + 1

        indices = torch.arange(target_len, device=batch.device).repeat(b, 1)
        is_mask = indices < x.unsqueeze(1)

        for i in range(b):
            is_mask[i] = is_mask[i][torch.randperm(target_len)]

        is_mask = torch.cat(
            (
                torch.zeros(b, prompt_index.sum(), dtype=torch.bool, device=batch.device),
                is_mask,
            ),
            dim=1,
        )

        noisy_batch = torch.where(is_mask, self.mask_id, batch)
        return noisy_batch, (x / target_len).unsqueeze(1).repeat(1, l)

    @torch.no_grad()
    def get_loglikelihood(self, prefix, target):
        """
        Estimate log-likelihood using the THINK model directly (MDM style).
        This uses the base LLaDA model's bidirectional attention for MC estimation.
        """
        seq = torch.cat([prefix, target])[None, :]
        seq = seq.repeat((self.batch_size, 1)).to(self.think_device1)
        prompt_index = torch.arange(seq.shape[1], device=self.think_device1) < len(prefix)

        loss_acc = []
        for _ in range(self.mc_num // self.batch_size):
            perturbed_seq, p_mask = self._forward_process(seq, prompt_index)
            mask_indices = perturbed_seq == self.mask_id

            # Use think model directly for likelihood (full bidirectional)
            inputs_embeds = self.model.embed_think(perturbed_seq)
            hidden_states, _ = self.model.run_think_model(
                inputs_embeds=inputs_embeds,
                attention_mask=torch.ones_like(perturbed_seq, dtype=torch.bool),
                use_cache=False,
                output_hidden_states=False,
            )

            # Get logits from the think model's lm_head
            if self.model.architecture == "LLaDA":
                logits = F.linear(
                    hidden_states[-1] if isinstance(hidden_states, tuple) else hidden_states,
                    self.model.talk_lm_head_weight,
                    self.model.talk_lm_head_bias,
                )
            else:
                logits = self.model.think_model_root.lm_head(
                    hidden_states[-1] if isinstance(hidden_states, tuple) else hidden_states
                )

            loss = (
                F.cross_entropy(logits[mask_indices], seq[mask_indices], reduction="none")
                / p_mask[mask_indices]
            )
            loss = loss.sum() / self.batch_size
            loss_acc.append(loss.item())

        return -sum(loss_acc) / len(loss_acc)

    @torch.no_grad()
    def suffix_greedy_prediction(self, prefix, target):
        if not self.is_check_greedy:
            return False
        return False

    def loglikelihood(self, requests):
        def _tokenize(e):
            prefix, target = self._encode_pair(e["prefix"], e["target"])
            return {
                "prefix_text": e["prefix"],
                "target_text": e["target"],
                "prefix": prefix,
                "target": target,
            }

        ds = [{"prefix": req.args[0], "target": req.args[1]} for req in requests]
        ds = Dataset.from_list(ds)
        ds = ds.map(_tokenize)
        ds = ds.with_format("torch")

        out = []
        with torch.no_grad():
            for elem in tqdm(ds, desc="Computing likelihood..."):
                prefix = elem["prefix"]
                target = elem["target"]
                ll = self.get_loglikelihood(prefix, target)
                is_target_greedy_dec = self.suffix_greedy_prediction(prefix, target)
                out.append((ll, 1.0 if is_target_greedy_dec else 0.0))
        torch.cuda.empty_cache()
        return out

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError

    def generate_until(self, requests):
        output = [None] * len(requests)
        num_tokens = 0
        total_nfe = 0

        start_time = time.time()
        pbar = tqdm(total=len(requests), desc="Generating...")

        for req_idx, req in enumerate(requests):
            question = req.args[0]
            stop_tokens = req.args[1]["until"]

            m = [{"role": "user", "content": question}]
            user_input = self.tokenizer.apply_chat_template(
                m, add_generation_prompt=True, tokenize=False
            )
            ids_1d = self.tokenizer(
                user_input, return_tensors="pt", add_special_tokens=False
            ).input_ids[0]

            prompt_len = int(ids_1d.shape[0])
            input_batch = ids_1d.unsqueeze(0).to(self.think_device1)

            generated, nfe = self.t3_generate(input_batch)
            total_nfe += nfe

            gen_ids = generated[0, prompt_len:]
            gen_text = self.tokenizer.decode(gen_ids, skip_special_tokens=False)

            for stop_seq in stop_tokens:
                if stop_seq in gen_text:
                    gen_text = gen_text.split(stop_seq)[0]

            gen_ids_clean = self.tokenizer(gen_text)["input_ids"]
            if self.show_speed:
                num_tokens += sum(1 for t in gen_ids_clean if t != 126081)

            gen_text_clean = self.tokenizer.decode(
                gen_ids_clean, skip_special_tokens=True
            )
            output[req_idx] = gen_text_clean

            if req_idx < 3 or req_idx % 100 == 0:
                print("=" * 20)
                print(f"[{req_idx}/{len(requests)}] answer:", gen_text_clean[:200])
                print("nfe:", nfe)
                print("=" * 20)

            pbar.update(1)

        pbar.close()

        end_time = time.time()
        if self.show_speed:
            elapsed = end_time - start_time
            print(f"Total tokens generated: {num_tokens}")
            print(f"Total time: {elapsed:.2f}s")
            print(f"Tokens/sec: {num_tokens / elapsed:.2f}")
            print(f"Total NFE: {total_nfe}")

        return output


if __name__ == "__main__":
    cli_evaluate()
