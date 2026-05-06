"""lm_eval harness for Think-Then-Talk (T3).

Drop-in lm_eval model that wraps `T3DecodeEngine` for `generate_until` and uses
the think backbone directly for `loglikelihood` (MC log-likelihood estimation,
LLaDA-style). Same decode core as `inference.py` — see `model/inference_engine.py`.

Structure based on Fast-dLLM's `eval_llada.py` pattern.
"""
import json
import os
import time
from typing import Any, Dict

import torch
import torch.nn.functional as F
from datasets import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from lm_eval.api.instance import Instance  # noqa: F401  (kept for downstream callers)
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model

from eval.overrides import DenoiseOverrides
from model.inference_engine import T3DecodeEngine
from model.modeling_t3 import T3Model
from model.modeling_t3_infer import T3InferenceModel
from utils import load_ckpt, set_seed


class _BaseT3EvalHarness(LM):
    """Common eval harness; subclasses fix `MODEL_CLS`."""

    MODEL_CLS = T3Model

    def __init__(
        self,
        ckpt_path: str = "",
        mask_id: int = 126336,
        max_length: int = 4096,
        batch_size: int = 1,
        mc_num: int = 128,
        is_check_greedy: bool = False,
        gen_length: int = 256,
        block_size: int = 8,
        device: str = "cuda",
        save_dir=None,
        show_speed: bool = False,
        prompt_prefix: str = "",
        prompt_suffix: str = "",
        seed: int = 0,
        **override_kwargs: Any,
    ):
        super().__init__()

        set_seed(int(seed))

        self.mask_id = int(mask_id)
        self.batch_size = int(batch_size)
        self.mc_num = int(mc_num)
        self.max_length = int(max_length)
        self.is_check_greedy = bool(is_check_greedy)
        self.gen_length = int(gen_length)
        self.block_size = int(block_size)
        self.show_speed = bool(show_speed)
        self.save_dir = save_dir
        self.prompt_prefix = prompt_prefix
        self.prompt_suffix = prompt_suffix
        self.device = str(device)

        # Load checkpoint config and apply optional shell-supplied denoise overrides.
        with open(os.path.join(ckpt_path, "config.json")) as f:
            self.model_config = json.load(f)
        self.model_config = DenoiseOverrides.from_kwargs(**override_kwargs).apply(
            self.model_config
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config["pretrained_model_name_or_path"],
            trust_remote_code=True,
        )

        self.model = self.MODEL_CLS(
            self.model_config,
            train=False,
            device=self.device,
        )
        self.model.eval()
        load_ckpt(ckpt_path, self.model, optimizer=None, scheduler=None, map_location="cpu")

        self.engine = T3DecodeEngine(
            model=self.model,
            model_config=self.model_config,
            block_size=self.block_size,
            gen_length=self.gen_length,
            mask_token_id=self.mask_id,
        )

        print(
            f"[T3Eval] block_size={self.block_size} gen_length={self.gen_length} "
            f"steps_per_block={self.engine.steps_per_block} "
            f"reveal_k={self.engine.reveal_k}"
        )
        print(
            "[T3Eval] reveal_cfg=" + json.dumps(self.engine.reveal_cfg, ensure_ascii=False)
            + " decode_cfg=" + json.dumps(self.engine.decode_cfg, ensure_ascii=False)
        )

        self._rank = 0
        self._world_size = 1

    # ---- LM interface ---------------------------------------------------------

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def loglikelihood(self, requests):
        """Estimate log-likelihood per request via Monte Carlo forward diffusion.

        Uses the think backbone directly (full bidirectional attention) — does
        not go through the block-by-block decode engine.
        """
        rows = [{"prefix": req.args[0], "target": req.args[1]} for req in requests]
        ds = Dataset.from_list(rows).map(self._tokenize_pair).with_format("torch")

        out = []
        with torch.no_grad():
            for elem in tqdm(ds, desc="Computing likelihood..."):
                ll = self._get_loglikelihood(elem["prefix"], elem["target"])
                greedy_match = self._is_target_greedy(elem["prefix"], elem["target"])
                out.append((ll, 1.0 if greedy_match else 0.0))
        torch.cuda.empty_cache()
        return out

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError

    def generate_until(self, requests):
        """Run block-by-block generation per request via the shared engine."""
        outputs = [None] * len(requests)
        total_nfe = 0
        total_tokens = 0
        start = time.time()

        pbar = tqdm(total=len(requests), desc="Generating...")
        for idx, req in enumerate(requests):
            question = self._build_question(req.args[0])
            messages = [{"role": "user", "content": question}]
            prompt_text = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False,
            )
            prompt_ids = self.tokenizer(
                prompt_text, return_tensors="pt", add_special_tokens=False,
            ).input_ids.to(self.device)
            prompt_len = int(prompt_ids.shape[1])

            generated, nfe = self.engine.generate(prompt_ids)
            total_nfe += nfe

            gen_text = self.tokenizer.decode(
                generated[0, prompt_len:], skip_special_tokens=True,
            )
            outputs[idx] = self._prepare_generation_for_eval(req, gen_text)

            if self.show_speed:
                gen_ids_clean = self.tokenizer(gen_text)["input_ids"]
                total_tokens += sum(1 for t in gen_ids_clean if t != 126081)

            print("=" * 20)
            print("answer: ", gen_text)
            print("nfe: ", nfe, "  avg nfe: ", total_nfe / (idx + 1))
            print("=" * 20, end="\n\n")
            pbar.update(1)
        pbar.close()

        if self.show_speed:
            elapsed = time.time() - start
            print(
                f"Total tokens: {total_tokens}   total time: {elapsed:.2f}s   "
                f"tokens/sec: {total_tokens / elapsed:.2f}   total nfe: {total_nfe}"
            )

        return outputs

    # ---- Internals -----------------------------------------------------------

    def _build_question(self, question: str) -> str:
        return f"{self.prompt_prefix}{question}{self.prompt_suffix}"

    def _prepare_generation_for_eval(self, req, gen_text: str) -> str:
        """Hook for subclasses; default is identity."""
        return gen_text

    def _tokenize_pair(self, e: Dict[str, str]) -> Dict[str, Any]:
        prefix, target = self._encode_pair(e["prefix"], e["target"])
        return {
            "prefix_text": e["prefix"],
            "target_text": e["target"],
            "prefix": prefix,
            "target": target,
        }

    def _encode_pair(self, context: str, continuation: str):
        # Standard lm_eval pattern: shift trailing whitespace from context onto
        # continuation so tokenization sees consistent boundaries.
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]
        whole = self.tokenizer(context + continuation)["input_ids"]
        ctx = self.tokenizer(context)["input_ids"]
        return ctx, whole[len(ctx):]

    def _forward_process(self, batch: torch.Tensor, prompt_index: torch.Tensor):
        """Monte Carlo forward diffusion for log-likelihood estimation."""
        b, l = batch.shape
        target_len = (l - prompt_index.sum()).item()
        k = torch.randint(1, target_len + 1, (), device=batch.device)
        x = torch.round(
            torch.linspace(float(k), k + (b - 1) * (target_len / b), steps=b, device=batch.device)
        ).long()
        x = ((x - 1) % target_len) + 1

        indices = torch.arange(target_len, device=batch.device).repeat(b, 1)
        is_mask = indices < x.unsqueeze(1)
        for i in range(b):
            is_mask[i] = is_mask[i][torch.randperm(target_len)]

        is_mask = torch.cat(
            (torch.zeros(b, prompt_index.sum(), dtype=torch.bool, device=batch.device), is_mask),
            dim=1,
        )
        noisy = torch.where(is_mask, self.mask_id, batch)
        return noisy, (x / target_len).unsqueeze(1).repeat(1, l)

    @torch.no_grad()
    def _get_loglikelihood(self, prefix: torch.Tensor, target: torch.Tensor) -> float:
        seq = torch.cat([prefix, target])[None, :].repeat((self.batch_size, 1)).to(self.device)
        prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)

        loss_acc = []
        for _ in range(self.mc_num // self.batch_size):
            perturbed, p_mask = self._forward_process(seq, prompt_index)
            mask_idx = perturbed == self.mask_id

            inputs_embeds = self.model.embed_think(perturbed)
            hidden_states, _ = self.model.run_think_model(
                inputs_embeds=inputs_embeds,
                attention_mask=torch.ones_like(perturbed, dtype=torch.bool),
                use_cache=False,
                output_hidden_states=False,
                prefer_no_mask=True,
            )
            h = hidden_states[-1] if isinstance(hidden_states, tuple) else hidden_states

            if self.model.architecture == "LLaDA":
                logits = F.linear(h, self.model.talk_lm_head_weight, self.model.talk_lm_head_bias)
            else:
                logits = self.model.think_model_root.lm_head(h)

            loss = (
                F.cross_entropy(logits[mask_idx], seq[mask_idx], reduction="none")
                / p_mask[mask_idx]
            )
            loss_acc.append((loss.sum() / self.batch_size).item())
        return -sum(loss_acc) / len(loss_acc)

    @torch.no_grad()
    def _is_target_greedy(self, prefix, target) -> bool:
        # Currently disabled in lm_eval calls; kept for API parity.
        return False if not self.is_check_greedy else False


@register_model("t3_model")
class T3EvalHarness(_BaseT3EvalHarness):
    MODEL_CLS = T3Model


@register_model("t3_model_infer")
class T3InferenceEvalHarness(_BaseT3EvalHarness):
    MODEL_CLS = T3InferenceModel


def main() -> None:
    # `cli_evaluate` is lazy-imported because pulling it at module top costs
    # 1-2 s of import time we don't want unit tests / quick scripts to pay.
    from lm_eval.__main__ import cli_evaluate
    cli_evaluate()


if __name__ == "__main__":
    main()
