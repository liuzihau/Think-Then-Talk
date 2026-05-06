import hashlib
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import torch
from datasets import Dataset, Features, Sequence, Value, load_dataset, load_from_disk, concatenate_datasets
import numpy as np

HENDRYCKS_MATH_CONFIGS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]


def build_dataset_rank(
    tokenizer,
    datapaths: str,
    max_len: int,
    target_len: int,
    *,
    splits: str = "chat",
    cache_root: str = "./hf_datasets_cache",
    num_proc: int = 8,
    test_split_ratio: float = 0.05,
    get_test_subset: bool = False,
    seed: int = 42,
    short_target_mode: str = "skip",  # "pad_eos", "skip", or "keep"
    strip_nemotron_math_prompt_prefix: bool = False,
    assistant_think_drop_probs: Optional[str] = None,
    train_subset_percents: Optional[str] = None,
    test_subset_percents: Optional[str] = None,
    train_subset_offsets: Optional[str] = None,
    test_subset_offsets: Optional[str] = None,
):
    """
    datapaths:
      - comma-separated list of:
        - local path to a dataset saved with save_to_disk(), OR
        - HF Hub dataset name (optionally with config, e.g. "org/name" or "org/name:config")

    Output format (unchanged):
      - HuggingFace Dataset with columns: ["input_ids", "target", "attention_mask"]
      - set_format(type="torch") is applied before returning

    Caching:
      1) raw dataset cached to disk under cache_root/<repo_or_path>/<split>
      2) processed dataset cached to disk under cache_root/processed/<cache_key>/(train|test)
    """

    import shutil

    # -----------------------------
    # Helpers
    # -----------------------------
    SEP = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    ROLES_OK = {"human", "user", "gpt", "assistant"}
    NEMOTRON_MATH_PROMPT_PREFIX = (
        "Solve the following math problem. Make sure to put the answer "
        "(and only answer) inside \\boxed{}.\n\n"
    )

    def _is_local_saved_dataset(p: str) -> bool:
        path = Path(p)
        if not path.exists() or not path.is_dir():
            return False
        return (
            (path / "dataset_info.json").exists()
            or (path / "state.json").exists()
            or (path / "data").exists()
        )

    def _parse_hf_id(s: str) -> Tuple[str, Optional[str]]:
        s = s.strip()
        if ":" in s:
            repo_id, config = s.split(":", 1)
            repo_id, config = repo_id.strip(), config.strip()
            return repo_id, (config if config else None)
        return s, None

    def _default_hf_config(repo_id: str, config: Optional[str]) -> Optional[str]:
        if config is not None:
            return config
        if repo_id == "gsm8k":
            return "main"
        if repo_id in {"google-research-datasets/mbpp", "mbpp"}:
            return "full"
        if repo_id in {"cais/mmlu", "hendrycks_test"}:
            return "all"
        return None

    def _safe_dirname(s: str) -> str:
        s = s.strip()
        return re.sub(r"[^\w\-.]+", "_", s)

    def _expand_special_datapaths(
        paths_in: List[str],
        splits_in: List[str],
    ) -> Tuple[List[str], List[str]]:
        out_paths: List[str] = []
        out_splits: List[str] = []

        for datapath, requested_split in zip(paths_in, splits_in):
            repo_id, config = _parse_hf_id(datapath)
            if repo_id == "EleutherAI/hendrycks_math" and config is None:
                out_paths.extend([f"{repo_id}:{cfg}" for cfg in HENDRYCKS_MATH_CONFIGS])
                out_splits.extend([requested_split] * len(HENDRYCKS_MATH_CONFIGS))
            else:
                out_paths.append(datapath)
                out_splits.append(requested_split)

        return out_paths, out_splits

    def _expand_special_aligned_values(
        paths_in: List[str],
        values_in: Optional[List[float]],
    ) -> Optional[List[float]]:
        if values_in is None:
            return None
        if len(paths_in) != len(values_in):
            raise ValueError(
                f"aligned values length ({len(values_in)}) must match original datapaths length ({len(paths_in)})."
            )

        out_values: List[float] = []
        for datapath, value in zip(paths_in, values_in):
            repo_id, config = _parse_hf_id(datapath)
            if repo_id == "EleutherAI/hendrycks_math" and config is None:
                out_values.extend([value] * len(HENDRYCKS_MATH_CONFIGS))
            else:
                out_values.append(value)
        return out_values

    def _parse_percent_list(v: Optional[str], n: int, name: str) -> Optional[List[float]]:
        if v is None:
            return None
        parts = [x.strip() for x in str(v).split(",") if x.strip()]
        if not parts:
            return None
        if len(parts) == 1 and n > 1:
            parts = parts * n
        if len(parts) != n:
            raise ValueError(
                f"{name} has {len(parts)} values, but datapaths has {n}. "
                "Provide one value or one per datapath."
            )
        out: List[float] = []
        for p in parts:
            fp = float(p)
            if fp < 0 or fp > 100:
                raise ValueError(f"{name} values must be in [0, 100], got {fp}")
            out.append(fp)
        return out

    def _deterministic_unit_float(*parts: Any) -> float:
        payload = "||".join(str(p) for p in parts).encode("utf-8")
        digest = hashlib.sha256(payload).digest()
        return int.from_bytes(digest[:8], "big") / float(1 << 64)

    def _strip_assistant_think_blocks(text: str) -> str:
        if not isinstance(text, str):
            return text
        stripped = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        stripped = re.sub(r"\n{3,}", "\n\n", stripped)
        return stripped.strip()

    def _resolve_split(datapath: str, requested_split: str, use_test_split: bool) -> str:
        if requested_split != "auto":
            return requested_split
        repo_id, _ = _parse_hf_id(datapath)
        if repo_id == "gsm8k":
            return "test" if use_test_split else "train"
        if repo_id == "EleutherAI/hendrycks_math":
            return "test" if use_test_split else "train"
        if repo_id in {"cais/mmlu", "hendrycks_test"}:
            return "test" if use_test_split else "auxiliary_train"
        if repo_id in {"openai/openai_humaneval", "openai_humaneval"}:
            return "test"
        if repo_id in {"google-research-datasets/mbpp", "mbpp"}:
            return "test" if use_test_split else "train"
        if repo_id == "nvidia/Llama-Nemotron-Post-Training-Dataset":
            return "chat"
        if repo_id == "allenai/tulu-3-sft-mixture":
            return "train"
        return requested_split

    def _use_native_train_test_split(datapath: str, requested_split: str) -> bool:
        if requested_split != "auto":
            return False
        repo_id, _ = _parse_hf_id(datapath)
        return repo_id in {
            "gsm8k",
            "EleutherAI/hendrycks_math",
            "cais/mmlu",
            "hendrycks_test",
            "openai/openai_humaneval",
            "openai_humaneval",
            "google-research-datasets/mbpp",
            "mbpp",
        }

    def _ensure_pad_token():
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.unk_token_id

    def _apply_chat_template(messages: List[Dict[str, str]]) -> str:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        ).removesuffix("<|start_header_id|>assistant<|end_header_id|>\n\n")

    def _tokenize_ids(text: str) -> List[int]:
        _ensure_pad_token()
        ids = tokenizer(
            text,
            return_tensors=None,
            add_special_tokens=False,
        )["input_ids"]
        return ids

    def _build_prompt_and_target(conversation: str) -> Optional[Tuple[List[int], List[int], List[int]]]:
        full_ids = _tokenize_ids(conversation)

        if len(full_ids) > max_len:
            return None

        turns = conversation.split(SEP)
        if len(turns) < 2:
            return None

        prompt = ""
        for turn in turns[:-1]:
            prompt += turn + SEP

        input_ids = _tokenize_ids(prompt)
        target = full_ids[len(input_ids):]

        if len(target) < target_len:
            if short_target_mode == "skip":
                return None
            if short_target_mode == "pad_eos":
                eos_id = getattr(tokenizer, "eos_token_id", None)
                if eos_id is None:
                    _ensure_pad_token()
                    eos_id = tokenizer.pad_token_id
                if eos_id is None:
                    eos_id = getattr(tokenizer, "unk_token_id", 0)
                target = target + [int(eos_id)] * (target_len - len(target))
            elif short_target_mode == "keep":
                pass
            else:
                raise ValueError(
                    f"Unsupported short_target_mode={short_target_mode}. Use 'pad_eos', 'skip', or 'keep'."
                )

        attention_mask = [1] * len(input_ids)
        return input_ids, target, attention_mask

    def _normalize_gsm8k_answer(answer: str) -> str:
        if not isinstance(answer, str):
            return answer

        normalized = re.sub(
            r"\$?<<([^<>]+)>>([^\s]+)",
            lambda m: f"{m.group(2)} \\({m.group(1)}\\)",
            answer,
        )
        normalized = re.sub(
            r"(?m)^####\s*(.+?)\s*$",
            lambda m: rf"\boxed{{{m.group(1).strip()}}}",
            normalized,
        )
        return normalized

    def _mmlu_answer_to_index(answer: Any, num_choices: int) -> Optional[int]:
        if answer is None:
            return None
        if isinstance(answer, (int, np.integer)):
            idx = int(answer)
            return idx if 0 <= idx < num_choices else None
        if isinstance(answer, str):
            s = answer.strip()
            if not s:
                return None
            if s.isdigit():
                idx = int(s)
                return idx if 0 <= idx < num_choices else None
            ch = s[0].upper()
            if "A" <= ch <= "Z":
                idx = ord(ch) - ord("A")
                return idx if 0 <= idx < num_choices else None
        return None

    def _normalize_mbpp_tests(tests: Any) -> List[str]:
        if tests is None:
            return []
        if isinstance(tests, str):
            s = tests.strip()
            return [s] if s else []
        if isinstance(tests, (list, tuple)):
            out = []
            for item in tests:
                if item is None:
                    continue
                s = str(item).strip()
                if s:
                    out.append(s)
            return out
        s = str(tests).strip()
        return [s] if s else []

    def _empty_processed_dataset():
        empty_features = Features({
            "input_ids": Sequence(Value("int64")),
            "target": Sequence(Value("int64")),
            "attention_mask": Sequence(Value("int64")),
        })
        ds_empty = Dataset.from_dict(
            {"input_ids": [], "target": [], "attention_mask": []},
            features=empty_features,
        )
        ds_empty.set_format(type="torch")
        return ds_empty

    # -----------------------------
    # Dataset-specific adapters
    # -----------------------------
    def _iter_conversations_from_batch(
        datapath: str,
        requested_split: str,
        examples: Dict,
        think_drop_prob: float,
    ) -> List[str]:
        conversations: List[str] = []

        repo_id, _ = _parse_hf_id(datapath)

        if repo_id == "nvidia/Llama-Nemotron-Post-Training-Dataset":
            data_pts = len(examples.get("input", []))
            for i in range(data_pts):
                source = examples["input"][i]
                response = examples["output"][i]

                if not source or source[0].get("role") not in ROLES_OK:
                    try:
                        print(source[0].get("role"))
                    except Exception:
                        print("bad_source_role")
                    continue

                messages = []
                for msg in source:
                    content = msg["content"]
                    if (
                        strip_nemotron_math_prompt_prefix
                        and isinstance(content, str)
                        and content.startswith(NEMOTRON_MATH_PROMPT_PREFIX)
                    ):
                        content = content.removeprefix(NEMOTRON_MATH_PROMPT_PREFIX)
                    messages.append({"role": msg["role"], "content": content})
                if think_drop_prob > 0.0 and _deterministic_unit_float(
                    datapath,
                    requested_split,
                    seed,
                    i,
                    response,
                ) < think_drop_prob:
                    response = _strip_assistant_think_blocks(response)
                if not response:
                    continue
                messages.append({"role": "assistant", "content": response})

                conversations.append(_apply_chat_template(messages))

            return conversations

        if repo_id == "allenai/tulu-3-sft-mixture":
            data_pts = len(examples.get("messages", []))
            for i in range(data_pts):
                source = examples["messages"][i]
                if not source:
                    continue

                messages = []
                if source[0].get("role") == "system":
                    messages = [source[0]]
                    source = source[1:]

                for msg in source:
                    messages.append({"role": msg["role"], "content": msg["content"]})
                    if msg.get("role") == "assistant":
                        conversations.append(_apply_chat_template(messages))

            return conversations

        if repo_id == "gsm8k":
            data_pts = len(examples.get("question", []))
            for i in range(data_pts):
                question = examples["question"][i]
                answer = _normalize_gsm8k_answer(examples["answer"][i])
                if not question or not answer:
                    continue

                messages = []
                messages.append({"role": "user", "content": question})
                messages.append({"role": "assistant", "content": answer})
                conversations.append(_apply_chat_template(messages))

            return conversations

        if repo_id == "EleutherAI/hendrycks_math":
            problems = examples.get("problem", [])
            solutions = examples.get("solution", [])
            data_pts = min(len(problems), len(solutions))
            for i in range(data_pts):
                problem = problems[i]
                solution = solutions[i]
                if not problem or not solution:
                    continue

                user_prompt = (
                    "Solve the following math problem. Make sure to put the answer "
                    "(and only answer) inside \\boxed{}.\n\n"
                    f"{str(problem).strip()}"
                )

                messages = []
                messages.append({"role": "user", "content": user_prompt})
                messages.append({"role": "assistant", "content": str(solution).strip()})
                conversations.append(_apply_chat_template(messages))

            return conversations

        if repo_id in {"openai/openai_humaneval", "openai_humaneval"}:
            prompts = examples.get("prompt", [])
            solutions = examples.get("canonical_solution", [])
            data_pts = min(len(prompts), len(solutions))
            for i in range(data_pts):
                prompt = prompts[i]
                solution = solutions[i]
                if not prompt or not solution:
                    continue
                user_prompt = (
                    "Complete the following Python function.\n\n"
                    f"{prompt}"
                )
                messages = []
                messages.append({"role": "user", "content": user_prompt})
                messages.append({"role": "assistant", "content": solution})
                conversations.append(_apply_chat_template(messages))

            return conversations

        if repo_id in {"google-research-datasets/mbpp", "mbpp"}:
            prompts = examples.get("text", [])
            solutions = examples.get("code", [])
            test_lists = examples.get("test_list", [])
            test_setup_codes = examples.get("test_setup_code", [])
            data_pts = min(len(prompts), len(solutions))
            for i in range(data_pts):
                prompt = prompts[i]
                solution = solutions[i]
                if not prompt or not solution:
                    continue

                tests = _normalize_mbpp_tests(test_lists[i] if i < len(test_lists) else None)
                test_setup_code = ""
                if i < len(test_setup_codes) and test_setup_codes[i] is not None:
                    test_setup_code = str(test_setup_codes[i]).strip()

                user_prompt = (
                    "Write a Python function or program that satisfies the following specification.\n\n"
                    f"{str(prompt).strip()}"
                )
                if tests:
                    user_prompt += "\n\nYour solution should pass these example tests:\n"
                    user_prompt += "\n".join(tests)
                if test_setup_code:
                    user_prompt += "\n\nUse this setup code if needed:\n"
                    user_prompt += test_setup_code

                messages = []
                messages.append({"role": "user", "content": user_prompt})
                messages.append({"role": "assistant", "content": str(solution).strip()})
                conversations.append(_apply_chat_template(messages))

            return conversations

        if repo_id in {"cais/mmlu", "hendrycks_test"}:
            questions = examples.get("question", [])
            choices_all = examples.get("choices", [])
            answers = examples.get("answer", [])
            data_pts = min(len(questions), len(choices_all), len(answers))
            for i in range(data_pts):
                question = questions[i]
                choices = choices_all[i]
                answer = answers[i]
                if not question or not choices:
                    continue
                answer_idx = _mmlu_answer_to_index(answer, len(choices))
                if answer_idx is None:
                    continue

                options = []
                for j, choice_text in enumerate(choices):
                    label = chr(ord("A") + j)
                    options.append(f"{label}. {choice_text}")
                choices_text = "\n".join(options)
                user_prompt = (
                    "Choose the correct option for the following question.\n\n"
                    f"Question: {question}\n\n"
                    f"Options:\n{choices_text}"
                )
                answer_label = chr(ord("A") + answer_idx)
                answer_text = choices[answer_idx]
                assistant = f"The correct answer is {answer_label}. {answer_text}"

                messages = []
                messages.append({"role": "user", "content": user_prompt})
                messages.append({"role": "assistant", "content": assistant})
                conversations.append(_apply_chat_template(messages))

            return conversations

        raise ValueError(f"Unsupported datapath for preprocessing: {datapath}")

    def _make_preprocess_fn(
        datapath: str,
        requested_split: str,
        think_drop_prob: float,
    ) -> Callable[[Dict], Dict]:
        def _fn(examples: Dict) -> Dict:
            new_examples = {"attention_mask": [], "target": [], "input_ids": []}

            conversations = _iter_conversations_from_batch(
                datapath,
                requested_split,
                examples,
                think_drop_prob,
            )
            for conv in conversations:
                out = _build_prompt_and_target(conv)
                if out is None:
                    continue
                input_ids, target, attention_mask = out
                new_examples["input_ids"].append(input_ids)
                new_examples["target"].append(target)
                new_examples["attention_mask"].append(attention_mask)

            return new_examples

        return _fn

    # -----------------------------
    # Resolve datapaths
    # -----------------------------
    cache_root = Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)

    paths = [p.strip() for p in datapaths.split(",") if p.strip()]
    all_splits = [p.strip() for p in splits.split(",") if p.strip()]

    if not paths:
        raise ValueError("datapaths is empty. Provide at least one dataset path or HF dataset id.")

    if len(all_splits) == 1 and len(paths) > 1:
        all_splits = all_splits * len(paths)
    if len(paths) != len(all_splits):
        raise ValueError(
            f"paths ({len(paths)}) and splits ({len(all_splits)}) must have same length "
            "or splits must provide exactly one value."
        )

    original_paths = list(paths)
    train_pcts = _parse_percent_list(train_subset_percents, len(original_paths), "train_subset_percents")
    test_pcts = _parse_percent_list(test_subset_percents, len(original_paths), "test_subset_percents")
    train_offsets = _parse_percent_list(train_subset_offsets, len(original_paths), "train_subset_offsets")
    test_offsets = _parse_percent_list(test_subset_offsets, len(original_paths), "test_subset_offsets")
    think_drop_pcts = _parse_percent_list(
        assistant_think_drop_probs,
        len(original_paths),
        "assistant_think_drop_probs",
    )

    paths, all_splits = _expand_special_datapaths(paths, all_splits)
    train_pcts = _expand_special_aligned_values(original_paths, train_pcts)
    test_pcts = _expand_special_aligned_values(original_paths, test_pcts)
    train_offsets = _expand_special_aligned_values(original_paths, train_offsets)
    test_offsets = _expand_special_aligned_values(original_paths, test_offsets)
    think_drop_pcts = _expand_special_aligned_values(original_paths, think_drop_pcts)
    use_explicit_subset_percents = (
        (train_pcts is not None)
        or (test_pcts is not None)
        or (train_offsets is not None)
        or (test_offsets is not None)
    )
    if use_explicit_subset_percents:
        if train_pcts is None:
            train_pcts = [100.0] * len(paths)
        if test_pcts is None:
            test_pcts = [0.0] * len(paths)
        if train_offsets is None:
            train_offsets = [0.0] * len(paths)
        if test_offsets is None:
            test_offsets = [0.0] * len(paths)

    final_ds = None

    for i_path, (datapath, requested_split) in enumerate(zip(paths, all_splits)):
        split = _resolve_split(datapath, requested_split, get_test_subset)

        # 1) Load raw dataset
        if _is_local_saved_dataset(datapath):
            ds = load_from_disk(datapath)
        else:
            repo_id, config = _parse_hf_id(datapath)
            config = _default_hf_config(repo_id, config)
            cache_key = repo_id if config is None else f"{repo_id}:{config}"
            raw_local_path = cache_root / _safe_dirname(cache_key) / split

            if _is_local_saved_dataset(str(raw_local_path)):
                ds = load_from_disk(str(raw_local_path))
            else:
                if config is None:
                    ds_hf = load_dataset(repo_id, split=split)
                else:
                    ds_hf = load_dataset(repo_id, config, split=split)

                raw_local_path.mkdir(parents=True, exist_ok=True)
                ds_hf.save_to_disk(str(raw_local_path))
                ds = load_from_disk(str(raw_local_path))

        # 2) Shuffle then split
        ds = ds.shuffle(seed=seed)

        use_native_split = _use_native_train_test_split(datapath, requested_split)
        train_pct_i = train_pcts[i_path] if use_explicit_subset_percents else None
        test_pct_i = test_pcts[i_path] if use_explicit_subset_percents else None
        train_offset_i = train_offsets[i_path] if use_explicit_subset_percents else None
        test_offset_i = test_offsets[i_path] if use_explicit_subset_percents else None
        think_drop_prob_i = 0.0 if think_drop_pcts is None else (float(think_drop_pcts[i_path]) / 100.0)

        if use_explicit_subset_percents:
            n_total = len(ds)
            if use_native_split:
                pct = float(test_pct_i if get_test_subset else train_pct_i)
                offset = float(test_offset_i if get_test_subset else train_offset_i)
                if offset + pct > 100.0 + 1e-9:
                    raise ValueError(
                        f"subset offset + percent exceeds 100 for {datapath} "
                        f"(offset={offset}, pct={pct})."
                    )
                start = int(n_total * (offset / 100.0))
                n_take = int(n_total * (pct / 100.0))
                end = start + n_take
                ds1 = ds.select(range(start, end)) if end > start else ds.select([])
                print(
                    f"dataset rank: native {'TEST' if get_test_subset else 'TRAIN'} split "
                    f"taking {pct:.3f}% from offset {offset:.3f}% => {len(ds1)}/{n_total}"
                )
            else:
                trp = float(train_pct_i)
                tep = float(test_pct_i)
                tro = float(train_offset_i)
                teo = float(test_offset_i)
                n_test = int(n_total * (tep / 100.0))
                n_train = int(n_total * (trp / 100.0))
                n_test_offset = int(n_total * (teo / 100.0))
                n_train_offset = int(n_total * (tro / 100.0))
                test_start = n_test_offset
                test_end = test_start + n_test
                train_start = n_test + n_train_offset
                train_end = train_start + n_train
                if teo + tep > 100.0 + 1e-9:
                    raise ValueError(
                        f"test_subset_offsets + test_subset_percents exceeds 100 for {datapath} "
                        f"(offset={teo}, test={tep})."
                    )
                if tep + tro + trp > 100.0 + 1e-9:
                    raise ValueError(
                        f"reserved test + train_subset_offsets + train_subset_percents exceeds 100 for {datapath} "
                        f"(test={tep}, offset={tro}, train={trp})."
                    )
                if train_start < test_end:
                    raise ValueError(
                        f"train subset overlaps test subset for {datapath} "
                        f"(test_end_idx={test_end}, train_start_idx={train_start})."
                    )
                if get_test_subset:
                    ds1 = ds.select(range(test_start, test_end)) if test_end > test_start else ds.select([])
                    print(
                        f"dataset rank: disjoint TEST subset taking {tep:.3f}% "
                        f"from offset {teo:.3f}% => {len(ds1)}/{n_total}"
                    )
                else:
                    ds1 = ds.select(range(train_start, train_end)) if train_end > train_start else ds.select([])
                    print(
                        f"dataset rank: disjoint TRAIN subset taking {trp:.3f}% "
                        f"from offset {tro:.3f}% => {len(ds1)}/{n_total} "
                        f"(after reserving {tep:.3f}% for test)"
                    )
        elif test_split_ratio > 0 and len(ds) > 1 and not use_native_split:
            split_dict = ds.train_test_split(test_size=test_split_ratio, seed=seed)
            ds1 = split_dict["test"] if get_test_subset else split_dict["train"]
            print(
                f"dataset rank: returning {'TEST' if get_test_subset else 'TRAIN'} split ({len(ds1)} examples)"
            )
        else:
            ds1 = ds
            if use_native_split:
                print(
                    f"dataset rank: returning native {'TEST' if get_test_subset else 'TRAIN'} split ({len(ds1)} examples)"
                )
            else:
                print(f"dataset rank: returning FULL dataset ({len(ds1)} examples)")

        # Handle empty split before any processed cache load.
        if len(ds1) == 0:
            ds1_proc = _empty_processed_dataset()
            if final_ds is None:
                final_ds = ds1_proc
            else:
                final_ds = concatenate_datasets([final_ds, ds1_proc])
            continue

        original_columns = ds1.column_names

        # 3) Processed caching
        processed_root = cache_root / "processed"
        processed_root.mkdir(parents=True, exist_ok=True)

        tok_id = getattr(tokenizer, "name_or_path", "unknown_tokenizer")
        proc_key = _safe_dirname(
            f"{datapath}:{split}:req{requested_split}:{tok_id}:max{max_len}:tgt{target_len}:"
            f"seed{seed}:short{short_target_mode}:stripnemo{int(strip_nemotron_math_prompt_prefix)}:tsr{test_split_ratio}:"
            f"thinkdrop{think_drop_prob_i}:"
            f"trpct{train_pct_i if train_pct_i is not None else 'na'}:"
            f"tepct{test_pct_i if test_pct_i is not None else 'na'}:"
            f"troff{train_offset_i if train_offset_i is not None else 'na'}:"
            f"teoff{test_offset_i if test_offset_i is not None else 'na'}:v5"
        )
        processed_path = processed_root / proc_key / ("test" if get_test_subset else "train")

        ds1_proc = None
        if _is_local_saved_dataset(str(processed_path)):
            try:
                ds1_proc = load_from_disk(str(processed_path))
            except Exception as e:
                print(f"[WARN] Failed to load processed dataset cache at {processed_path}: {e}")
                print("[WARN] Removing broken cache and rebuilding...")
                shutil.rmtree(processed_path, ignore_errors=True)
                ds1_proc = None

        if ds1_proc is None:
            preprocess_fn = _make_preprocess_fn(datapath, requested_split, think_drop_prob_i)
            ds1_proc = ds1.map(
                preprocess_fn,
                batched=True,
                num_proc=num_proc,
                remove_columns=original_columns,
                load_from_cache_file=False,
            )
            ds1_proc.save_to_disk(str(processed_path))

        ds1_proc.set_format(type="torch")

        if final_ds is None:
            final_ds = ds1_proc
        else:
            final_ds = concatenate_datasets([final_ds, ds1_proc])

    if final_ds is None:
        final_ds = _empty_processed_dataset()

    final_ds.set_format(type="torch")
    return final_ds

def build_block_attention_mask(
    max_length: int,
    inp_len: int,
    prefix_len: int,
    window_len: int,
    mask_len: int,
    block_size: int,
    device=None,
) -> torch.BoolTensor:
    """Dense allow-mask for the T3 4-region layout: [ inp | prefix | window | mask ].

    Returns a `[max_length, max_length]` bool tensor where `True` means
    "query position can attend to key position". Source of truth for the
    pattern; consulted by `tests/test_attn_equivalence.py` and the docstring
    in `model/attention/flex_block_mask.py`. Allow rules:

        inp     -> inp                                      (bidirectional)
        prefix  -> inp ∪ prefix[0..b]                       (blockwise causal)
        window  -> inp ∪ prefix ∪ window[0..b]              (same shape as prefix)
        mask    -> inp ∪ prefix ∪ window[0..b-1] ∪ mask[b]  (sees prior windows
                                                             and own block; not
                                                             its own label window[b])
    """
    device = device or "cpu"
    m = torch.zeros((max_length, max_length), dtype=torch.bool, device=device)

    o_inp = 0
    o_pre = o_inp + inp_len
    o_win = o_pre + prefix_len
    o_msk = o_win + window_len

    def allow(r0, r1, c0, c1):
        if r1 > r0 and c1 > c0:
            m[r0:r1, c0:c1] = True

    # 1) inp -> inp
    allow(o_inp, o_inp + inp_len, o_inp, o_inp + inp_len)

    # 2) prefix block b -> inp ∪ prefix[0..b]
    assert prefix_len % block_size == 0
    for b in range(prefix_len // block_size):
        r0 = o_pre + b * block_size
        r1 = o_pre + (b + 1) * block_size
        allow(r0, r1, o_inp, o_pre + (b + 1) * block_size)

    # 3) window block b -> inp ∪ prefix ∪ window[0..b]
    assert window_len % block_size == 0
    for b in range(window_len // block_size):
        r0 = o_win + b * block_size
        r1 = o_win + (b + 1) * block_size
        allow(r0, r1, o_inp, o_win + (b + 1) * block_size)

    # 4) mask block b -> inp ∪ prefix ∪ window[0..b-1] ∪ mask[b]
    assert mask_len % block_size == 0
    for b in range(mask_len // block_size):
        r0 = o_msk + b * block_size
        r1 = o_msk + (b + 1) * block_size
        allow(r0, r1, o_inp, o_win + b * block_size)
        allow(r0, r1, o_msk + b * block_size, o_msk + (b + 1) * block_size)

    return m


class DataCollatorWithPaddingV2:
    """Collates per-sample sequences into the T3 4-region batch layout.

    Output schema (consumed by `train_h200.py` and the FlexAttention BlockMask
    cache key in `get_block_mask_for_batch`):

        input_ids       [B, max_length]   long
        position_ids    [B, max_length]   long  (window/mask blocks share
                                                 positions with the prefix)
        attention_mask  [B, max_length]   bool  (1 = real token, 0 = pad)
        loss_mask       [B, max_length]   long  (1 = mask region, 0 elsewhere)
        target          [B, total_length] long  (the window we're denoising
                                                 toward; this *is* the label)
        inp_len/prefix_len/window_len/mask_len   [B] int32
        block_size, max_length                    Python ints (cache-key inputs)
    """

    def __init__(
        self,
        block_size: int = 32,
        block_num: int = 8,
        mask_token_id: int = 126336,
        pad_token_id: int = 126081,
        eos_token_id: int = 126348,
        start_end_ratio: float = 0.2,
    ):
        self.block_size = block_size
        self.block_num = block_num
        self.total_length = self.block_size * self.block_num
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id if eos_token_id is not None else pad_token_id
        # `start_end_ratio` controls the three-way roll on the prefix start:
        # half of `start_end_ratio` of samples start at 0 (beginning of target),
        # half end-aligned (close to the tail), the rest uniform random.
        self.start_end_ratio = start_end_ratio

    def _sample_block_aligned_starts(
        self,
        target_lens: List[int],
        device: torch.device,
    ) -> torch.LongTensor:
        """Three-way roll on the prefix start, then block-align.

        Returns a `[B]` long tensor where `starts[i]` is the prefix length
        (in tokens) for sample `i`, guaranteed to be a multiple of
        `block_size` and in `[0, target_lens[i])`.
        """
        B = len(target_lens)
        total = self.total_length
        bs = self.block_size

        # Two upper bounds on the start index:
        #   normal_max: leaves room for the full target window after the prefix.
        #   end_max:   end-aligned, leaves only 3/4 of the window — used by the
        #              "end" branch so the model occasionally sees mostly-decoded
        #              targets near the tail.
        normal_max = torch.tensor(
            [max(t - total + 1, 1) for t in target_lens],
            device=device, dtype=torch.long,
        )
        end_max = torch.tensor(
            [max(t - (total * 3) // 4 + 1, 1) for t in target_lens],
            device=device, dtype=torch.long,
        )

        roll = torch.rand(B, device=device)
        p = self.start_end_ratio / 2.0
        is_start = roll < p
        is_end = roll > (1.0 - p)
        is_random = ~(is_start | is_end)

        starts = torch.zeros(B, device=device, dtype=torch.long)
        starts[is_end] = end_max[is_end] - 1
        if is_random.any():
            u = torch.rand(int(is_random.sum()), device=device)
            starts[is_random] = (u * normal_max[is_random].float()).long()

        return (starts // bs) * bs

    def _build_one_sample(
        self,
        inp: torch.Tensor,
        target: torch.Tensor,
        start: int,
        *,
        out_input_ids: torch.Tensor,       # [max_length] slice
        out_position_ids: torch.Tensor,    # [max_length] slice
        out_attention_mask: torch.Tensor,  # [max_length] slice
        out_loss_mask: torch.Tensor,       # [max_length] slice
        out_target: torch.Tensor,          # [total_length] slice
        mask_tokens: torch.Tensor,         # [total_length], cached on device
        device: torch.device,
        dtype_ids: torch.dtype,
    ) -> Tuple[int, int, int, int]:
        """Fill row `i`'s pre-allocated slots for one sample.

        Writes regions directly into the output slices — no intermediate
        `torch.cat` of [inp | prefix | window | mask]. Returns
        `(inp_len, prefix_len, window_len, mask_len)` for the caller to record.
        """
        total = self.total_length
        bs = self.block_size

        inp_len = inp.shape[0]
        prefix_len = start
        window = target[start:start + total]

        # Pad short windows with EOS so every sample contributes a full block.
        if window.size(0) < total:
            eos_pad = torch.full(
                (total - window.size(0),), self.eos_token_id,
                dtype=target.dtype, device=device,
            )
            window = torch.cat([window, eos_pad], dim=0)

        # Region offsets in the output sequence.
        window_len = total - bs   # window region drops the last block
        mask_len = total          # mask region carries all `block_num` blocks
        o_inp = 0
        o_pre = o_inp + inp_len
        o_win = o_pre + prefix_len
        o_msk = o_win + window_len
        L = o_msk + mask_len

        # input_ids: write each region in place.
        out_input_ids[o_inp:o_pre] = inp
        out_input_ids[o_pre:o_win] = target[:prefix_len].to(dtype_ids)
        out_input_ids[o_win:o_msk] = window[:window_len]
        out_input_ids[o_msk:o_msk + mask_len] = mask_tokens[:mask_len]

        # position_ids: prefix uses standard 0..prefix_end; window block b and
        # mask block b share positions starting at prefix_end (so the model
        # treats window[b] and mask[b] as the same logical position).
        prefix_end = inp_len + prefix_len
        pos_dtype = out_position_ids.dtype
        out_position_ids[o_inp:o_win] = torch.arange(
            0, prefix_end, device=device, dtype=pos_dtype,
        )
        out_position_ids[o_win:o_msk] = torch.arange(
            prefix_end, prefix_end + window_len, device=device, dtype=pos_dtype,
        )
        out_position_ids[o_msk:o_msk + mask_len] = torch.arange(
            prefix_end, prefix_end + mask_len, device=device, dtype=pos_dtype,
        )

        out_attention_mask[:L] = 1
        out_loss_mask[L - mask_len:L] = 1
        out_target[:] = window

        return inp_len, prefix_len, window_len, mask_len

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        def _to_1d(x: torch.Tensor) -> torch.Tensor:
            if x.dim() == 2 and x.size(0) == 1:
                return x.squeeze(0)
            return x

        B = len(features)
        device = features[0]["input_ids"].device

        input_ids_list = [_to_1d(f["input_ids"]) for f in features]
        target_list    = [_to_1d(f["target"]) for f in features]
        target_lens    = [t.size(0) for t in target_list]

        starts = self._sample_block_aligned_starts(target_lens, device)
        starts_py = [int(s) for s in starts.tolist()]

        # Each sample contributes inp + prefix + (total - bs) + total tokens.
        seq_lens = [
            input_ids_list[i].size(0) + starts_py[i] + 2 * self.total_length - self.block_size
            for i in range(B)
        ]
        max_length = max(seq_lens)

        # Pre-allocate batch tensors. The unused tail of each row stays at the
        # initial fill value (pad_token_id / 0) — `attention_mask` marks the
        # real region for downstream consumers.
        dtype_ids = input_ids_list[0].dtype
        batch_input_ids       = torch.full((B, max_length), self.pad_token_id, dtype=dtype_ids,    device=device)
        batch_position_ids    = torch.zeros((B, max_length),                    dtype=dtype_ids,    device=device)
        batch_attention_mask  = torch.zeros((B, max_length),                    dtype=torch.bool,   device=device)
        batch_loss_mask       = torch.zeros((B, max_length),                    dtype=torch.long,   device=device)
        batch_target          = torch.empty((B, self.total_length),             dtype=target_list[0].dtype, device=device)

        batch_inp_len    = torch.zeros((B,), dtype=torch.int32, device=device)
        batch_prefix_len = torch.zeros((B,), dtype=torch.int32, device=device)
        batch_window_len = torch.zeros((B,), dtype=torch.int32, device=device)
        batch_mask_len   = torch.zeros((B,), dtype=torch.int32, device=device)

        mask_tokens = torch.full(
            (self.total_length,), self.mask_token_id,
            dtype=dtype_ids, device=device,
        )

        for i in range(B):
            inp_len, prefix_len, window_len, mask_len = self._build_one_sample(
                inp=input_ids_list[i],
                target=target_list[i],
                start=starts_py[i],
                out_input_ids=batch_input_ids[i],
                out_position_ids=batch_position_ids[i],
                out_attention_mask=batch_attention_mask[i],
                out_loss_mask=batch_loss_mask[i],
                out_target=batch_target[i],
                mask_tokens=mask_tokens,
                device=device,
                dtype_ids=dtype_ids,
            )
            batch_inp_len[i]    = inp_len
            batch_prefix_len[i] = prefix_len
            batch_window_len[i] = window_len
            batch_mask_len[i]   = mask_len

        return {
            "input_ids":      batch_input_ids,
            "position_ids":   batch_position_ids,
            "target":         batch_target,
            "attention_mask": batch_attention_mask,
            "loss_mask":      batch_loss_mask,
            # Per-sample T3 region offsets (FlexAttention BlockMask is built from these).
            "inp_len":        batch_inp_len,
            "prefix_len":     batch_prefix_len,
            "window_len":     batch_window_len,
            "mask_len":       batch_mask_len,
            "block_size":     self.block_size,
            "max_length":     max_length,
        }


