import re
import sys
import json
from pathlib import Path

import numpy as np

from postprocess_code import (
    TeeContext,
    _log_path,
    _stderr,
    clean_math_completion,
    get_raw_completion,
    is_humaneval_sample,
    read_jsonl,
    sample_task_id,
    score_humaneval,
    score_math,
    update_results_json,
    write_jsonl,
    write_text_report,
    extract_last_boxed_answer,
    normalize_newlines,
)


def is_gsm8k_sample(sample):
    doc = sample.get("doc", {})
    if not isinstance(doc, dict):
        return False
    if "question" not in doc or "answer" not in doc:
        return False
    target = str(sample.get("target", ""))
    return "####" in target


def select_rows_for_scoring(data):
    filters = {sample.get("filter") for sample in data}
    if "flexible-extract" in filters:
        return [sample for sample in data if sample.get("filter") == "flexible-extract"]
    return data


def _extract_hash_answer(text):
    matches = re.findall(r"(?m)^####\s*(.+?)\s*$", text)
    if matches:
        return matches[-1].strip()
    return None


def _extract_answer_phrase(text):
    matches = re.findall(
        r"(?:final answer|the answer|answer)(?:\s+is|\s*=|\s*:)?\s*(.+)",
        text,
        flags=re.IGNORECASE,
    )
    if matches:
        return matches[-1].splitlines()[0].strip().rstrip(".,")
    return None


def _extract_last_number(text):
    matches = re.findall(r"-?\$?\d[\d,]*(?:\.\d+)?", text)
    if matches:
        return matches[-1]
    return None


def _gsm8k_clean_for_filter(sample):
    completion = normalize_newlines(get_raw_completion(sample)).strip()
    boxed = extract_last_boxed_answer(completion)
    if boxed:
        return boxed
    return completion


def _gsm8k_flexible_extract(text):
    matches = re.findall(r"(-?[$0-9.,]{2,})|(-?[0-9]+)", text)
    if not matches:
        return "[invalid]"
    match = matches[-1]
    if isinstance(match, tuple):
        non_empty = [value for value in match if value]
        if not non_empty:
            return "[invalid]"
        match = non_empty[0]
    return match.strip()


def _gsm8k_exact_match(prediction, target):
    predictions = np.array([prediction])
    references = np.array([str(target)])

    for pattern in (",", r"\$", r"(?s).*#### ", r"\.$"):
        predictions = np.array([re.sub(pattern, "", x) for x in predictions])
        references = np.array([re.sub(pattern, "", x) for x in references])

    predictions = np.char.lower(predictions)
    references = np.char.lower(references)
    return float(bool((predictions == references)[0]))


def _normalize_gsm8k_value(text):
    if text is None:
        return ""

    value = normalize_newlines(str(text)).strip()
    if not value:
        return ""

    for extractor in (_extract_hash_answer, extract_last_boxed_answer, _extract_answer_phrase):
        candidate = extractor(value)
        if candidate:
            value = candidate
            break

    value = value.replace("\\(", "").replace("\\)", "")
    value = value.replace("\\[", "").replace("\\]", "")
    value = value.replace("$", "").replace(",", "").strip()
    value = re.sub(r"\s+", " ", value)

    numeric = _extract_last_number(value)
    if numeric is not None:
        value = numeric.replace("$", "").replace(",", "")

    value = value.strip().rstrip(".")
    return value


def clean_gsm8k_completion(sample):
    completion = _gsm8k_clean_for_filter(sample)
    return _normalize_gsm8k_value(completion)


def score_gsm8k(data):
    predictions = [clean_gsm8k_completion(sample) for sample in data]
    filtered_predictions = [_gsm8k_flexible_extract(_gsm8k_clean_for_filter(sample)) for sample in data]
    per_sample_scores = [
        _gsm8k_exact_match(prediction, sample.get("target", ""))
        for prediction, sample in zip(filtered_predictions, data)
    ]
    return predictions, per_sample_scores, "gsm8k", "exact_match,postprocess"


def resolve_samples_file(file_path: Path) -> Path:
    if file_path.name.startswith("samples_") and file_path.suffix == ".jsonl":
        return file_path

    if file_path.name.startswith("results_") and file_path.suffix == ".json":
        with open(file_path, "r", encoding="utf-8") as file:
            results = json.load(file)

        config = results.get("config", {})
        tasks = config.get("tasks") or []
        if isinstance(tasks, str):
            tasks = [tasks]
        if not tasks:
            configs = results.get("configs")
            if isinstance(configs, dict) and configs:
                tasks = list(configs.keys())
        if not tasks:
            result_tasks = results.get("results")
            if isinstance(result_tasks, dict) and result_tasks:
                tasks = list(result_tasks.keys())
        if not tasks:
            raise ValueError(f"Could not infer task name from results file: {file_path}")

        task_name = str(tasks[0]).split(",")[0]
        timestamp = file_path.stem.removeprefix("results_")
        candidate = file_path.with_name(f"samples_{task_name}_{timestamp}.jsonl")
        if candidate.exists():
            return candidate

        matches = sorted(file_path.parent.glob(f"samples_{task_name}_*.jsonl"))
        if len(matches) == 1:
            return matches[0]

        raise FileNotFoundError(
            f"Could not find matching samples jsonl for results file {file_path}. "
            f"Tried {candidate.name} and found {len(matches)} task-matched candidates."
        )

    return file_path


def main():
    input_path = Path(sys.argv[1]).expanduser().resolve()
    file_path = resolve_samples_file(input_path)
    log_path = _log_path(input_path)

    with TeeContext(log_path):
        print(f"log path: {log_path}")
        if file_path != input_path:
            print(f"resolved samples path: {file_path}")
        data = read_jsonl(file_path)
        data = select_rows_for_scoring(data)

        if not data:
            raise ValueError(f"no samples found in {file_path}")

        if all(is_humaneval_sample(sample) for sample in data):
            predictions, per_sample_scores, task_name, metric_name = score_humaneval(data)
        elif all(is_gsm8k_sample(sample) for sample in data):
            predictions, per_sample_scores, task_name, metric_name = score_gsm8k(data)
        else:
            predictions, per_sample_scores, task_name, metric_name = score_math(data, file_path)

        mean_score = sum(per_sample_scores) / len(per_sample_scores)
        stderr = _stderr(per_sample_scores)
        print(mean_score)

        res = [
            {
                "task_id": sample_task_id(sample),
                "raw_completion": get_raw_completion(sample),
                "cleaned_completion": pred[0] if isinstance(pred, list) else pred,
                "score": score,
            }
            for sample, pred, score in zip(data, predictions, per_sample_scores)
        ]

        for item in res:
            print("=" * 80)
            print(item["task_id"], item["score"])
            print("[RAW]")
            print(item["raw_completion"])
            print("[CLEANED]")
            print(item["cleaned_completion"])

        write_jsonl(res, str(file_path) + ".cleaned")
        write_text_report(res, str(file_path) + ".txt")
        update_results_json(file_path, task_name, metric_name, mean_score, stderr)


if __name__ == "__main__":
    main()
