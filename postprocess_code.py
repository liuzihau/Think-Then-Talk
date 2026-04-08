import json
import math
import os
import re
import sys
import textwrap
from pathlib import Path

os.environ["HF_ALLOW_CODE_EVAL"] = "1"

_PASS_AT_K = None


class TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)


class TeeContext:
    def __init__(self, log_path):
        self.log_path = log_path
        self.log_file = None
        self.stdout = None
        self.stderr = None

    def __enter__(self):
        Path(self.log_path).parent.mkdir(parents=True, exist_ok=True)
        self.log_file = open(self.log_path, "w", encoding="utf-8")
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        sys.stdout = TeeStream(self.stdout, self.log_file)
        sys.stderr = TeeStream(self.stderr, self.log_file)
        return self.log_path

    def __exit__(self, exc_type, exc, tb):
        sys.stdout.flush()
        sys.stderr.flush()
        sys.stdout = self.stdout
        sys.stderr = self.stderr
        self.log_file.close()


def _load_pass_at_k():
    global _PASS_AT_K
    if _PASS_AT_K is None:
        import evaluate as hf_evaluate

        _PASS_AT_K = hf_evaluate.load("code_eval")
    return _PASS_AT_K


def pass_at_1(references, predictions):
    return _load_pass_at_k().compute(
        references=references,
        predictions=predictions,
        k=[1],
    )[0]["pass@1"]


def read_jsonl(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            data.append(json.loads(line))
    return data


def write_jsonl(data, file_path):
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as file:
        for item in data:
            file.write(json.dumps(item) + "\n")


def write_text_report(items, file_path):
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as file:
        for item in items:
            file.write("=" * 80 + "\n")
            file.write(f"{item['task_id']} {item['score']}\n")
            file.write("[RAW]\n")
            file.write(item["raw_completion"])
            if not item["raw_completion"].endswith("\n"):
                file.write("\n")
            file.write("[CLEANED]\n")
            file.write(item["cleaned_completion"])
            if not item["cleaned_completion"].endswith("\n"):
                file.write("\n")
            file.write("\n")


def strip_fences(text):
    text = text.lstrip()
    if text.startswith("```python"):
        text = text[len("```python") :].lstrip()
    elif text.startswith("```"):
        text = text[len("```") :].lstrip()

    if "```" in text:
        text = text.split("```", 1)[0]
    return text.rstrip()


def extract_fenced_code(text):
    match = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def normalize_newlines(text):
    return text.replace("\r\n", "\n").replace("\r", "\n")


def get_raw_completion(sample):
    return sample["resps"][0][0]


def sample_task_id(sample):
    doc = sample.get("doc", {})
    if isinstance(doc, dict):
        task_id = doc.get("task_id")
        if task_id:
            return task_id
        problem = doc.get("problem")
        if problem:
            return problem
    return str(sample.get("doc_id", "unknown"))


def is_humaneval_sample(sample):
    doc = sample.get("doc", {})
    task_id = doc.get("task_id") if isinstance(doc, dict) else None
    return isinstance(task_id, str) and task_id.lower().startswith("humaneval")


def is_math_sample(sample):
    doc = sample.get("doc", {})
    return isinstance(doc, dict) and bool({"problem", "solution", "answer"}.intersection(doc.keys()))


def remove_repeated_prefix(prompt, completion):
    text = completion.lstrip()
    if text.startswith(prompt):
        return text[len(prompt) :].lstrip()

    prompt_lines = [line.rstrip() for line in prompt.strip().splitlines() if line.strip()]
    text_lines = text.splitlines()
    while prompt_lines and text_lines and prompt_lines[0] == text_lines[0].rstrip():
        prompt_lines.pop(0)
        text_lines.pop(0)
    return "\n".join(text_lines).lstrip()


def _extract_target_function_text(text, entry_point):
    pattern = re.compile(rf"^(?:async\s+def|def)\s+{re.escape(entry_point)}\s*\(", re.MULTILINE)
    match = pattern.search(text)
    if not match:
        return None

    lines = text[match.start() :].splitlines()
    if not lines:
        return None

    captured = [lines[0]]
    body_indent = None
    seen_body = False

    for line in lines[1:]:
        stripped = line.strip()
        indent = len(line) - len(line.lstrip(" "))

        if stripped:
            if body_indent is None and indent > 0:
                body_indent = indent
            if body_indent is not None and indent < body_indent:
                break
            if body_indent is None and indent == 0:
                break
            seen_body = True
        elif seen_body:
            captured.append(line)
            continue

        captured.append(line)

    return "\n".join(captured).rstrip()


def _extract_body_from_function_text(function_text):
    lines = function_text.splitlines()
    if len(lines) <= 1:
        return ""

    body_lines = lines[1:]
    non_empty = [line for line in body_lines if line.strip()]
    if not non_empty:
        return ""

    min_indent = min(len(line) - len(line.lstrip(" ")) for line in non_empty)
    body = "\n".join(
        line[min_indent:] if len(line) >= min_indent else line for line in body_lines
    ).rstrip()

    stripped = body.lstrip()
    if stripped.startswith('"""'):
        end = stripped.find('"""', 3)
        if end != -1:
            body = stripped[end + 3 :].lstrip("\n")
    elif stripped.startswith("'''"):
        end = stripped.find("'''", 3)
        if end != -1:
            body = stripped[end + 3 :].lstrip("\n")

    return body.rstrip()


def _strip_trailing_top_level_blocks(text):
    stop_markers = (
        "\nclass ",
        "\ndef ",
        "\nasync def ",
        "\nif ",
        "\nprint",
        "\nMETADATA",
        "\ncheck(",
        "\nassert ",
    )
    cut = len(text)
    for marker in stop_markers:
        idx = text.find(marker)
        if idx != -1:
            cut = min(cut, idx)
    return text[:cut].rstrip()


def _indent_body(body):
    body = textwrap.dedent(body).strip("\n")
    if not body:
        return ""
    return "\n".join(("    " + line) if line.strip() else "" for line in body.splitlines())


def _looks_like_code_line(line):
    stripped = line.strip()
    if not stripped:
        return False
    return (
        line.startswith((" ", "\t"))
        or stripped.startswith(
            (
                "return",
                "for ",
                "while ",
                "if ",
                "elif ",
                "else:",
                "try:",
                "except",
                "with ",
                "raise ",
                "assert ",
                "yield ",
                "pass",
                "break",
                "continue",
                "#",
                "from ",
                "import ",
            )
        )
        or "=" in stripped
    )


def extract_humaneval_body(prompt, completion, entry_point):
    raw_text = normalize_newlines(completion)
    fenced = extract_fenced_code(raw_text)
    text = fenced if fenced is not None else strip_fences(raw_text)
    text = remove_repeated_prefix(prompt, text)

    function_text = _extract_target_function_text(text, entry_point)
    if function_text:
        body = _extract_body_from_function_text(function_text)
        if body.strip():
            return body.rstrip()

    if fenced is not None and "def " in text:
        return ""

    text = _strip_trailing_top_level_blocks(text).strip("\n")
    if not text.strip():
        return ""

    lines = text.splitlines()
    for idx, line in enumerate(lines):
        if _looks_like_code_line(line):
            candidate = "\n".join(lines[idx:]).rstrip()
            if candidate:
                return textwrap.dedent(candidate).rstrip()
            break

    return ""


def clean_humaneval_completion(sample):
    prompt = sample["doc"]["prompt"]
    completion = get_raw_completion(sample)
    entry_point = sample["doc"]["entry_point"]
    body = _indent_body(extract_humaneval_body(prompt, completion, entry_point))
    return prompt + body + ("\n" if body and not body.endswith("\n") else "")


def extract_last_boxed_answer(text):
    boxed_matches = list(re.finditer(r"\\boxed\s*(\{)?", text))
    if not boxed_matches:
        return None

    match = boxed_matches[-1]
    start = match.end()
    if match.group(1) == "{":
        depth = 1
        idx = start
        while idx < len(text):
            char = text[idx]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start:idx].strip()
            idx += 1
        return text[match.start() :].strip()

    line = text[start:].lstrip()
    if not line:
        return None
    return line.splitlines()[0].strip()


def clean_math_completion(sample):
    completion = normalize_newlines(get_raw_completion(sample)).strip()
    boxed = extract_last_boxed_answer(completion)
    if boxed:
        return boxed

    answer_matches = re.findall(
        r"(?:final answer|answer)(?:\s+is|\s*=|\s*:)?\s*(.+)",
        completion,
        flags=re.IGNORECASE,
    )
    if answer_matches:
        candidate = answer_matches[-1].splitlines()[0].strip().rstrip(".,")
        if candidate:
            return candidate

    tail_lines = [line.strip() for line in completion.splitlines() if line.strip()]
    if tail_lines:
        last_line = tail_lines[-1].rstrip(".,")
        if last_line:
            return last_line
    return completion



def remove_right_units(string):
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        if len(splits) == 2:
            return splits[0]
    return string


def fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split and split[0] != "{":
            new_string += "\\sqrt{" + split[0] + "}" + split[1:]
        else:
            new_string += "\\sqrt" + split
    return new_string


def fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        for substr in substrs[1:]:
            new_str += "\\frac"
            if not substr:
                return string
            if substr[0] == "{":
                new_str += substr
            else:
                if len(substr) < 2:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    post_substr = substr[2:] if len(substr) > 2 else ""
                    new_str += "{" + a + "}{" + b + "}" + post_substr
                else:
                    post_substr = substr[2:] if len(substr) > 2 else ""
                    new_str += "{" + a + "}" + b + post_substr
    return new_str


def fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a, b = string.split("/")
    try:
        a_int = int(a)
        b_int = int(b)
        if string == f"{a_int}/{b_int}":
            return f"\\frac{{{a_int}}}{{{b_int}}}"
    except Exception:
        pass
    return string



def remove_right_units(string):
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        if len(splits) == 2:
            return splits[0]
    return string


def fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split and split[0] != "{":
            new_string += "\\sqrt{" + split[0] + "}" + split[1:]
        else:
            new_string += "\\sqrt" + split
    return new_string


def fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        for substr in substrs[1:]:
            new_str += "\\frac"
            if not substr:
                return string
            if substr[0] == "{":
                new_str += substr
            else:
                if len(substr) < 2:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    post_substr = substr[2:] if len(substr) > 2 else ""
                    new_str += "{" + a + "}{" + b + "}" + post_substr
                else:
                    post_substr = substr[2:] if len(substr) > 2 else ""
                    new_str += "{" + a + "}" + b + post_substr
    return new_str


def fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a, b = string.split("/")
    try:
        a_int = int(a)
        b_int = int(b)
        if string == f"{a_int}/{b_int}":
            return f"\\frac{{{a_int}}}{{{b_int}}}"
    except Exception:
        pass
    return string


def normalize_math_answer(text):
    string = normalize_newlines(str(text))
    string = string.replace("\n", "")
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("tfrac", "frac").replace("dfrac", "frac")
    string = string.replace("\\left", "").replace("\\right", "")
    string = string.replace("^{\\circ}", "").replace("^\\circ", "")
    string = string.replace("\\$", "").replace("$", "")
    string = remove_right_units(string)
    string = string.replace("\\%", "").replace("%", "")
    string = string.replace(" .", " 0.").replace("{.", "{0.")
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string
    if len(string.split("=")) == 2 and len(string.split("=")[0]) <= 2:
        string = string.split("=")[1]
    string = fix_sqrt(string)
    string = string.replace(" ", "")
    string = fix_fracs(string)
    if string == "0.5":
        string = "\\frac{1}{2}"
    string = fix_a_slash_b(string)
    return string


def infer_math_task_name(sample_path):
    name = Path(sample_path).name
    if name.startswith("samples_") and name.endswith(".jsonl"):
        stem = name[len("samples_") : -len(".jsonl")]
        parts = stem.rsplit("_", 1)
        if len(parts) == 2:
            return parts[0]
    return "hendrycks_math"


def _results_json_path(sample_path):
    sample_path = Path(sample_path)
    name = sample_path.name
    if not name.startswith("samples_") or not name.endswith(".jsonl"):
        return None
    timestamp = name.rsplit("_", 1)[-1]
    result_name = f"results_{timestamp}".replace(".jsonl", ".json")
    return sample_path.with_name(result_name)


def _log_path(sample_path):
    return Path(str(sample_path) + ".log")


def _stderr(scores):
    n = len(scores)
    if n <= 1:
        return 0.0
    mean = sum(scores) / n
    return math.sqrt(mean * (1.0 - mean) / n)


def update_results_json(sample_path, task_name, metric_name, mean_score, stderr):
    result_path = _results_json_path(sample_path)
    print(f"sample path: {sample_path}")
    print(f"expected result path: {result_path}")
    print(f"result path exists: {bool(result_path and result_path.exists())}")

    if result_path is None or not result_path.exists():
        print(f"warning: no matching results json found for {sample_path}", file=sys.stderr)
        return

    with open(result_path, "r", encoding="utf-8") as file:
        obj = json.load(file)

    results = obj.setdefault("results", {})
    task_results = results.setdefault(task_name, {})
    task_results[metric_name] = mean_score
    task_results[f"{metric_name}_stderr"] = stderr

    if task_name.startswith("hendrycks_math_"):
        subtask_scores = [
            value.get(metric_name)
            for key, value in results.items()
            if key.startswith("hendrycks_math_") and metric_name in value
        ]
        subtask_scores = [score for score in subtask_scores if score is not None]
        if subtask_scores:
            root_results = results.setdefault("hendrycks_math", {})
            root_results[metric_name] = sum(subtask_scores) / len(subtask_scores)

    with open(result_path, "w", encoding="utf-8") as file:
        json.dump(obj, file, indent=2)
        file.write("\n")

    print(f"updated {result_path}")


def score_humaneval(data):
    references = [sample["target"] for sample in data]
    predictions = [[clean_humaneval_completion(sample)] for sample in data]
    per_sample_scores = [
        pass_at_1([reference], [prediction])
        for reference, prediction in zip(references, predictions)
    ]
    return predictions, per_sample_scores, "humaneval", "pass@1,create_test"




def score_math(data, sample_path):
    predictions = [clean_math_completion(sample) for sample in data]
    targets = [str(sample["target"]).strip() for sample in data]
    per_sample_scores = [
        float(normalize_math_answer(prediction) == normalize_math_answer(target))
        for prediction, target in zip(predictions, targets)
    ]
    return predictions, per_sample_scores, infer_math_task_name(sample_path), "exact_match,postprocess"


def main():
    file_path = Path(sys.argv[1])
    log_path = _log_path(file_path)

    with TeeContext(log_path):
        print(f"log path: {log_path}")
        data = read_jsonl(file_path)

        if not data:
            raise ValueError(f"no samples found in {file_path}")

        if all(is_humaneval_sample(sample) for sample in data):
            predictions, per_sample_scores, task_name, metric_name = score_humaneval(data)
        elif all(is_math_sample(sample) for sample in data):
            predictions, per_sample_scores, task_name, metric_name = score_math(data, file_path)
        else:
            raise ValueError(
                "mixed or unsupported sample types; expected all HumanEval or all math samples"
            )

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
