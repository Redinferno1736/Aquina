"""
Aquina A3 - Data Preparation
Loads CodeSearchNet (all 6 languages), labels each function with cyclomatic
complexity via `lizard`, and writes out a single labeled parquet file ready
for training.

Expected input layout (adjust CODESEARCHNET_ROOT if yours differs):
    CODESEARCHNET_ROOT/
        python/train.jsonl (or .jsonl.gz)
        java/train.jsonl
        javascript/train.jsonl
        php/train.jsonl
        ruby/train.jsonl
        go/train.jsonl

Each line in these files is a JSON object shaped like the sample you showed:
    {"repo", "path", "func_name", "original_string", "language", "code", ...}

Run:
    python prepare_data.py --root /path/to/codesearchnet --out ../data/labeled_functions.parquet
"""

import argparse
import gzip
import json
import os
from pathlib import Path

import lizard
import pandas as pd
from tqdm import tqdm

# Maps CodeSearchNet's language folder name -> a fake filename extension so
# lizard.analyze_source_code can infer the correct language parser.
LANG_TO_EXT = {
    "python": "py",
    "java": "java",
    "javascript": "js",
    "php": "php",
    "ruby": "rb",
    "go": "go",
}

# Fixed integer id per language - used later for the embedding lookup table.
# Keep this mapping identical between data prep, training, and inference.
LANG_TO_ID = {
    "python": 0,
    "java": 1,
    "javascript": 2,
    "php": 3,
    "ruby": 4,
    "go": 5,
}


def iter_jsonl(path: Path):
    """Yields dicts from a .jsonl or .jsonl.gz file."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def find_split_files(root: Path, language: str, split: str):
    """
    CodeSearchNet is often shipped as multiple sharded files, e.g.
    python/final/jsonl/train/python_train_0.jsonl.gz ... _13.jsonl.gz
    This walks the language folder and grabs every file matching the split.
    """
    lang_dir = root / language
    if not lang_dir.exists():
        return []
    matches = []
    for p in lang_dir.rglob("*"):
        if p.is_file() and split in p.name and (p.suffix in (".jsonl", ".gz")):
            matches.append(p)
    return sorted(matches)


def label_row(code: str, language: str):
    """
    Runs lizard on a single function's code string.
    Returns the cyclomatic complexity of the (first, ideally only) function
    lizard finds, or None if it can't parse anything usable.
    """
    ext = LANG_TO_EXT[language]
    if language == "php" and "<?php" not in code:
        code = "<?php\n" + code
    try:
        result = lizard.analyze_file.analyze_source_code(f"snippet.{ext}", code)
    except Exception:
        return None

    funcs = result.function_list
    if len(funcs) != 1:
        # Skip snippets where lizard finds zero or more than one function -
        # we want a clean one-function-per-row dataset.
        return None

    fn = funcs[0]
    return {
        "cyclomatic_complexity": fn.cyclomatic_complexity,
        "nloc": fn.nloc,
        "token_count": fn.token_count,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True,
                         help="Path to CodeSearchNet root folder")
    parser.add_argument("--split", type=str, default="train",
                         help="Which split to load: train/valid/test")
    parser.add_argument("--out", type=str, default="../data/labeled_functions.parquet")
    parser.add_argument("--max_per_language", type=int, default=60000,
                         help="Cap rows per language so no single language dominates")
    parser.add_argument("--max_code_chars", type=int, default=4000,
                         help="Skip functions longer than this many characters "
                              "(keeps things sane before tokenization truncation)")
    args = parser.parse_args()

    root = Path(args.root)
    all_rows = []

    for language in LANG_TO_ID:
        files = find_split_files(root, language, args.split)
        if not files:
            print(f"[skip] no files found for language={language} under {root/language}")
            continue

        kept = 0
        print(f"[{language}] found {len(files)} file(s)")
        for fp in files:
            for record in tqdm(iter_jsonl(fp), desc=f"{language}:{fp.name}"):
                if kept >= args.max_per_language:
                    break

                code = record.get("code") or record.get("original_string")
                if not code or len(code) > args.max_code_chars:
                    continue

                labels = label_row(code, language)
                if labels is None:
                    continue

                all_rows.append({
                    "code": code,
                    "language": language,
                    "language_id": LANG_TO_ID[language],
                    **labels,
                })
                kept += 1

            if kept >= args.max_per_language:
                break

        print(f"[{language}] kept {kept} labeled functions")

    df = pd.DataFrame(all_rows)
    print(f"\nTotal labeled rows: {len(df)}")
    print(df.groupby("language")["cyclomatic_complexity"].describe())

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
