# ASTEfficiencyChecker

Aquina — Dev B, model **B2** on the [split plan](aquina-dev-split-plan.md): a
rule-based (no ML) tool that parses submitted code, estimates its time/space
complexity from AST structure, and compares that estimate against the known-optimal
complexity for the matching problem.

No backend, no networking, no deployment — this is the standalone model only,
built to be imported/called by whatever wraps it later.

## Status

| Piece | Status |
|---|---|
| IR layer, generic complexity rules | ✅ built + tested |
| Python parser (stdlib `ast`) | ✅ built + tested, zero dependencies |
| C / C++ / Java / JS / TS / Rust / Go / C# parsers (tree-sitter) | ⚠️ built, **not yet tested** — no network access in the build environment to install tree-sitter grammars. Run against real files locally and report mismatches (see "Known gap" below). |
| 17 algorithm detectors | ✅ built + tested (Python) |
| Complexity estimator | ✅ built + tested |
| Dataset matcher (LeetCode/Codeforces/AtCoder lookup) | ✅ built + tested against a small sample dataset |
| Embedding-based semantic matching | interface only — plug in B3 (Sentence Embedding model) when it exists; falls back to bag-of-words similarity until then |
| Report generator (text + JSON) | ✅ built + tested |

## Quick start

```bash
pip install -r requirements.txt   # only needed for non-Python languages + running tests
python main.py languages
python main.py analyze tests/samples/sample_algorithms.py
python main.py analyze path/to/file.cpp --json
python main.py analyze path/to/project_dir --recursive
```

## Architecture

```
source file
    │
    ▼
parsers/*.py          language-specific adapter → engine/ir.py IRNode tree
    │
    ▼
engine/rules.py        generic complexity rules (cyclomatic complexity, nesting depth, loop nesting)
engine/detectors.py     algorithm/idiom pattern detectors (binary search, DP, DFS/BFS, ...)
    │
    ▼
engine/complexity.py    combines rule + detector signal into a time/space estimate
    │
    ▼
engine/matcher.py       matches against data/raw/*.jsonl problem datasets (via engine/embeddings.py)
engine/comparison.py    estimated vs known-optimal complexity → verdict
    │
    ▼
engine/report.py        explainable report (text or JSON)
```

The **IR (`engine/ir.py`)** is the one seam that keeps everything downstream
language-agnostic — every parser adapter's only job is producing `IRNode` trees;
nothing past that point ever looks at a language-specific AST again.

## Language support

- **Python** — parsed with the standard library `ast` module. No extra
  dependency, fully tested in this build.
- **C, C++, Java, JavaScript, TypeScript, Rust, Go, C#** — parsed via
  [tree-sitter](https://tree-sitter.github.io/tree-sitter/) grammars
  (`parsers/tree_sitter_parser.py`, driven by the `NODE_KIND_MAPS` table in
  that file). **This code could not be executed in the build environment**
  (no network access to `pip install tree-sitter-languages` or download
  grammars) — it was written directly against each grammar's published
  node-type reference, but treat it as unverified until you run it locally.

### Known gap — needs a local test pass

Run `python main.py analyze <file>` against a real file in each
non-Python language and compare the output to what you'd expect. The most
likely failure mode is a wrong or missing node-type string in
`NODE_KIND_MAPS` (in `parsers/tree_sitter_parser.py`) — e.g. a grammar using
a different node name than what's mapped, which shows up as `IRKind.UNKNOWN`
nodes silently not being recognized by rules/detectors (rather than a crash).
Send back the mismatches and they can be fixed in one pass.

## Detectors implemented

Binary Search, Two Pointer, Sliding Window, DFS (recursive), BFS
(queue-based), Memoization, Dynamic Programming (tabulation), HashMap/Set
usage, Heap/Priority Queue, Union-Find, Prefix Sum, Monotonic Stack,
Backtracking, Merge Sort, Quick Sort, Topological Sort (Kahn's), Dijkstra,
Floyd-Warshall (via triple-nested-loop shape).

**Not yet implemented** (structurally harder to detect reliably from AST
shape alone, or lower priority for a v1 — flagged here rather than shipped
as hollow always-miss stubs): Trie, Segment Tree, Fenwick Tree, Difference
Array, Monotonic Queue, Greedy (no consistent structural signature),
Bellman-Ford, Kruskal, Prim, Binary Lifting.

Every detector is conservative by design — it looks for structural
signatures (loop shape, call patterns, naming conventions, data-structure
usage) and requires `config.DETECTOR_CONFIDENCE_THRESHOLD` (default 0.55)
confidence before it's reported. False negatives (missing a real pattern)
are preferred over false positives (claiming a pattern that isn't there).

## Dataset matching

No datasets are bundled — there was no network access in the build
environment to scrape LeetCode/Codeforces/AtCoder. Point
`config.DATASET_PATHS` at your own exported JSONL files, one problem per
line:

```json
{"id": "leetcode-704", "source": "leetcode", "title": "Binary Search",
 "statement": "...", "tags": ["binary-search"],
 "optimal_time_complexity": "O(log n)", "optimal_space_complexity": "O(1)"}
```

A 3-entry sample is included at `data/raw/leetcode.jsonl` for smoke-testing;
`data/raw/codeforces.jsonl` and `data/raw/atcoder.jsonl` are left empty —
the tool degrades gracefully and reports which sources it found vs. skipped.

## Semantic matching / embeddings (B3 hook)

Per the split plan, the Sentence Embedding model is a separate model (B3),
not part of this one. `engine/embeddings.py` defines the plug-in point:

```python
from engine import embeddings
embeddings.set_embedder(your_model.encode)  # fn(text: str) -> list[float]
```

Until that's wired in, matching falls back to a deterministic bag-of-words
cosine similarity (no ML dependency) so the matcher is testable end-to-end
today.

## Testing

```bash
pytest tests/ -v
```

Tests cover the Python parser/IR conversion (including a nested-function
recursion-detection edge case that was caught and fixed during development),
detector pattern matching (including a false-positive regression test for
sliding-window on deeply nested loops), and the complexity comparison logic
(including a substring-matching bug — `O(n)` vs `O(n^2)` — that was caught
and fixed during development).

## Folder structure

```
ASTEfficiencyChecker/
├── data/
│   ├── raw/            # dataset JSONL files (leetcode.jsonl etc.)
│   └── processed/      # reserved for cached indices/embeddings
├── parsers/             # one adapter per language + shared base
├── engine/               # IR, rules, detectors, complexity, matcher, comparison, report
├── scripts/              # utility scripts
├── tests/                # pytest suite + sample source files
├── config.py
├── main.py
├── README.md
└── requirements.txt
```
