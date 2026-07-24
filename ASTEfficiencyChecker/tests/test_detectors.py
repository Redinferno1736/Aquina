from parsers.python_parser import PythonParser
from engine.detectors import run_detectors


def analyze(src):
    prog = PythonParser().parse(src, "<test>")
    return {f.name: run_detectors(f.node) for f in prog.functions}


def test_binary_search_detected():
    src = (
        "def bs(arr, target):\n"
        "    lo, hi = 0, len(arr) - 1\n"
        "    while lo <= hi:\n"
        "        mid = (lo + hi) // 2\n"
        "        if arr[mid] == target:\n"
        "            return mid\n"
        "        elif arr[mid] < target:\n"
        "            lo = mid + 1\n"
        "        else:\n"
        "            hi = mid - 1\n"
        "    return -1\n"
    )
    results = analyze(src)["bs"]
    names = {r.algorithm for r in results}
    assert "Binary Search" in names


def test_hashmap_detected():
    src = (
        "def two_sum(nums, target):\n"
        "    seen = {}\n"
        "    for i, x in enumerate(nums):\n"
        "        if target - x in seen:\n"
        "            return [seen[target - x], i]\n"
        "        seen[x] = i\n"
        "    return []\n"
    )
    results = analyze(src)["two_sum"]
    names = {r.algorithm for r in results}
    assert "HashMap/Set-based lookup" in names


def test_no_false_positive_on_trivial_function():
    src = "def add(a, b):\n    return a + b\n"
    results = analyze(src)["add"]
    assert results == []


def test_triple_nested_loop_not_misclassified_as_sliding_window():
    src = (
        "def f(m):\n"
        "    n = len(m)\n"
        "    total = 0\n"
        "    for i in range(n):\n"
        "        for j in range(n):\n"
        "            for k in range(n):\n"
        "                total += m[i][j] * m[j][k]\n"
        "    return total\n"
    )
    results = analyze(src)["f"]
    names = {r.algorithm for r in results}
    assert "Sliding Window" not in names


def test_memoized_fib_detected():
    src = (
        "def fib(n, memo={}):\n"
        "    if n in memo:\n"
        "        return memo[n]\n"
        "    if n <= 1:\n"
        "        return n\n"
        "    memo[n] = fib(n - 1, memo) + fib(n - 2, memo)\n"
        "    return memo[n]\n"
    )
    results = analyze(src)["fib"]
    names = {r.algorithm for r in results}
    assert "Memoization" in names
