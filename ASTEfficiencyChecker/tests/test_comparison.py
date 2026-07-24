from engine.comparison import compare_complexity, _normalize


def test_normalize_does_not_confuse_n_and_n_squared():
    assert _normalize("O(n)") == "O(n)"
    assert _normalize("O(n^2)") == "O(n^2)"
    assert _normalize("O(n log n)") == "O(n log n)"
    assert _normalize("O(log n)") == "O(log n)"


def test_matching_complexity_is_optimal():
    v = compare_complexity("O(log n)", "O(log n)")
    assert v.verdict == "optimal"


def test_worse_complexity_is_suboptimal():
    v = compare_complexity("O(n^2)", "O(n log n)")
    assert v.verdict == "suboptimal"


def test_unparseable_complexity_is_unknown():
    v = compare_complexity("O(V + E)", "O(log n)")
    assert v.verdict == "unknown"
