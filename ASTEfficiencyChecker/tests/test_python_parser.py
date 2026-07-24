from parsers.python_parser import PythonParser
from engine.ir import IRKind


def parse(src):
    return PythonParser().parse(src, "<test>")


def test_extracts_top_level_functions():
    src = "def a():\n    pass\ndef b():\n    pass\n"
    prog = parse(src)
    names = {f.name for f in prog.functions}
    assert names == {"a", "b"}


def test_recursive_call_detected():
    src = "def fact(n):\n    if n <= 1:\n        return 1\n    return n * fact(n - 1)\n"
    prog = parse(src)
    fn = prog.functions[0]
    recursive = fn.node.find_all(IRKind.RECURSIVE_CALL)
    assert len(recursive) == 1
    assert recursive[0].name == "fact"


def test_non_recursive_call_not_flagged_recursive():
    src = "def f(x):\n    return len(x)\n"
    prog = parse(src)
    fn = prog.functions[0]
    assert len(fn.node.find_all(IRKind.RECURSIVE_CALL)) == 0
    assert len(fn.node.find_all(IRKind.CALL)) == 1


def test_nested_function_scoping_does_not_leak_recursion_flag():
    # calling `outer` from inside `inner` should NOT be flagged recursive on `inner`,
    # and calling `inner` from itself SHOULD be flagged recursive.
    src = (
        "def outer():\n"
        "    def inner(n):\n"
        "        if n == 0:\n"
        "            return outer()\n"
        "        return inner(n - 1)\n"
        "    return inner(3)\n"
    )
    prog = parse(src)
    inner = [f for f in prog.functions if f.name == "inner"][0]
    rec_names = {n.name for n in inner.node.find_all(IRKind.RECURSIVE_CALL)}
    assert rec_names == {"inner"}


def test_subscript_assign_detected():
    src = "def f(a):\n    a[0] = 5\n"
    prog = parse(src)
    fn = prog.functions[0]
    assert len(fn.node.find_all(IRKind.SUBSCRIPT_ASSIGN)) == 1
