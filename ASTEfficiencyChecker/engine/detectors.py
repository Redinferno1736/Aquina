# engine/detectors.py
from .ir import Module, Loop, Branch, FunctionDef, FunctionCall, MathOp, Assignment

class AlgorithmDetector:
    def __init__(self):
        self.detected_algorithms = []

    def detect(self, module: Module) -> str:
        for func in module.functions:
            if self._is_binary_search(func):
                return "Binary Search"
            if self._is_dfs(func):
                return "DFS"
            if self._is_bfs(func):
                return "BFS"
            if self._is_two_pointer(func):
                return "Two Pointer"
            if self._is_dp(func):
                return "Dynamic Programming"
        return "Unknown"

    def _is_binary_search(self, func: FunctionDef) -> bool:
        for node in func.body:
            if isinstance(node, Loop):
                has_branch = any(isinstance(n, Branch) for n in node.body)
                # Updated to look inside the string for C++ compatibility
                has_div = any(isinstance(n, MathOp) and ('/' in n.operator or '>>' in n.operator) for n in node.body)
                if has_branch and has_div:
                    return True
        return False

    def _is_dfs(self, func: FunctionDef) -> bool:
        has_branch = False
        has_recursive_call = False
        for node in func.body:
            if isinstance(node, Branch) or isinstance(node, Loop):
                has_branch = True
            if isinstance(node, FunctionCall) and node.name == func.name:
                has_recursive_call = True
        return has_branch and has_recursive_call

    def _is_bfs(self, func: FunctionDef) -> bool:
        has_loop = False
        has_alloc_in_loop = False
        for node in func.body:
            if isinstance(node, Loop):
                has_loop = True
                has_alloc_in_loop = any(isinstance(n, Assignment) and n.is_allocation for n in node.body)
        return has_loop and has_alloc_in_loop

    def _is_two_pointer(self, func: FunctionDef) -> bool:
        for node in func.body:
            if isinstance(node, Loop):
                branches = [n for n in node.body if isinstance(n, Branch)]
                if len(branches) > 1:
                    return True
        return False

    def _is_dp(self, func: FunctionDef) -> bool:
        has_alloc = any(isinstance(n, Assignment) and n.is_allocation for n in func.body)
        has_recursion = any(isinstance(n, FunctionCall) and n.name == func.name for n in func.body)
        nested_loops = any(isinstance(n, Loop) and any(isinstance(inner, Loop) for inner in n.body) for n in func.body)
        return has_alloc and (has_recursion or nested_loops)