# engine/rules.py
from .ir import Module, FunctionDef, Loop, Branch, FunctionCall, Assignment, MathOp
from .complexity import Complexity

class RuleEngine:
    def __init__(self):
        self.time_complexity = Complexity.O_1
        self.space_complexity = Complexity.O_1
        self.max_loop_depth = 0
        self.has_recursion = False
        self.allocations = 0
        self.has_logarithmic_op = False

    def estimate(self, module: Module):
        for func in module.functions:
            self._walk_func(func, func.name)
        
        self._calculate_complexities()
        return self.time_complexity, self.space_complexity

    def _walk_func(self, func: FunctionDef, current_func_name: str):
        depth = self._calculate_loop_depth(func.body, current_func_name)
        if depth > self.max_loop_depth:
            self.max_loop_depth = depth

    def _calculate_loop_depth(self, nodes: list, current_func_name: str, current_depth: int = 0) -> int:
        max_d = current_depth
        for node in nodes:
            if isinstance(node, Loop):
                d = self._calculate_loop_depth(node.body, current_func_name, current_depth + 1)
                max_d = max(max_d, d)
            elif isinstance(node, Branch):
                d = self._calculate_loop_depth(node.body, current_func_name, current_depth)
                max_d = max(max_d, d)
            elif isinstance(node, FunctionCall):
                if node.name == current_func_name:
                    self.has_recursion = True
            elif isinstance(node, Assignment):
                if node.is_allocation:
                    self.allocations += 1
            elif isinstance(node, MathOp):
                # Detect halving operations typical in logarithmic algorithms (updated for C++)
                if '/' in node.operator or '//' in node.operator or '>>' in node.operator:
                    self.has_logarithmic_op = True
        return max_d

    def _calculate_complexities(self):
        # Time Complexity Logic
        if self.max_loop_depth == 1:
            if self.has_logarithmic_op:
                self.time_complexity = Complexity.O_LOG_N
            else:
                self.time_complexity = Complexity.O_N
        elif self.max_loop_depth == 2:
            if self.has_logarithmic_op:
                self.time_complexity = Complexity.O_N_LOG_N
            else:
                self.time_complexity = Complexity.O_N2
        elif self.max_loop_depth == 3:
            self.time_complexity = Complexity.O_N3
        elif self.max_loop_depth > 3:
            self.time_complexity = Complexity.O_N_K
        elif self.has_recursion:
            if self.has_logarithmic_op:
                self.time_complexity = Complexity.O_LOG_N
            else:
                self.time_complexity = Complexity.O_N_LOG_N
        else:
            self.time_complexity = Complexity.O_1

        # Space Complexity Logic
        if self.has_recursion or self.allocations > 0:
            self.space_complexity = Complexity.O_N
        else:
            self.space_complexity = Complexity.O_1