# engine/complexity.py
from enum import Enum

class Complexity(Enum):
    O_1 = "O(1)"
    O_LOG_N = "O(log n)"
    O_N = "O(n)"
    O_N_LOG_N = "O(n log n)"
    O_N2 = "O(n²)"
    O_N3 = "O(n³)"
    O_N_K = "O(n^k)"
    O_2_N = "O(2^n)"
    O_N_FACT = "O(n!)"
    O_V = "O(V)"
    O_E = "O(E)"
    O_V_PLUS_E = "O(V+E)"
    O_V_LOG_V = "O(V log V)"
    O_E_LOG_V = "O(E log V)"
    UNKNOWN = "Unknown"

    @classmethod
    def from_string(cls, val: str):
        for member in cls:
            if member.value == val:
                return member
        return cls.UNKNOWN