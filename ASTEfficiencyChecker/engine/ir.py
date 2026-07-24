# engine/ir.py
from dataclasses import dataclass, field
from typing import List, Optional, Any

@dataclass
class IRNode:
    pass

@dataclass
class Module(IRNode):
    functions: List['FunctionDef'] = field(default_factory=list)

@dataclass
class FunctionDef(IRNode):
    name: str
    body: List[IRNode] = field(default_factory=list)

@dataclass
class Loop(IRNode):
    loop_type: str  # 'for', 'while'
    body: List[IRNode] = field(default_factory=list)

@dataclass
class Branch(IRNode):
    body: List[IRNode] = field(default_factory=list)

@dataclass
class FunctionCall(IRNode):
    name: str

@dataclass
class Assignment(IRNode):
    is_allocation: bool = False

@dataclass
class MathOp(IRNode):
    operator: str # '+', '-', '*', '/', '>>', '<<'