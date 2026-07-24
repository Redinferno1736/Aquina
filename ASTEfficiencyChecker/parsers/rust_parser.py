# parsers/rust_parser.py
from .base import BaseParser
import tree_sitter_rust

class RustParser(BaseParser):
    def __init__(self):
        mappings = {
            'function': ['function_item'],
            'identifier': ['identifier'],
            'loop': ['for_expression', 'while_expression', 'loop_expression'],
            'branch': ['if_expression', 'match_expression'],
            'call': ['call_expression'],
            'assignment': ['assignment_expression'],
            'math_op': ['binary_expression']
        }
        super().__init__("rust", tree_sitter_rust.language(), mappings)