# parsers/go_parser.py
from .base import BaseParser
import tree_sitter_go

class GoParser(BaseParser):
    def __init__(self):
        mappings = {
            'function': ['function_declaration', 'method_declaration'],
            'identifier': ['identifier'],
            'loop': ['for_statement'],
            'branch': ['if_statement', 'switch_statement'],
            'call': ['call_expression'],
            'assignment': ['assignment_statement'],
            'math_op': ['binary_expression']
        }
        super().__init__("go", tree_sitter_go.language(), mappings)