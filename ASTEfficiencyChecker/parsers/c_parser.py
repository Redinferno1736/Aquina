# parsers/c_parser.py
from .base import BaseParser
import tree_sitter_c

class CParser(BaseParser):
    def __init__(self):
        mappings = {
            'function': ['function_definition'],
            'identifier': ['identifier'],
            'loop': ['for_statement', 'while_statement', 'do_statement'],
            'branch': ['if_statement', 'switch_statement'],
            'call': ['call_expression'],
            'assignment': ['assignment_expression'],
            'math_op': ['binary_expression']
        }
        super().__init__("c", tree_sitter_c.language(), mappings)