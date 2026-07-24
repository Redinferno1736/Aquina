# parsers/csharp_parser.py
from .base import BaseParser
import tree_sitter_c_sharp

class CsharpParser(BaseParser):
    def __init__(self):
        mappings = {
            'function': ['method_declaration'],
            'identifier': ['identifier'],
            'loop': ['for_statement', 'while_statement', 'foreach_statement'],
            'branch': ['if_statement', 'switch_statement'],
            'call': ['invocation_expression'],
            'assignment': ['assignment_expression'],
            'math_op': ['binary_expression']
        }
        super().__init__("csharp", tree_sitter_c_sharp.language(), mappings)