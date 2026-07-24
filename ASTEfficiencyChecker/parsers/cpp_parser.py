# parsers/cpp_parser.py
from .base import BaseParser
import tree_sitter_cpp

class CppParser(BaseParser):
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
        super().__init__("cpp", tree_sitter_cpp.language(), mappings)