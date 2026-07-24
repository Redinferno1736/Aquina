# parsers/python_parser.py
from .base import BaseParser
import tree_sitter_python

class PythonParser(BaseParser):
    def __init__(self):
        mappings = {
            'function': ['function_definition'],
            'identifier': ['identifier'],
            'loop': ['for_statement', 'while_statement'],
            'branch': ['if_statement', 'elif_clause', 'else_clause'],
            'call': ['call'],
            'assignment': ['assignment'],
            'math_op': ['binary_operator']
        }
        super().__init__("python", tree_sitter_python.language(), mappings)