# parsers/javascript_parser.py
from .base import BaseParser
import tree_sitter_javascript

class JavascriptParser(BaseParser):
    def __init__(self):
        mappings = {
            'function': ['function_declaration', 'arrow_function'],
            'identifier': ['identifier'],
            'loop': ['for_statement', 'while_statement', 'for_in_statement'],
            'branch': ['if_statement', 'switch_statement'],
            'call': ['call_expression'],
            'assignment': ['assignment_expression'],
            'math_op': ['binary_expression']
        }
        super().__init__("javascript", tree_sitter_javascript.language(), mappings)