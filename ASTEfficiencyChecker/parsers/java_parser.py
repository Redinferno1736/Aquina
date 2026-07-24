# parsers/java_parser.py
from .base import BaseParser
import tree_sitter_java

class JavaParser(BaseParser):
    def __init__(self):
        mappings = {
            'function': ['method_declaration'],
            'identifier': ['identifier'],
            'loop': ['for_statement', 'while_statement', 'do_statement'],
            'branch': ['if_statement', 'switch_expression'],
            'call': ['method_invocation'],
            'assignment': ['assignment_expression'],
            'math_op': ['binary_expression']
        }
        super().__init__("java", tree_sitter_java.language(), mappings)