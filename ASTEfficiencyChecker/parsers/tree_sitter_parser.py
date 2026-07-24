"""
parsers/tree_sitter_parser.py — generic adapter that converts a tree-sitter
parse tree into the shared IR, for every non-Python language.

Requires: pip install tree-sitter tree-sitter-languages
(see requirements.txt). This module could not be executed/tested in the
build environment (no network access to install the grammars) — the node
kind maps below were built from each grammar's published node-type
reference, but treat this file as "needs a local test pass," per our plan:
run scripts/run_analysis.py against real source files in each language,
and report back mismatches (usually a wrong/missing node-type string in the
map below) so they can be corrected in one pass.

Design: each language gets a NODE_KIND_MAP translating tree-sitter node
`type` strings to IRKind, plus small per-language hooks for the handful of
things that aren't simple string→kind mappings (call callee extraction,
recursion detection, comprehension-equivalents, etc).
"""

from __future__ import annotations
from engine.ir import IRNode, IRKind, IRFunction, IRProgram
from parsers.base import LanguageParser, ParserUnavailableError

try:
    from tree_sitter_languages import get_parser as _ts_get_parser
    _TREE_SITTER_AVAILABLE = True
except Exception:
    _TREE_SITTER_AVAILABLE = False


# tree-sitter node `type` -> IRKind, per language. Extend as needed — any
# node type not listed here becomes IRKind.UNKNOWN and is still walked (its
# children are still visited), it just isn't specially recognized by rules
# or detectors.
NODE_KIND_MAPS: dict[str, dict[str, IRKind]] = {
    "c": {
        "translation_unit": IRKind.MODULE,
        "function_definition": IRKind.FUNCTION_DEF,
        "if_statement": IRKind.IF,
        "for_statement": IRKind.FOR,
        "while_statement": IRKind.WHILE,
        "do_statement": IRKind.WHILE,
        "switch_statement": IRKind.SWITCH,
        "call_expression": IRKind.CALL,
        "return_statement": IRKind.RETURN,
        "break_statement": IRKind.BREAK,
        "continue_statement": IRKind.CONTINUE,
        "assignment_expression": IRKind.ASSIGN,
        "binary_expression": IRKind.BINARY_OP,
        "unary_expression": IRKind.UNARY_OP,
        "subscript_expression": IRKind.INDEX,
        "field_expression": IRKind.ATTRIBUTE,
        "identifier": IRKind.NAME,
        "number_literal": IRKind.LITERAL,
        "string_literal": IRKind.LITERAL,
    },
    "cpp": {
        "translation_unit": IRKind.MODULE,
        "function_definition": IRKind.FUNCTION_DEF,
        "if_statement": IRKind.IF,
        "for_statement": IRKind.FOR,
        "for_range_loop": IRKind.FOR,
        "while_statement": IRKind.WHILE,
        "do_statement": IRKind.WHILE,
        "switch_statement": IRKind.SWITCH,
        "call_expression": IRKind.CALL,
        "return_statement": IRKind.RETURN,
        "break_statement": IRKind.BREAK,
        "continue_statement": IRKind.CONTINUE,
        "assignment_expression": IRKind.ASSIGN,
        "binary_expression": IRKind.BINARY_OP,
        "unary_expression": IRKind.UNARY_OP,
        "subscript_expression": IRKind.INDEX,
        "field_expression": IRKind.ATTRIBUTE,
        "identifier": IRKind.NAME,
        "number_literal": IRKind.LITERAL,
        "string_literal": IRKind.LITERAL,
        "lambda_expression": IRKind.LAMBDA,
        "class_specifier": IRKind.CLASS_DEF,
    },
    "java": {
        "program": IRKind.MODULE,
        "method_declaration": IRKind.FUNCTION_DEF,
        "class_declaration": IRKind.CLASS_DEF,
        "if_statement": IRKind.IF,
        "for_statement": IRKind.FOR,
        "enhanced_for_statement": IRKind.FOR,
        "while_statement": IRKind.WHILE,
        "do_statement": IRKind.WHILE,
        "switch_expression": IRKind.SWITCH,
        "method_invocation": IRKind.CALL,
        "return_statement": IRKind.RETURN,
        "break_statement": IRKind.BREAK,
        "continue_statement": IRKind.CONTINUE,
        "assignment_expression": IRKind.ASSIGN,
        "binary_expression": IRKind.BINARY_OP,
        "unary_expression": IRKind.UNARY_OP,
        "array_access": IRKind.INDEX,
        "field_access": IRKind.ATTRIBUTE,
        "identifier": IRKind.NAME,
        "decimal_integer_literal": IRKind.LITERAL,
        "string_literal": IRKind.LITERAL,
        "lambda_expression": IRKind.LAMBDA,
        "object_creation_expression": IRKind.CALL,
        "try_statement": IRKind.TRY,
    },
    "javascript": {
        "program": IRKind.MODULE,
        "function_declaration": IRKind.FUNCTION_DEF,
        "function": IRKind.FUNCTION_DEF,
        "arrow_function": IRKind.FUNCTION_DEF,
        "method_definition": IRKind.FUNCTION_DEF,
        "class_declaration": IRKind.CLASS_DEF,
        "if_statement": IRKind.IF,
        "for_statement": IRKind.FOR,
        "for_in_statement": IRKind.FOR,
        "while_statement": IRKind.WHILE,
        "do_statement": IRKind.WHILE,
        "switch_statement": IRKind.SWITCH,
        "call_expression": IRKind.CALL,
        "return_statement": IRKind.RETURN,
        "break_statement": IRKind.BREAK,
        "continue_statement": IRKind.CONTINUE,
        "assignment_expression": IRKind.ASSIGN,
        "binary_expression": IRKind.BINARY_OP,
        "unary_expression": IRKind.UNARY_OP,
        "subscript_expression": IRKind.INDEX,
        "member_expression": IRKind.ATTRIBUTE,
        "identifier": IRKind.NAME,
        "number": IRKind.LITERAL,
        "string": IRKind.LITERAL,
        "object": IRKind.DICT_LITERAL,
        "array": IRKind.LIST_LITERAL,
        "try_statement": IRKind.TRY,
    },
    "typescript": {},  # filled below (extends javascript map)
    "rust": {
        "source_file": IRKind.MODULE,
        "function_item": IRKind.FUNCTION_DEF,
        "impl_item": IRKind.CLASS_DEF,
        "if_expression": IRKind.IF,
        "for_expression": IRKind.FOR,
        "while_expression": IRKind.WHILE,
        "loop_expression": IRKind.WHILE,
        "match_expression": IRKind.SWITCH,
        "call_expression": IRKind.CALL,
        "return_expression": IRKind.RETURN,
        "break_expression": IRKind.BREAK,
        "continue_expression": IRKind.CONTINUE,
        "assignment_expression": IRKind.ASSIGN,
        "binary_expression": IRKind.BINARY_OP,
        "unary_expression": IRKind.UNARY_OP,
        "index_expression": IRKind.INDEX,
        "field_expression": IRKind.ATTRIBUTE,
        "identifier": IRKind.NAME,
        "integer_literal": IRKind.LITERAL,
        "string_literal": IRKind.LITERAL,
        "closure_expression": IRKind.LAMBDA,
    },
    "go": {
        "source_file": IRKind.MODULE,
        "function_declaration": IRKind.FUNCTION_DEF,
        "method_declaration": IRKind.FUNCTION_DEF,
        "if_statement": IRKind.IF,
        "for_statement": IRKind.FOR,
        "expression_switch_statement": IRKind.SWITCH,
        "call_expression": IRKind.CALL,
        "return_statement": IRKind.RETURN,
        "break_statement": IRKind.BREAK,
        "continue_statement": IRKind.CONTINUE,
        "assignment_statement": IRKind.ASSIGN,
        "short_var_declaration": IRKind.ASSIGN,
        "binary_expression": IRKind.BINARY_OP,
        "unary_expression": IRKind.UNARY_OP,
        "index_expression": IRKind.INDEX,
        "selector_expression": IRKind.ATTRIBUTE,
        "identifier": IRKind.NAME,
        "int_literal": IRKind.LITERAL,
        "interpreted_string_literal": IRKind.LITERAL,
        "func_literal": IRKind.LAMBDA,
    },
    "csharp": {
        "compilation_unit": IRKind.MODULE,
        "method_declaration": IRKind.FUNCTION_DEF,
        "local_function_statement": IRKind.FUNCTION_DEF,
        "class_declaration": IRKind.CLASS_DEF,
        "if_statement": IRKind.IF,
        "for_statement": IRKind.FOR,
        "foreach_statement": IRKind.FOR,
        "while_statement": IRKind.WHILE,
        "do_statement": IRKind.WHILE,
        "switch_statement": IRKind.SWITCH,
        "invocation_expression": IRKind.CALL,
        "return_statement": IRKind.RETURN,
        "break_statement": IRKind.BREAK,
        "continue_statement": IRKind.CONTINUE,
        "assignment_expression": IRKind.ASSIGN,
        "binary_expression": IRKind.BINARY_OP,
        "prefix_unary_expression": IRKind.UNARY_OP,
        "element_access_expression": IRKind.INDEX,
        "member_access_expression": IRKind.ATTRIBUTE,
        "identifier": IRKind.NAME,
        "integer_literal": IRKind.LITERAL,
        "string_literal": IRKind.LITERAL,
        "lambda_expression": IRKind.LAMBDA,
        "try_statement": IRKind.TRY,
    },
}
# TypeScript is a syntactic superset of the JS grammar in tree-sitter-languages
NODE_KIND_MAPS["typescript"] = dict(NODE_KIND_MAPS["javascript"])

# Which node type holds a function/method's own name, per language
FUNCTION_NAME_FIELD = {
    "c": "declarator", "cpp": "declarator", "java": "name", "javascript": "name",
    "typescript": "name", "rust": "name", "go": "name", "csharp": "name",
}

# tree-sitter field name used to pull the callee out of a call node
CALL_FUNCTION_FIELD = {
    "c": "function", "cpp": "function", "java": "name", "javascript": "function",
    "typescript": "function", "rust": "function", "go": "function", "csharp": "expression",
}


class TreeSitterParser(LanguageParser):
    def __init__(self, language: str):
        if not _TREE_SITTER_AVAILABLE:
            raise ParserUnavailableError(
                "tree_sitter_languages is not installed. Run: "
                "pip install tree-sitter tree-sitter-languages"
            )
        self.language = language
        grammar_name = {
            "c": "c", "cpp": "cpp", "java": "java", "javascript": "javascript",
            "typescript": "typescript", "rust": "rust", "go": "go", "csharp": "c_sharp",
        }[language]
        try:
            self._ts_parser = _ts_get_parser(grammar_name)
        except Exception as e:
            raise ParserUnavailableError(
                f"Could not load tree-sitter grammar '{grammar_name}' for language "
                f"'{language}': {e}"
            )
        self._node_map = NODE_KIND_MAPS[language]
        self._current_func_stack: list[str] = []

    def parse(self, source: str, filepath: str = "<string>") -> IRProgram:
        source_bytes = source.encode("utf-8")
        tree = self._ts_parser.parse(source_bytes)
        root = tree.root_node
        module_node = self._convert(root, source_bytes)
        functions = self._extract_functions(module_node, source)
        return IRProgram(
            language=self.language, filepath=filepath, module_node=module_node,
            functions=functions, source=source,
        )

    # -- conversion ---------------------------------------------------------

    def _text(self, node, source_bytes: bytes) -> str:
        return source_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")

    def _convert(self, node, source_bytes: bytes) -> IRNode:
        kind = self._node_map.get(node.type, IRKind.UNKNOWN)
        line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1
        name = None
        meta = {}

        if kind == IRKind.FUNCTION_DEF:
            name = self._extract_function_name(node, source_bytes)
            self._current_func_stack.append(name or "")
        elif kind in (IRKind.CALL,):
            name = self._extract_call_callee(node, source_bytes)
            # only the innermost enclosing function counts as "self" — matches
            # python_parser.py's fix for the same nested-function leak
            if name and self._current_func_stack and name == self._current_func_stack[-1]:
                kind = IRKind.RECURSIVE_CALL
            meta["callee"] = name
        elif kind == IRKind.NAME:
            name = self._text(node, source_bytes)
        elif kind == IRKind.FOR:
            meta["iter_desc"] = self._text(node, source_bytes)[:80]

        children = [self._convert(c, source_bytes) for c in node.children]

        if self._node_map.get(node.type) == IRKind.FUNCTION_DEF:
            self._current_func_stack.pop()

        return IRNode(kind=kind, name=name, line=line, end_line=end_line, children=children, meta=meta)

    def _extract_function_name(self, node, source_bytes: bytes) -> str | None:
        field_name = FUNCTION_NAME_FIELD.get(self.language)
        if field_name:
            try:
                child = node.child_by_field_name(field_name)
                if child is not None:
                    txt = self._text(child, source_bytes)
                    # for C/C++ "declarator" the text may include params; take first identifier-ish token
                    return txt.split("(")[0].strip().split()[-1] if txt else None
            except Exception:
                pass
        # fallback: look for first identifier child
        for c in node.children:
            if c.type == "identifier":
                return self._text(c, source_bytes)
        return None

    def _extract_call_callee(self, node, source_bytes: bytes) -> str | None:
        field_name = CALL_FUNCTION_FIELD.get(self.language)
        if field_name:
            try:
                child = node.child_by_field_name(field_name)
                if child is not None:
                    txt = self._text(child, source_bytes)
                    return txt.split(".")[-1].split("::")[-1]
            except Exception:
                pass
        return None

    # -- function extraction --------------------------------------------

    def _extract_functions(self, module_node: IRNode, source: str) -> list[IRFunction]:
        funcs = []
        lines = source.splitlines()
        for node in module_node.walk():
            if node.kind == IRKind.FUNCTION_DEF:
                snippet = "\n".join(lines[max(node.line - 1, 0):node.end_line])
                funcs.append(IRFunction(
                    name=node.name or "<anonymous>", node=node, params=[],
                    line=node.line, end_line=node.end_line,
                    language=self.language, source=snippet,
                ))
        return funcs
