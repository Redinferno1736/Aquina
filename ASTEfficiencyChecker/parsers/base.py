# parsers/base.py
from tree_sitter import Parser, Language
from engine.ir import Module, FunctionDef, Loop, Branch, FunctionCall, Assignment, MathOp

class BaseParser:
    def __init__(self, language_name: str, ts_language, mappings: dict):
        self.language_name = language_name
        self.ts_language = ts_language
        self.parser = Parser(Language(ts_language))
        self.mappings = mappings

    def parse(self, code: str) -> Module:
        tree = self.parser.parse(bytes(code, "utf8"))
        root = tree.root_node
        module = Module()
        for child in root.children:
            if child.type in self.mappings.get('function', []):
                module.functions.append(self._parse_function(child))
        return module

    def _parse_function(self, node) -> FunctionDef:
        name_node = self._find_child_by_type(node, self.mappings.get('identifier', []))
        name = name_node.text.decode('utf8') if name_node else "anonymous"
        body_nodes = self._traverse_body(node)
        return FunctionDef(name=name, body=body_nodes)

    def _traverse_body(self, node) -> list:
        ir_nodes = []
        for child in node.children:
            if child.type in self.mappings.get('loop', []):
                ir_nodes.append(Loop(loop_type="loop", body=self._traverse_body(child)))
            elif child.type in self.mappings.get('branch', []):
                ir_nodes.append(Branch(body=self._traverse_body(child)))
            elif child.type in self.mappings.get('call', []):
                name = child.text.decode('utf8').split('(')[0].strip()
                ir_nodes.append(FunctionCall(name=name))
            elif child.type in self.mappings.get('assignment', []):
                is_alloc = "new" in child.text.decode('utf8') or "[" in child.text.decode('utf8')
                ir_nodes.append(Assignment(is_allocation=is_alloc))
            elif child.type in self.mappings.get('math_op', []):
                ir_nodes.append(MathOp(operator=child.text.decode('utf8').strip()))
            else:
                ir_nodes.extend(self._traverse_body(child))
        return ir_nodes

    def _find_child_by_type(self, node, types):
        for child in node.children:
            if child.type in types:
                return child
        return None