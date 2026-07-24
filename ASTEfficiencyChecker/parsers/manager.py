# parsers/manager.py
import tree_sitter_python
import tree_sitter_c
import tree_sitter_cpp
import tree_sitter_java
import tree_sitter_javascript
import tree_sitter_typescript
import tree_sitter_rust
import tree_sitter_go
import tree_sitter_c_sharp

from .python_parser import PythonParser
from .c_parser import CParser
from .cpp_parser import CppParser
from .java_parser import JavaParser
from .javascript_parser import JavascriptParser
from .typescript_parser import TypescriptParser
from .rust_parser import RustParser
from .go_parser import GoParser
from .csharp_parser import CsharpParser

class ParserManager:
    def __init__(self):
        self.parsers = {
            "python": PythonParser(),
            "c": CParser(),
            "cpp": CppParser(),
            "java": JavaParser(),
            "javascript": JavascriptParser(),
            "typescript": TypescriptParser(),
            "rust": RustParser(),
            "go": GoParser(),
            "csharp": CsharpParser(),
        }

    def get_parser(self, code: str, hint: str = None):
        if hint and hint.lower() in self.parsers:
            return self.parsers[hint.lower()], hint.lower()
        
        if "def " in code and ":" in code: return self.parsers["python"], "python"
        if "fn " in code and "->" in code: return self.parsers["rust"], "rust"
        if "func " in code: return self.parsers["go"], "go"
        
        return self.parsers["cpp"], "cpp"