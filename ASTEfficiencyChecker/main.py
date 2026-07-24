# main.py
import argparse
import json
from engine.analyzer import Analyzer

def main():
    parser = argparse.ArgumentParser(description="ASTEfficiencyChecker Standalone Engine")
    parser.add_argument("file", help="Path to the source code file to analyze")
    parser.add_argument("--lang", help="Programming language hint", default=None)
    args = parser.parse_args()

    with open(args.file, "r", encoding="utf-8") as f:
        code = f.read()

    analyzer = Analyzer()
    report = analyzer.analyze(code, language_hint=args.lang)
    
    print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()