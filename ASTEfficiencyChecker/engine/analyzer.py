# engine/analyzer.py
from parsers.manager import ParserManager
from .rules import RuleEngine
from .detectors import AlgorithmDetector
from .matcher import ProblemMatcher
from .report import AnalysisReport
from .complexity import Complexity

class Analyzer:
    def __init__(self):
        self.parser_manager = ParserManager()
        self.rule_engine = RuleEngine()
        self.detector = AlgorithmDetector()
        self.matcher = ProblemMatcher()

    def analyze(self, code: str, language_hint: str = None) -> AnalysisReport:
        parser, detected_lang = self.parser_manager.get_parser(code, language_hint)
        ir_module = parser.parse(code)

        est_time, est_space = self.rule_engine.estimate(ir_module)
        algorithm = self.detector.detect(ir_module)
        matched_data, confidence = self.matcher.match(code, algorithm)

        opt_time = Complexity.UNKNOWN
        opt_space = Complexity.UNKNOWN
        problem_title = None
        platform = None

        if matched_data:
            problem_title = matched_data.get("title")
            platform = matched_data.get("platform")
            opt_time = Complexity.from_string(matched_data.get("optimal_time_complexity", "Unknown"))
            opt_space = Complexity.from_string(matched_data.get("optimal_space_complexity", "Unknown"))

        diff = "Optimal" if est_time == opt_time else "Suboptimal"
        reasoning = f"Code mapped to IR revealing depth {self.rule_engine.max_loop_depth} and recursion {self.rule_engine.has_recursion}."
        evidence = f"Algorithm structural signature matched {algorithm}."
        suggestions = "Consider memoization or reducing nested loops." if diff == "Suboptimal" else "Code meets theoretical limits."

        return AnalysisReport(
            detected_language=detected_lang,
            detected_algorithm=algorithm,
            matched_problem=problem_title,
            platform=platform,
            confidence=confidence,
            estimated_time_complexity=est_time,
            estimated_auxiliary_space=est_space,
            optimal_time_complexity=opt_time,
            optimal_auxiliary_space=opt_space,
            difference=diff,
            reasoning=reasoning,
            evidence=evidence,
            suggestions=suggestions
        )