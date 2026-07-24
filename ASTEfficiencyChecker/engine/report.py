# engine/report.py
from dataclasses import dataclass
from typing import Optional
from .complexity import Complexity

@dataclass
class AnalysisReport:
    detected_language: str
    detected_algorithm: str
    matched_problem: Optional[str]
    platform: Optional[str]
    confidence: float
    estimated_time_complexity: Complexity
    estimated_auxiliary_space: Complexity
    optimal_time_complexity: Optional[Complexity]
    optimal_auxiliary_space: Optional[Complexity]
    difference: str
    reasoning: str
    evidence: str
    suggestions: str

    def to_dict(self):
        return {
            "Detected Language": self.detected_language,
            "Detected Algorithm": self.detected_algorithm,
            "Matched Problem": self.matched_problem,
            "Platform": self.platform,
            "Confidence": round(self.confidence, 2),
            "Estimated Time Complexity": self.estimated_time_complexity.value,
            "Estimated Auxiliary Space": self.estimated_auxiliary_space.value,
            "Optimal Time Complexity": self.optimal_time_complexity.value if self.optimal_time_complexity else None,
            "Optimal Auxiliary Space": self.optimal_auxiliary_space.value if self.optimal_auxiliary_space else None,
            "Difference": self.difference,
            "Reasoning": self.reasoning,
            "Evidence": self.evidence,
            "Suggestions": self.suggestions
        }