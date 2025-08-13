def qc_score(coverage: float, semantics: float, clarity: float, redundancy: float) -> float:
    return 0.4*coverage + 0.3*semantics + 0.2*clarity + 0.1*(1.0 - redundancy)
