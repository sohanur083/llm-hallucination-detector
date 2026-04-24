from llm_hallucination_detector import detect, Verdict


def test_faithful_answer_scores_high():
    r = detect(
        question="What is 2+2?",
        answer="2 plus 2 equals 4.",
        context="Two plus two equals four.",
    )
    assert r.score > 0.5
    assert r.verdict in (Verdict.FAITHFUL, Verdict.SUSPECT)


def test_hallucinated_answer_scores_low():
    r = detect(
        question="Who wrote Hamlet?",
        answer="Hamlet was written by Isaac Newton in the 1800s.",
        context="Hamlet is a tragedy written by William Shakespeare around 1600.",
    )
    assert r.score < 0.7


def test_result_has_signals():
    r = detect(
        question="Q?",
        answer="A.",
        context="C.",
    )
    assert "nli" in r.signals
    assert "grounding" in r.signals
    assert "consistency" in r.signals
