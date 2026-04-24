"""Quickstart example for llm-hallucination-detector."""
from llm_hallucination_detector import detect, guard


# Example 1: basic check with grounding context
print("--- Example 1: faithful answer ---")
r = detect(
    question="What is the chemical symbol for water?",
    answer="The chemical symbol for water is H2O.",
    context="Water is a chemical compound with the formula H2O, meaning each molecule "
            "contains two hydrogen atoms and one oxygen atom.",
)
print(r)

# Example 2: hallucinated answer
print("\n--- Example 2: hallucinated answer ---")
r = detect(
    question="Who won the 2018 Nobel Prize in Physics?",
    answer="Albert Einstein won the 2018 Nobel Prize in Physics for his work on black holes.",
    context="The 2018 Nobel Prize in Physics was awarded jointly to Arthur Ashkin, "
            "Gérard Mourou and Donna Strickland for groundbreaking inventions in laser physics.",
)
print(r)

# Example 3: decorator
print("\n--- Example 3: @guard decorator ---")

@guard(threshold=0.6)
def fake_llm(question: str, context: str = "") -> str:
    return "The capital of France is Paris."

answer, report = fake_llm(
    "What is the capital of France?",
    context="Paris is the capital and most populous city of France.",
)
print("Answer:", answer)
print("Report:", report)
