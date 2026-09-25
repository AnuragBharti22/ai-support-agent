import json
from agent import ask

# Test set: question + what a correct answer should contain
test_cases = [
    {
        "question": "What is UPI?",
        "expected_topic": "an instant payment system developed by NPCI for inter-bank transactions"
    },
    {
        "question": "Who developed UPI?",
        "expected_topic": "National Payments Corporation of India (NPCI)"
    },
    {
        "question": "When was UPI launched?",
        "expected_topic": "11 April 2016"
    },
    {
        "question": "What organization regulates UPI?",
        "expected_topic": "Reserve Bank of India (RBI)"
    },
    {
        "question": "What is the capital of Italy?",
        "expected_topic": "NOT_ANSWERABLE"
    }
]

def judge_answer(question, expected_topic, actual_answer):
    """Uses the LLM itself to judge if the answer is correct/faithful."""
    from groq import Groq
    from dotenv import load_dotenv
    import os

    load_dotenv()
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    if expected_topic == "NOT_ANSWERABLE":
        judge_prompt = f"""The question "{question}" cannot be answered from the given documents.
A correct system should refuse to answer or say it doesn't know.

Actual answer given: "{actual_answer}"

Did the system correctly refuse/decline to answer? Reply with only YES or NO."""
    else:
        judge_prompt = f"""Question: {question}
Expected topic the answer should cover: {expected_topic}
Actual answer given: "{actual_answer}"

Does the actual answer correctly and faithfully cover the expected topic? Reply with only YES or NO."""

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": judge_prompt}]
    )
    return response.choices[0].message.content.strip()


# Run evaluation
results = []
correct_count = 0

for case in test_cases:
    print(f"Testing: {case['question']}")
    actual_answer = ask(case["question"])
    verdict = judge_answer(case["question"], case["expected_topic"], actual_answer)
    is_correct = "YES" in verdict.upper()
    if is_correct:
        correct_count += 1

    results.append({
        "question": case["question"],
        "answer": actual_answer,
        "verdict": verdict,
        "correct": is_correct
    })
    print(f"  Verdict: {verdict}\n")

# Summary
print("=" * 50)
print(f"SCORE: {correct_count}/{len(test_cases)} ({correct_count/len(test_cases)*100:.0f}%)")
print("=" * 50)

# Save detailed results to a file
with open("eval_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print("\nDetailed results saved to eval_results.json")