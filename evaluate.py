import contextlib
import io
import json
import re
import time

from agent import ask

# expect_tools: "none"  -> the agent must NOT call any tool
#               [names] -> the agent must call at least one of these tools
#               None    -> don't check tools
# contains_any: the answer must contain at least one of these strings
TEST_CASES = [
    {
        "name": "Document question: who developed UPI",
        "question": "Who developed UPI?",
        "expect_tools": ["search_documents"],
        "contains_any": ["NPCI", "National Payments Corporation"],
    },
    {
        "name": "Document question: launch date",
        "question": "When was UPI launched?",
        "expect_tools": ["search_documents"],
        "contains_any": ["2016"],
    },
    {
        "name": "Document question: regulator",
        "question": "Which organization regulates UPI?",
        "expect_tools": ["search_documents"],
        "contains_any": ["Reserve Bank", "RBI"],
    },
    {
        "name": "General knowledge: no tools needed",
        "question": "What is the capital of Italy?",
        "expect_tools": "none",
        "contains_any": ["Rome"],
    },
    {
        "name": "General knowledge: arithmetic",
        "question": "What is 12 multiplied by 12?",
        "expect_tools": "none",
        "contains_any": ["144"],
    },
    {
        "name": "Time question uses the datetime tool",
        "question": "What time is it right now?",
        "expect_tools": ["get_current_datetime"],
        "contains_any": [":"],
    },
    {
        "name": "News question uses a search tool and gives links",
        "question": "What is the latest news about UPI this week?",
        "expect_tools": ["news_search", "web_search"],
        "contains_any": ["http"],
    },
    {
        "name": "Follow-up question uses chat memory",
        "question": "When was it launched?",
        "history": [
            {"role": "user", "content": "Who developed UPI?"},
            {"role": "assistant", "content": "UPI was developed by the National Payments Corporation of India (NPCI)."},
        ],
        "expect_tools": None,
        "contains_any": ["2016"],
    },
]


def run_case(case):
    """Run one question, capturing which tools the agent called from its log lines."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        answer = ask(case["question"], history=case.get("history"))
    tools_used = re.findall(r"\[Agent called tool: (\w+)", buffer.getvalue())
    return answer, tools_used


def check_case(case, answer, tools_used):
    problems = []

    expected = case["expect_tools"]
    if expected == "none":
        if tools_used:
            problems.append(f"expected no tools, but used {tools_used}")
    elif expected is not None:
        if not any(tool in tools_used for tool in expected):
            problems.append(f"expected one of {expected}, but used {tools_used or 'no tools'}")

    needles = case.get("contains_any")
    if needles and not any(n.lower() in answer.lower() for n in needles):
        problems.append(f"answer did not contain any of {needles}")

    return problems


if __name__ == "__main__":
    results = []
    passed = 0

    for case in TEST_CASES:
        try:
            answer, tools_used = run_case(case)
            problems = check_case(case, answer, tools_used)
        except Exception as e:
            answer, tools_used, problems = "", [], [f"crashed: {e}"]

        ok = not problems
        passed += 1 if ok else 0

        print(f"{'PASS' if ok else 'FAIL'}  {case['name']}")
        print(f"      tools used: {tools_used or 'none'}")
        for problem in problems:
            print(f"      problem: {problem}")

        results.append({
            "name": case["name"],
            "question": case["question"],
            "answer": answer,
            "tools_used": tools_used,
            "passed": ok,
            "problems": problems,
        })

        time.sleep(3)  # stay under the API's tokens-per-minute limit

    print("\n" + "=" * 50)
    print(f"SCORE: {passed}/{len(TEST_CASES)} ({passed / len(TEST_CASES) * 100:.0f}%)")
    print("=" * 50)

    with open("eval_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("Detailed results saved to eval_results.json")