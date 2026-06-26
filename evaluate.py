import argparse
from dataclasses import dataclass
from typing import Iterable, List

from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from config import CHAT_MODEL
from query import answer_question


JUDGE_PROMPT = """You are an impartial evaluator.
Given the context below and the answer provided, determine whether every single claim in the answer is directly and explicitly supported by the context.
Output only the word YES if the answer is fully grounded, or NO if the answer contains anything not present in the context.
No explanation needed."""


DEFAULT_QUESTIONS = [
    "What is the purpose of the Delegation of Financial Powers Rules 2024?",
    "What procurement procedures are described for capital acquisition?",
    "What are the responsibilities of authorities exercising financial powers?",
    "What documents discuss naval regulations?",
    "What is the policy for buying office furniture on Mars?",
]


@dataclass
class EvaluationResult:
    question: str
    answer: str
    faithfulness: str


def judge_faithfulness(question: str, answer: str, context: str) -> str:
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", JUDGE_PROMPT),
            (
                "human",
                "Question:\n{question}\n\nContext:\n{context}\n\nAnswer:\n{answer}",
            ),
        ]
    )
    judge = ChatGoogleGenerativeAI(model=CHAT_MODEL, temperature=0)
    response = (prompt | judge).invoke(
        {"question": question, "context": context, "answer": answer}
    )
    verdict = response.content.strip().upper()
    return "YES" if verdict.startswith("YES") else "NO"


def evaluate_questions(questions: Iterable[str]) -> List[EvaluationResult]:
    results = []
    for question in questions:
        answer, _, context = answer_question(question)
        faithfulness = judge_faithfulness(question, answer, context)
        results.append(
            EvaluationResult(
                question=question,
                answer=answer,
                faithfulness=faithfulness,
            )
        )
    return results


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Evaluate RAG answer faithfulness.")
    parser.add_argument(
        "--questions-file",
        help="Optional text file with one evaluation question per line.",
    )
    args = parser.parse_args()

    if args.questions_file:
        with open(args.questions_file, "r", encoding="utf-8") as handle:
            questions = [line.strip() for line in handle if line.strip()]
    else:
        questions = DEFAULT_QUESTIONS

    results = evaluate_questions(questions)
    yes_count = sum(result.faithfulness == "YES" for result in results)
    no_count = len(results) - yes_count

    for index, result in enumerate(results, start=1):
        print(f"\n{index}. {result.question}")
        print(f"Faithful: {result.faithfulness}")
        print(result.answer)

    print("\nSummary")
    print(f"YES: {yes_count}")
    print(f"NO: {no_count}")
    print(f"Total: {len(results)}")


if __name__ == "__main__":
    main()
