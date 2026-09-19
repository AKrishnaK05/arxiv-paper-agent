import os
import sys

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from app.pipeline import ResearchPipeline


def run_cli():
    print("=" * 70)
    print("Autonomous arXiv Paper Digest & QA Agent")
    print("=" * 70)
    print("Enter an arXiv ID, URL, or research topic to generate an Executive Briefing.")
    print("Type 'exit' or 'quit' at any prompt to stop.\n")

    pipeline = ResearchPipeline()

    while True:
        try:
            user_input = input("Research Topic or arXiv ID > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in {"exit", "quit", "q"}:
            print("Exiting. Happy researching!")
            break

        print("\nProcessing paper (retrieving, parsing, indexing)... Please wait.")

        # Ensure the initial request triggers the executive briefing
        briefing_prompt = (
            user_input
            if any(k in user_input.lower() for k in ["briefing", "summary", "summarize", "overview"])
            else f"Give me an executive briefing for {user_input}"
        )

        result = pipeline.run(
            user_input=briefing_prompt,
            output_format="markdown"
        )

        if isinstance(result, dict) and "error" in result:
            print(f"\nError: {result['error']}\n")
            continue

        print("\n" + "=" * 70)
        print(result)
        print("=" * 70)

        # Transition into follow-up QA Loop
        active_paper_id = pipeline.current_paper_id
        print(f"\nQA Mode Active — You can now ask follow-up questions about this paper ({active_paper_id}).")
        print("Type 'new' to analyze a different paper, or 'exit' to quit.\n")

        while True:
            try:
                question = input("Follow-up Question > ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nGoodbye!")
                return

            if not question:
                continue

            if question.lower() in {"exit", "quit", "q"}:
                print("Exiting. Happy researching!")
                return

            if question.lower() == "new":
                pipeline.reset()
                print("\n" + "-" * 70 + "\n")
                break

            print("\nRetrieving grounded context from paper...")
            qa_result = pipeline.run(
                user_input=question,
                paper_id=pipeline.current_paper_id,
                output_format="markdown"
            )

            if isinstance(qa_result, dict) and "error" in qa_result:
                print(f"\nError: {qa_result['error']}\n")
                continue

            print("\n" + "-" * 70)
            print(qa_result)
            print("-" * 70 + "\n")


if __name__ == "__main__":
    run_cli()
