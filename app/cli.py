import logging
import os
import sys
import warnings

# Suppress background library warnings that interfere with terminal input/output
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("chromadb").setLevel(logging.ERROR)

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from app.pipeline import ResearchPipeline


def run_cli():
    print("=" * 70, flush=True)
    print("Autonomous arXiv Paper Digest & QA Agent", flush=True)
    print("=" * 70, flush=True)
    print("Enter an arXiv ID, URL, or research topic to generate an Executive Briefing.", flush=True)
    print("Type 'exit' or 'quit' at any prompt to stop.\n", flush=True)

    pipeline = ResearchPipeline()

    while True:
        try:
            sys.stdout.flush()
            user_input = input("Research Topic or arXiv ID > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Goodbye!", flush=True)
            break

        if not user_input:
            continue

        if user_input.lower() in {"exit", "quit", "q"}:
            print("Exiting. Happy researching!", flush=True)
            break

        if user_input.lower() in {"new", "reset", "clear"}:
            pipeline.reset()
            print("\nSession reset. Ready for a new paper.\n", flush=True)
            continue

        print("\nProcessing paper (retrieving, parsing, indexing)... Please wait.", flush=True)

        # Ensure the initial request triggers the executive briefing
        briefing_prompt = (
            user_input
            if any(k in user_input.lower() for k in ["briefing", "summary", "summarize", "overview"])
            else f"Give me an executive briefing for {user_input}"
        )

        result = pipeline.run(
            user_input=briefing_prompt,
            intent="briefing",
            output_format="markdown"
        )

        if isinstance(result, dict) and "error" in result:
            print(f"\nError: {result['error']}\n", flush=True)
            continue

        print("\n" + "=" * 70, flush=True)
        print(result, flush=True)
        print("=" * 70, flush=True)

        # Transition into follow-up QA Loop
        active_paper_id = pipeline.current_paper_id
        print(f"\nQA Mode Active - You can now ask follow-up questions about this paper ({active_paper_id}).", flush=True)
        print("Type 'new' to analyze a different paper, or 'exit' to quit.\n", flush=True)

        while True:
            try:
                sys.stdout.flush()
                question = input("Follow-up Question > ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nExiting. Goodbye!", flush=True)
                return

            if not question:
                continue

            if question.lower() in {"exit", "quit", "q"}:
                print("Exiting. Happy researching!", flush=True)
                return

            if question.lower() in {"new", "reset"}:
                pipeline.reset()
                print("\n" + "-" * 70 + "\n", flush=True)
                break

            if question.lower() == "help":
                print("\nCommands in QA Mode:", flush=True)
                print("  - Ask any specific question to retrieve grounded context", flush=True)
                print("  - Type 'new' to analyze another paper", flush=True)
                print("  - Type 'exit' to quit\n", flush=True)
                continue

            print("\nRetrieving grounded context from paper...", flush=True)
            qa_result = pipeline.run(
                user_input=question,
                paper_id=pipeline.current_paper_id,
                intent="qa",
                output_format="markdown"
            )

            if isinstance(qa_result, dict) and "error" in qa_result:
                print(f"\nError: {qa_result['error']}\n", flush=True)
                continue

            print("\n" + "-" * 70, flush=True)
            print(qa_result, flush=True)
            print("-" * 70 + "\n", flush=True)


if __name__ == "__main__":
    run_cli()
