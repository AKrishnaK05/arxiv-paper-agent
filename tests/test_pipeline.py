import sys
sys.stdout.reconfigure(encoding="utf-8")

from app.pipeline import ResearchPipeline

pipeline = ResearchPipeline()

print("=" * 60)
print("TEST 1: Empty input")
print("=" * 60)
print(pipeline.run(""))

print("\n" + "=" * 60)
print("TEST 2: Invalid format")
print("=" * 60)
print(pipeline.run(
    "What is this paper about?",
    paper_id="2109.05633",
    output_format="xml"
))

print("\n" + "=" * 60)
print("TEST 3: Invalid paper")
print("=" * 60)
print(pipeline.run(
    "What is this paper about?",
    paper_id="9999.99999"
))

print("\n" + "=" * 60)
print("TEST 4: QA in Markdown Format")
print("=" * 60)
print(pipeline.run(
    "What datasets were used in 2109.05633?",
    output_format="markdown"
))

print("\n" + "=" * 60)
print("TEST 5: Briefing in Structured JSON Format")
print("=" * 60)
print(pipeline.run(
    "Give me an executive briefing of 2109.05633",
    output_format="json"
))