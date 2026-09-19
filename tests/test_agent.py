import json
import unittest
from unittest.mock import MagicMock

from app.query import understand_query
from app.chunking import chunk_pages
from app.output_formatter import (
    parse_pages,
    clean_sources,
    format_json,
    format_markdown,
    parse_markdown_sections,
)
from app.pipeline import ResearchPipeline


class TestQueryUnderstanding(unittest.TestCase):

    def test_detect_pure_arxiv_id(self):
        state = {"user_input": "2109.05633v1"}
        result = understand_query(state)
        self.assertEqual(result["query_type"], "paper_id")
        self.assertEqual(result["paper_id"], "2109.05633v1")

    def test_detect_arxiv_url(self):
        state = {"user_input": "https://arxiv.org/abs/2109.05633"}
        result = understand_query(state)
        self.assertEqual(result["query_type"], "paper_id")
        self.assertEqual(result["paper_id"], "2109.05633")

    def test_detect_embedded_id(self):
        state = {"user_input": "What datasets were used in 2109.05633?"}
        result = understand_query(state)
        self.assertEqual(result["query_type"], "paper_id")
        self.assertEqual(result["paper_id"], "2109.05633")

    def test_detect_topic_query(self):
        state = {"user_input": "quantum computing algorithms for optimization"}
        result = understand_query(state)
        self.assertEqual(result["query_type"], "topic")
        self.assertIsNone(result["paper_id"])

    def test_detect_briefing_intent(self):
        state = {"user_input": "give me an executive briefing of 2109.05633"}
        result = understand_query(state)
        self.assertEqual(result["intent"], "briefing")

    def test_detect_qa_intent(self):
        state = {"user_input": "What is the accuracy of the proposed model?"}
        result = understand_query(state)
        self.assertEqual(result["intent"], "qa")

    def test_reject_malformed_arxiv_id(self):
        malformed_inputs = [
            "999.999",
            "2109.56",
            "2109.0563abc",
            "2109.05633v",
            "123.456",
            "2109.056333",
        ]
        for inp in malformed_inputs:
            with self.subTest(inp=inp):
                state = {"user_input": inp}
                result = understand_query(state)
                self.assertEqual(result["query_type"], "invalid_paper_id")
                self.assertIn("Invalid arXiv paper ID", result["error"])

    def test_reject_malformed_arxiv_url(self):
        state = {"user_input": "https://arxiv.org/abs/999.999"}
        result = understand_query(state)
        self.assertEqual(result["query_type"], "invalid_paper_id")
        self.assertIn("Invalid arXiv URL", result["error"])

    def test_normal_topics_not_flagged_as_malformed(self):
        valid_topics = [
            "machine learning",
            "COVID 19",
            "GPT 4",
            "papers about transformer compression",
        ]
        for topic in valid_topics:
            with self.subTest(topic=topic):
                state = {"user_input": topic}
                result = understand_query(state)
                self.assertEqual(result["query_type"], "topic")
                self.assertNotIn("error", result)


class TestChunking(unittest.TestCase):

    def test_chunk_pages_structure(self):
        pages = [
            {"page_number": 1, "text": "This is page one text with some research details."},
            {"page_number": 2, "text": "This is page two text continuing the explanation."},
        ]
        chunks = chunk_pages(pages, paper_id="test-123", chunk_size=5, overlap=1)

        self.assertGreater(len(chunks), 0)
        first_chunk = chunks[0]
        self.assertEqual(first_chunk["paper_id"], "test-123")
        self.assertIn("chunk_id", first_chunk)
        self.assertIn("text", first_chunk)
        self.assertIn("pages", first_chunk)
        self.assertIn("start_page", first_chunk)
        self.assertIn("end_page", first_chunk)
        self.assertIsInstance(first_chunk["pages"], list)


class TestOutputFormatter(unittest.TestCase):

    def test_parse_pages(self):
        self.assertEqual(parse_pages("[8, 9]"), [8, 9])
        self.assertEqual(parse_pages([1, 2]), [1, 2])
        self.assertEqual(parse_pages("5"), [5])
        self.assertEqual(parse_pages(7), [7])
        self.assertEqual(parse_pages(""), [])

    def test_clean_sources_deduplication(self):
        raw_sources = [
            {"paper_id": "test-paper", "pages": "[1, 2]"},
            {"paper_id": "test-paper", "pages": [1, 2]},
            {"paper_id": "test-paper", "pages": "[3]"},
        ]
        cleaned = clean_sources(raw_sources)
        self.assertEqual(len(cleaned), 2)
        self.assertEqual(cleaned[0]["pages"], [1, 2])
        self.assertEqual(cleaned[1]["pages"], [3])

    def test_format_json_qa(self):
        sources = [{"paper_id": "2109.05633v1", "pages": "[1, 2]"}]
        output = format_json("The author used Python.", sources)
        data = json.loads(output)

        self.assertEqual(data["answer"], "The author used Python.")
        self.assertEqual(data["sources"], [{"paper_id": "2109.05633v1", "pages": [1, 2]}])

    def test_format_json_structured_briefing(self):
        briefing_text = """# Executive Briefing

## Problem
Text describing the core challenge.

## Approach
Text describing the model architecture.

## Limitations
Text describing limitations.
"""
        sources = [{"paper_id": "2109.05633v1", "pages": "[4, 5]"}]
        output = format_json(briefing_text, sources)
        data = json.loads(output)

        self.assertEqual(data["title"], "Executive Briefing")
        self.assertIn("Problem", data["sections"])
        self.assertIn("Approach", data["sections"])
        self.assertIn("Limitations", data["sections"])
        self.assertEqual(data["sections"]["Problem"], "Text describing the core challenge.")
        self.assertEqual(data["sources"][0]["pages"], [4, 5])

    def test_format_markdown_citations(self):
        sources = [
            {"paper_id": "2109.05633v1", "pages": "[8, 9]"},
            {"paper_id": "2109.05633v1", "pages": "1"},
        ]
        output = format_markdown("The model achieves 95% accuracy.", sources)

        self.assertIn("# Answer", output)
        self.assertIn("## Sources", output)
        self.assertIn("- Paper: `2109.05633v1` | Pages: 8, 9", output)
        self.assertIn("- Paper: `2109.05633v1` | Page: 1", output)
        # Ensure raw brackets "[8, 9]" are not printed in the citation line
        self.assertNotIn("Pages: [8, 9]", output)


class TestPipelineValidation(unittest.TestCase):

    def setUp(self):
        self.mock_vector_store = MagicMock()
        self.mock_rag = MagicMock()
        self.mock_briefing = MagicMock()

        self.pipeline = ResearchPipeline(
            vector_store=self.mock_vector_store,
            rag=self.mock_rag,
            briefing_service=self.mock_briefing,
        )

    def test_empty_input_validation(self):
        res1 = self.pipeline.run("")
        res2 = self.pipeline.run("   ")
        self.assertEqual(res1, {"error": "User input cannot be empty."})
        self.assertEqual(res2, {"error": "User input cannot be empty."})

    def test_unsupported_format_validation(self):
        res = self.pipeline.run("What is this paper?", output_format="xml")
        self.assertIn("error", res)
        self.assertIn("Unsupported output format: xml", res["error"])

    def test_session_persistence_for_qa(self):
        self.pipeline.current_paper_id = "2105.02723v1"
        self.pipeline.current_paper = {
            "arxiv_id": "2105.02723v1",
            "title": "Test Paper",
            "pdf_url": "http://example.com/test.pdf"
        }
        self.mock_rag.answer.return_value = {
            "answer": "Grounded response",
            "sources": [{"paper_id": "2105.02723v1", "pages": [1]}]
        }
        self.mock_vector_store.has_paper.return_value = True

        result = self.pipeline.run("What is the main finding?")
        self.assertIn("Grounded response", result)
        self.mock_rag.answer.assert_called_once()
        self.assertEqual(
            self.mock_rag.answer.call_args[1]["paper_id"],
            "2105.02723v1"
        )

    def test_malformed_id_pipeline_rejection(self):
        res = self.pipeline.run("999.999")
        self.assertIn("error", res)
        self.assertIn("Invalid arXiv paper ID", res["error"])


if __name__ == "__main__":
    unittest.main()
