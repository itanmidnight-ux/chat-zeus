import tempfile
import unittest
from pathlib import Path

from src.response_control import (
    build_user_response,
    clean_output,
    detect_question_type,
    summarize_intelligently,
)
from src.storage import StorageManager


class ResponseControlTests(unittest.TestCase):
    def test_detect_question_type_classifies_expected_levels(self) -> None:
        self.assertEqual(detect_question_type('que hora es'), 'simple')
        self.assertEqual(detect_question_type('como funciona un cohete'), 'explicativa')
        self.assertEqual(detect_question_type('diseña una nave espacial para 100 personas'), 'analitica')

    def test_clean_output_removes_internal_markers_and_urls(self) -> None:
        raw = 'Análisis completo del problema: Hipótesis RAG ML weights Logs Checkpoints https://example.com conclusión útil.'
        cleaned = clean_output(raw)

        self.assertNotIn('Análisis completo del problema', cleaned)
        self.assertNotIn('RAG', cleaned)
        self.assertNotIn('https://', cleaned)
        self.assertIn('conclusión útil', cleaned)

    def test_generate_short_human_response_for_simple_question(self) -> None:
        level, response = build_user_response(
            'que hora es',
            {
                'direct_answer': 'Son las 14:32 aproximadamente.',
                'summary': 'Son las 14:32 aproximadamente.',
                'conclusions': 'Son las 14:32 aproximadamente.',
            },
        )

        self.assertEqual(level, 'simple')
        self.assertLessEqual(len(response.splitlines()), 2)
        self.assertIn('zona horaria', response)

    def test_summarize_intelligently_reduces_redundancy(self) -> None:
        summary = summarize_intelligently(
            'Un cohete usa empuje. Un cohete usa empuje. Expulsa gases a gran velocidad. Expulsa gases a gran velocidad.',
            'explicativa',
        )

        self.assertIn('Un cohete usa empuje.', summary)
        self.assertEqual(summary.count('Un cohete usa empuje.'), 1)


class ResponseFeedbackStorageTests(unittest.TestCase):
    def test_response_feedback_is_persisted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            storage = StorageManager(Path(tmp) / 'knowledge.sqlite3', Path(tmp) / 'checkpoints')
            storage.append_response_feedback('simple', 8, 0.9)

            with storage._connect() as conn:  # noqa: SLF001 - test de persistencia interna
                row = conn.execute('SELECT question_type, response_length, satisfaction FROM response_feedback').fetchone()

            self.assertEqual(row['question_type'], 'simple')
            self.assertEqual(row['response_length'], 8)
            self.assertAlmostEqual(row['satisfaction'], 0.9)


if __name__ == '__main__':
    unittest.main()
