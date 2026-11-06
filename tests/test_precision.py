import unittest

from benchmarks.precision import QUALITY_CASES, score_quality


class PrecisionScoringTests(unittest.TestCase):
    def setUp(self):
        self.cases = {case["name"]: case for case in QUALITY_CASES}

    def test_answer_normalization_preserves_instruction_requirements(self):
        for name, text, expected in (
            ("addition", " 45\n", True),
            ("addition", "44", False),
            ("addition", "The answer is 45.", False),
            ("capital", " PARIS\n", True),
            ("water", "H₂O", True),
            ("water", "H2O2", False),
            ("reverse", "blue green red", True),
            ("reverse", "blue  green red", False),
            ("lowercase", "mixed case", True),
            ("lowercase", "Mixed Case", False),
        ):
            with self.subTest(name=name, text=text):
                self.assertEqual(score_quality(self.cases[name], text, "stop"), expected)

    def test_json_requires_exact_content_and_integer_without_markdown(self):
        case = self.cases["json"]
        self.assertTrue(score_quality(case, ' {"count": 2, "color": "blue"}\n', "stop"))
        for text in (
            '{"color":"blue","count":2,"extra":null}',
            '{"color":"red","count":2}',
            '{"color":"blue","count":"2"}',
            '{"color":"blue","count":2.0}',
            '{"color":"blue"}',
            '["blue",2]',
            'null',
            '{"color":"blue","count":2,}',
            '```json\n{"color":"blue","count":2}\n```',
        ):
            with self.subTest(text=text):
                self.assertFalse(score_quality(case, text, "stop"))

    def test_truncated_correct_answer_is_not_a_completed_task(self):
        self.assertFalse(score_quality(self.cases["addition"], "45", "length"))
        self.assertFalse(score_quality(self.cases["json"], '{"color":"blue","count":2}', "length"))


if __name__ == "__main__":
    unittest.main()
