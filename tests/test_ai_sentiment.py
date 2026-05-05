import unittest

from core.ai_sentiment import AISentimentEngine


class AISentimentParsingTests(unittest.TestCase):
    def test_extract_score_from_json_payload(self):
        score = AISentimentEngine._extract_score('{"score": 0.42}')
        self.assertEqual(score, 0.42)

    def test_extract_score_rejects_unrelated_numbers(self):
        text = "confidence=87 and regime=2; no explicit score provided"
        score = AISentimentEngine._extract_score(text)
        self.assertIsNone(score)

    def test_extract_score_accepts_explicit_assignment(self):
        score = AISentimentEngine._extract_score("sentiment_score: -0.33")
        self.assertEqual(score, -0.33)

    def test_extract_score_rejects_out_of_range_value(self):
        score = AISentimentEngine._extract_score('{"score": 1.7}')
        self.assertIsNone(score)


if __name__ == "__main__":
    unittest.main()
