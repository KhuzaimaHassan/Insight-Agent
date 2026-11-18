import unittest

import pandas as pd

from gemini_agent import DEFAULT_MODEL, GeminiAgent, MISSING_KEY_MESSAGE
from utils import create_report


PROFILE = {
    "rows": 3,
    "columns": 2,
    "missing_values_pct": 16.7,
    "numerical_columns": ["sales"],
    "categorical_columns": ["region"],
}
DATA = pd.DataFrame({"sales": [100, 200, None], "region": ["north", "north", "south"]})


class FakeResponse:
    text = "First fact\nSecond fact"


class FakeModels:
    def __init__(self):
        self.calls = []

    def generate_content(self, *, model, contents):
        self.calls.append((model, contents))
        return FakeResponse()


class FakeClient:
    def __init__(self, *, api_key):
        self.api_key = api_key
        self.models = FakeModels()


class GeminiAgentTests(unittest.TestCase):
    def test_missing_key_uses_local_analysis(self):
        agent = GeminiAgent(api_key="")

        self.assertFalse(agent.initialized)
        self.assertEqual(agent.status_message, MISSING_KEY_MESSAGE)
        self.assertIn("generated locally", agent.generate_data_summary(PROFILE, DATA))
        self.assertIn("Local profile", agent.generate_insights(DATA, [], PROFILE)[0])
        self.assertIn("AI assistance is unavailable", agent.answer_question("What changed?", PROFILE))
        self.assertGreaterEqual(len(agent.generate_recommendations(PROFILE, [])), 2)

    def test_current_model_uses_google_genai_client_contract(self):
        agent = GeminiAgent(api_key="test-key", client_factory=FakeClient)

        self.assertTrue(agent.initialized)
        self.assertEqual(agent.model_name, DEFAULT_MODEL)
        self.assertEqual(agent.generate_text("Say hello"), "First fact\nSecond fact")
        self.assertEqual(agent.client.models.calls[0], (DEFAULT_MODEL, "Say hello"))

    def test_recommendations_are_retained_in_offline_reports(self):
        agent = GeminiAgent(api_key="")

        recommendations = agent.generate_recommendations(PROFILE, ["The sample has missing sales values."])

        self.assertTrue(any("missing" in recommendation.lower() for recommendation in recommendations))
        report = create_report(DATA, PROFILE, [], [], recommendations=recommendations)
        self.assertIn(recommendations[0], report)


if __name__ == "__main__":
    unittest.main()
