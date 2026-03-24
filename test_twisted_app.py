import unittest

from twisted import extract_state, route_risk_by_state, fallback_chat_answer


class TestTwistedAppLogic(unittest.TestCase):
    def test_extract_state_from_area(self):
        self.assertEqual(extract_state("Dallas County, TX; Tarrant County, TX"), "TX")
        self.assertEqual(extract_state("Unknown"), "Unknown")

    def test_route_risk_by_state(self):
        rows = [
            {"area": "Dallas County, TX"},
            {"area": "Tarrant County, TX"},
            {"area": "Oklahoma County, OK"},
            {"area": "Tulsa County, OK"},
            {"area": "Tulsa County, OK"},
        ]
        out = route_risk_by_state(rows, ["TX", "OK", "KS"])
        counts = {r["state"]: r["active_alerts"] for r in out}
        self.assertEqual(counts["TX"], 2)
        self.assertEqual(counts["OK"], 3)
        self.assertEqual(counts["KS"], 0)

    def test_fallback_chat_answer(self):
        ans = fallback_chat_answer("What is the difference between a watch warning?")
        self.assertIn("watch", ans.lower())
        self.assertIn("warning", ans.lower())


if __name__ == "__main__":
    unittest.main()
