import unittest

from src.routing import Candidate, select


class RoutingTests(unittest.TestCase):
    def test_ties_and_immutable_smooth_weighted_sequence(self):
        candidates = (Candidate("b", 0, 0, 1), Candidate("a", 0, 0, 3))
        for policy in ("least-inflight", "least-queue-depth", "round-robin", "weighted-capacity"):
            self.assertEqual(select(candidates, policy, None, {})[0], "a")
        credits = {}
        selections = []
        for _ in range(8):
            chosen, credits = select(candidates, "weighted-capacity", None, credits)
            selections.append(chosen)
        self.assertEqual(selections, ["a", "a", "b", "a"] * 2)
        self.assertEqual(credits, {"a": 0, "b": 0})
        self.assertEqual(select(candidates, "round-robin", "a", {})[0], "b")
        self.assertEqual(select(candidates, "round-robin", "b", {})[0], "a")
        self.assertEqual(select((candidates[1],), "weighted-capacity", None, credits)[0], "a")
        self.assertEqual(credits, {"a": 0, "b": 0})

    def test_load_policies_use_only_eligible_snapshot(self):
        candidates = (Candidate("a", 2, 0, 1), Candidate("b", 1, 1, 1))
        self.assertEqual(select(candidates, "least-inflight", None, {})[0], "b")
        self.assertEqual(select(candidates, "least-queue-depth", None, {})[0], "a")
        with self.assertRaises(ValueError):
            select((), "round-robin", None, {})


if __name__ == "__main__":
    unittest.main()
