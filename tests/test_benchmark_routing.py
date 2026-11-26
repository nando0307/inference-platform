import unittest

from benchmarks.routing import POLICIES, SCENARIOS, simulate


class RoutingSimulationTests(unittest.IsolatedAsyncioTestCase):
    async def test_identical_seed_and_arrivals_across_policies(self):
        for scenario in SCENARIOS:
            results = [await simulate(policy, scenario, seed=17) for policy in POLICIES]
            self.assertTrue(all(result["evidence"] == "simulation" for result in results))
            # Independent seed-17 oracle from the documented 6s Poisson workloads;
            # catches a simulator that silently drops arrivals regardless of repeatability.
            expected_arrivals = {"saturation": 104}.get(scenario, 27)
            self.assertTrue(all(sum(result["outcomes"][key] for key in
                                    ("reserved", "server_busy", "not_ready")) == expected_arrivals
                                for result in results))
            self.assertTrue(all(sum(result["assignments"].values()) == result["outcomes"]["reserved"]
                                for result in results))
        self.assertGreater(results[0]["outcomes"]["server_busy"], 0)


if __name__ == "__main__":
    unittest.main()
