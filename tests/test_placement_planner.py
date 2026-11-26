"""CPU checks for dry-run planning and identical-condition simulated comparisons."""

from dataclasses import replace
import json
import math
import unittest

from benchmarks.placement_simulation import ROOT, compare, configurations, run, sensitivity
from benchmarks.workload_generator.config import load_scenario
from benchmarks.workload_generator.trace import generate_trace
from src.placement_planner import CONSERVATIVE, SHORT_EXPERIMENT, History, Snapshot, plan


class PlacementPlannerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.configs = configurations()
        cls.workers = {w.id: w for config in cls.configs.values() for w in config.workers}
        cls.models = [model.id for model in cls.configs['A'].models.values()]

    def snapshot(self, t=10, demand=None, memory=None):
        return Snapshot(1, t, t, dict(zip(self.models, demand or (1.5, 0.05))),
                        dict.fromkeys(self.models, 100.0), t,
                        {id: (0.7 if w.model == self.models[0] else 0.45) for id, w in self.workers.items()},
                        'assumed', {id: True for id in self.workers},
                        {id: (.35 if w.model == self.models[0] else .4) for id, w in self.workers.items()} if memory is None else memory,
                        t, 'assumed', shared_gpu_factor=.7)

    def test_configs_verified_not_inferred(self):
        expected = {'A': {'small-0', 'large-1'}, 'B': {'small-0', 'large-0', 'small-1'},
                    'C': {'small-0', 'large-0', 'large-1'}, 'D': {'small-0', 'large-0'}}
        for name, config in self.configs.items():
            self.assertEqual({w.id for w in config.workers}, expected[name])
            self.assertEqual(len(config.hosts['local'].gpu_uuids), 2)
            self.assertEqual(config.models, self.configs['A'].models)

    def test_profiles_and_schema(self):
        self.assertEqual((CONSERVATIVE.ewma_s, CONSERVATIVE.windows, CONSERVATIVE.improvement,
                          CONSERVATIVE.cooldown_s, CONSERVATIVE.residence_s, CONSERVATIVE.memory_reserve),
                         (300, 3, .15, 600, 900, .2))
        self.assertEqual((SHORT_EXPERIMENT.ewma_s, SHORT_EXPERIMENT.interval_s,
                          SHORT_EXPERIMENT.windows, SHORT_EXPERIMENT.improvement,
                          SHORT_EXPERIMENT.cooldown_s, SHORT_EXPERIMENT.residence_s, SHORT_EXPERIMENT.memory_reserve),
                         (30, 10, 3, .15, 60, 90, .2))
        with self.assertRaises(ValueError):
            replace(self.snapshot(), capacity_provenance='unknown')
        with self.assertRaises(ValueError):
            replace(self.snapshot(), demand_at_s=11)
        with self.assertRaises(ValueError):
            replace(self.snapshot(), capacity_rps={'small-0': -1})
        with self.assertRaises(ValueError):
            replace(self.snapshot(), demand_limit_rps=dict.fromkeys(self.models, .01))

    def test_stale_unknown_and_unready_hold_without_zero_fallback(self):
        base = self.snapshot()
        for changed in (replace(base, demand_at_s=None), replace(base, demand_at_s=0, at_s=30),
                        replace(base, demand_rps={self.models[0]: None, self.models[1]: 0}),
                        replace(base, gpu_memory={}), replace(base, worker_ready={})):
            result = plan(self.configs, 'A', changed)
            self.assertEqual(result.action, 'hold')
            self.assertIsNone(result.demand_rps)
        with self.assertRaises(ValueError):
            plan(self.configs, 'A', base, History(last_at_s=base.at_s))

    def test_transition_and_stability_units(self):
        history = History()
        decisions = []
        for t in (10, 20, 30):
            decision = plan(self.configs, 'A', self.snapshot(t), history)
            decisions.append(decision)
            history = decision.history
        self.assertEqual([d.action for d in decisions], ['hold', 'hold', 'recommend'])
        decision = decisions[-1]
        self.assertEqual(decision.candidate, 'B')
        self.assertEqual(decision.schema_version, 1)
        self.assertEqual(json.loads(json.dumps(decision.to_dict()))['profile']['name'], 'short-experiment')
        self.assertEqual(decision.provenance['capacity'], 'assumed')
        # Independent A→B oracle: target small capacity is 1.19 RPS; the three
        # pre-step capacities are .70, .49 and .98, giving 4.9 + 7 + 2.1 = 14 requests.
        self.assertAlmostEqual(decision.transition_lost_requests, 14.0)
        self.assertEqual([(step.action, step.worker) for step in decision.transition],
                         [('add', 'large-0'), ('add', 'small-1'), ('remove', 'large-1')])
        for step in decision.transition:
            self.assertEqual({self.workers[id].model for id in step.available_after}, set(self.models))
        # No switch until both 60s cooldown AND 90s residence elapse.
        large_peak_history = History(dict(zip(self.models, (.05, 1.5))), 30, last_change_s=30)
        blocked = plan(self.configs, 'B', self.snapshot(40, (.05, 1.5)), large_peak_history,
                       replace(SHORT_EXPERIMENT, windows=1))
        self.assertEqual(blocked.action, 'hold')
        self.assertIn('residence', blocked.reason)
        at_boundary = plan(self.configs, 'B', self.snapshot(120, (.05, 1.5)), large_peak_history,
                           replace(SHORT_EXPERIMENT, windows=1))
        self.assertEqual(at_boundary.action, 'recommend')
        self.assertEqual(at_boundary.candidate, 'C')

    def test_ewma_gaps_threshold_and_horizon(self):
        first = plan(self.configs, 'A', self.snapshot(10, (0, 0)))
        self.assertEqual(first.demand_rps, dict(zip(self.models, (0, 0))))
        second = plan(self.configs, 'A', self.snapshot(20, (1, 0)), first.history)
        self.assertAlmostEqual(second.demand_rps[self.models[0]], 1 - math.exp(-10 / 30))
        gap = plan(self.configs, 'A', self.snapshot(40, (1, 0)), second.history)
        self.assertEqual(gap.demand_rps[self.models[0]], 1)
        high_bar = replace(SHORT_EXPERIMENT, windows=1, improvement=1)
        self.assertEqual(plan(self.configs, 'A', self.snapshot(), profile=high_bar).action, 'hold')
        short_horizon = replace(SHORT_EXPERIMENT, windows=1, horizon_s=1)
        self.assertIn('horizon', plan(self.configs, 'A', self.snapshot(), profile=short_horizon).rejected['B'])

    def test_infeasible_steady_and_transition(self):
        memory = self.snapshot().gpu_memory | {'large-0': .5, 'small-0': .5}
        result = plan(self.configs, 'A', self.snapshot(memory=memory))
        self.assertIn('B', result.rejected)
        # Swap the only replica of each model across GPUs: both endpoints fit,
        # but no first move fits memory without losing a required model.
        swapped = replace(self.configs['B'], workers=tuple(w for w in self.configs['B'].workers if w.id != 'small-0'))
        memory = self.snapshot().gpu_memory | {'small-0': .56, 'small-1': .56, 'large-0': .25, 'large-1': .25}
        result = plan({'A': self.configs['A'], 'B': swapped}, 'A', self.snapshot(memory=memory))
        self.assertIn('transition', result.rejected['B'])
        self.assertIn('B', result.scores)

    def test_simulator_reproducibility_causality_and_sensitivity(self):
        scenario = load_scenario(ROOT / 'configs/workloads/small_peak.json')
        first = run(scenario, self.configs)
        self.assertEqual(first, run(scenario, self.configs))
        self.assertEqual(first['kind'], 'simulation')
        self.assertEqual(first['assumptions']['capacity_provenance'], 'assumed')
        comparison = compare(scenario, self.configs)
        self.assertEqual(len(comparison['runs']), 5)
        self.assertEqual({x['offered'] for x in comparison['runs']}, {first['offered']})
        # An altered future trace suffix cannot alter any decision before that suffix.
        trace = generate_trace(scenario)
        future = [{**row, 'model': self.models[1]} if row['scheduled_offset_s'] >= 80 else row for row in trace]
        a = run(scenario, self.configs, records=trace)
        b = run(scenario, self.configs, records=future)
        self.assertEqual([r for r in a['recommendations'] if r['at_s'] <= 80],
                         [r for r in b['recommendations'] if r['at_s'] <= 80])
        micro = load_scenario(ROOT / 'configs/workloads/microburst.json')
        self.assertEqual(run(micro, self.configs)['transitions'], 0)
        stale = run(scenario, self.configs, stale_windows=range(1, 1000))
        missing = run(scenario, self.configs, missing_windows=range(1, 1000))
        missing_comparison = compare(scenario, self.configs, missing_windows=range(1, 4))
        stale_comparison = compare(scenario, self.configs, stale_windows=range(1, 4))
        self.assertEqual(missing['transitions'], 0)
        self.assertEqual(stale['transitions'], 0)
        self.assertEqual(stale['final'], 'A')
        for result in (missing_comparison, stale_comparison):
            self.assertEqual({run['offered'] for run in result['runs']}, {first['offered']})
            adaptive_run = result['runs'][-1]
            self.assertEqual(adaptive_run['final'], 'A')
            self.assertEqual(adaptive_run['transitions'], 0)
            self.assertIn('missing or stale observation', {r['reason'] for r in adaptive_run['recommendations']})
        self.assertEqual(len(sensitivity(self.configs, ('small_peak', 'large_peak'))['results']), 10)


if __name__ == '__main__':
    unittest.main()
