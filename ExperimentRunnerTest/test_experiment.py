from unittest import TestCase
from ExperimentRunner.ExperimentRunner import Parameter, Experiment

import pandas as pd
import numpy as np

class TestExperiment(TestCase):
    def test_generate_tasks_length(self):
        self.parameterFoo = Parameter(name="foo", space=["A", "B"], default="C")
        self.parameterBar = Parameter(name="bar", space=range(10), default=0)
        self.runs = 3

        e = Experiment(runs=self.runs, seed=42, parameters=[self.parameterFoo], with_cluster=False)
        e.generate_tasks()
        self.assertEqual(len(e.tasks), self.runs * (len(self.parameterFoo.space) + 1))
        e.tasks = []
        e.parameters = [self.parameterFoo, self.parameterBar]
        e.generate_tasks()
        self.assertEqual(len(e.tasks), self.runs * (len(self.parameterFoo.space) + len(self.parameterBar.space)))


    def test_save_results(self):
        e1 = Experiment(with_cluster=False)
        e1.results = pd.DataFrame([{i : i**2 for i in range(10)}])
        e1.save_results(filename="test.pkl")
        e2 = Experiment(with_cluster=False)
        e2.load_results(filename="test.pkl")
        self.assertTrue(e1.results.equals(e2.results), msg="Data should be the same after reload")
