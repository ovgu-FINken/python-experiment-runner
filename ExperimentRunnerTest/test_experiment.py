from unittest import TestCase
from ExperimentRunner.ExperimentRunner import Parameter, Experiment

import pandas as pd
import numpy as np


def testfunction(param1, param2, seed):
    return pd.DataFrame([{"result1": param1**2, "result2": param1 + seed, "result3" :f"{param1}{param2}"}])

class TestExperiment(TestCase):
    def test_generate_tasks_length(self):
        self.parameterFoo = Parameter(name="foo", space=[1, 2], default=3)
        self.parameterBar = Parameter(name="bar", space=range(10), default=1.0)
        self.runs = 3

        e = Experiment(runs=self.runs, seed=42, parameters=[self.parameterFoo], with_cluster=False)
        e.generate_tasks()
        self.assertEqual(len(e.tasks), self.runs * (len(self.parameterFoo.space)))
        e.tasks = []
        e.parameters = [self.parameterFoo, self.parameterBar]
        e.generate_tasks()
        self.assertEqual(len(e.tasks), self.runs * (len(self.parameterFoo.space) + len(self.parameterBar.space) - 1))

    def test_save_results(self):
        e1 = Experiment(with_cluster=False)
        e1.results = pd.DataFrame([{i: i ** 2 for i in range(10)}])
        e1.save_results(filename="test.pkl")
        e2 = Experiment(with_cluster=False)
        e2.load_results(filename="test.pkl")
        self.assertTrue(e1.results.equals(e2.results), msg="Data should be the same after reload")

    def test_run_map(self):
        parameters = [
            Parameter(name="param1", space=[0,1,2]),
            Parameter(name="param2", default="a", values=["b", "c", "d"])
        ]
        e = Experiment(runs=3, parameters=parameters, with_cluster=False, seed=0, function=testfunction)
        e.generate_tasks()
        task_number = len(e.tasks)
        df = pd.DataFrame(e.tasks)
        e.run_map()

        self.assertEqual(len(e.results), task_number, msg="result number must equal task number")
        self.assertTrue((e.results["result1"] == e.results["param1"]**2).all())
        self.assertTrue((e.results["result2"] == e.results["param1"]+e.results["seed"]).all())
        self.assertTrue(len(e.results["task_time"]) > 0, msg="task_time needs to be present in data")
