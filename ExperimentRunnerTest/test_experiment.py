from unittest import TestCase
from ExperimentRunner.ExperimentRunner import Parameter, Experiment

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

