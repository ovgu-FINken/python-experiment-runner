import numpy as np
import pandas as pd
import ipyparallel as ipp

class Parameter:
    def __init__(self, name="Test", space=np.linspace(0, 5, num=10), default=0):
        self.name = name
        self.space = space
        self.default = default


class Experiment:
    def __init__(self, runs=31, seed=None, function=None, parameters=[Parameter()]):
        self.runs = runs
        self.seed = seed
        self.function = function
        self.reseed()
        self.tasks = []
        self.parameters = parameters
        self.rc = ipp.Client()

    @property
    def default_kwargs(self):
        return {p.name: p.default for p in self.parameters}

    def generate_tasks(self):
        for param in self.parameters:
            kwargs = self.default_kwargs
            for v in param.space:
                kwargs[param.name] = v
                self.reseed()
                for _ in range(self.runs):
                    kwargs["seed"] = self.random.randint(2 ** 32)
                    self.tasks.append(kwargs.copy())

    def reseed(self):
        self.random = np.random.RandomState(seed=self.seed)

    def run_sequential(self):
        results = []
        for task in self.tasks:
            results.append(self.function(task))
        self.results = pd.concat(results, ignore_index=True)

    def run_map(self):
        v = self.rc.load_balanced_view()
        results = v.map_async(self.function, self.tasks)
        results.wait_interactive()
        self.results = pd.concat(results.get())
        return self.results

if __name__ == "__main__":
    print(f"running")