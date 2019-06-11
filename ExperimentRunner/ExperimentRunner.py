import numpy as np
import pandas as pd
import ipyparallel as ipp
import json
import functools
import time
import seaborn

class Parameter:
    def __init__(self, name="Test", space=None, default=0, values=set()):
        """
        generate one parameter with given defaultgs
        :param name: name of the parameter
        :param space: will be used to create a set of values like np.linspace(...) * default
        :param default: default value for the parameter
        :param values: set of values for the parameter
        """
        self.name = name
        self.space = space
        self.default = default
        self.values = {default} # set of values for this param
        if space is not None:
            for x in self.space:
                self.values.add(x * default)
        self.values = self.values.union(values)


def run_task(function, parameters, kwargs):
    """
    pass parameters as (real) kwargs, not as dict
    add parameters to resulting dataframe
    :param kwargs: kwargs for the experiment function
    :return: dataframe with parameter settings included
    """
    results = function(**kwargs)
    # add parameter values to dataframe
    t = time.time()
    for k, v in kwargs.items():
        if k in [p.name for p in parameters]:
            results[k] = v
    results["seed"] = kwargs["seed"]
    t = time.time() - t
    results["task_time"] = t
    return results

class Experiment:

    rc = None
    lview = None

    def __init__(self, runs=31, seed=None, function=None, parameters=[Parameter()], with_cluster=True):
        self.runs = runs
        self.seed = seed
        self.function = function
        self.reseed()
        self.tasks = []
        self.parameters = parameters
        self.results = pd.DataFrame()
        self.parallel = with_cluster
        if with_cluster and Experiment.rc is None and Experiment.lview is None:

            Experiment.rc = ipp.Client()
            Experiment.lview = Experiment.rc.load_balanced_view()

    @property
    def default_kwargs(self):
        return {p.name: p.default for p in self.parameters}

    def queue_runs_for_kwargs(self, kwargs):
        for _ in range(self.runs):
            kwargs["seed"] = self.random.randint(2 ** 32)
            self.tasks.append(kwargs.copy())

    def generate_tasks(self):
        self.queue_runs_for_kwargs(self.default_kwargs)
        for param in self.parameters:
            kwargs = self.default_kwargs
            for v in param.values:
                if v == param.default:
                    continue
                kwargs[param.name] = v
                self.queue_runs_for_kwargs(kwargs)

    def reseed(self):
        self.random = np.random.RandomState(seed=self.seed)

    def run_map(self, parallel: bool = None, timeout=-1):
        if parallel is not None:
            self.parallel = parallel
        if self.parallel:
            results = Experiment.lview.map_async(functools.partial(run_task, self.function, self.parameters), self.tasks)
            results.wait_interactive(timeout=timeout)
            self.results = pd.concat(results.get())
        else:
            results = map(functools.partial(run_task, self.function, self.parameters), self.tasks)
            self.results = pd.concat(results)
        return self.results

    def save_results(self, filename="test.pkl"):
        pd.to_pickle(self.results, filename)

    def load_results(self, filename="test.pkl"):
        self.results = pd.read_pickle(filename)
        return self.results

    def save_parameters(self, filename="test.json"):
        parameters = [param.__dict__ for param in self.parameters]
        with open(filename, 'w') as json_file:
            json.dump(parameters, json_file)

    def load_parameters(self, filename="test.json"):
        pass

    def explore_parameter(self, data=None, parameters=None, name="Name"):
        """return a dataframe with the variations of one paramafer and all other paramaters at their default value"""
        df = data
        if df is None:
            df = self.results
        if parameters is None:
            parameters = self.parameters

        for parameter in parameters:
            if parameter.name != name and parameter.name in df.keys():
                df = df.loc[df[parameter.name] == parameter.default]
        return df

    def plot_parameter(self, data=None, parameters=None, name="Name", y="collisions", **kwargs):
        """plot a parameter with all other paramaters set to their default value"""
        df_eps_f = self.explore_parameter(data=data, parameters=parameters, name=name)
        return seaborn.catplot(data=df_eps_f, x="step_count", y=y, col=name, sharex=True, sharey=True, **kwargs)  # , kind="box")


if __name__ == "__main__":
    print(f"running")