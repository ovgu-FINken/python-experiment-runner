import numpy as np
import pandas as pd
import ipyparallel as ipp
import json
import functools
import time
import seaborn
import copy

class Parameter:
    def __init__(self, name="Test", space=None, default=0, values=set(), optimize=False, low=None, high=None, traverese_in_optimization=False):
        """
        generate one parameter with given defaults
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
        self.optimize = optimize
        self.low = low
        self.high = high
        self.best = None
        self.traverse_in_optimization = traverese_in_optimization

    def get_data(self):
        data = {"name" : self.name, "default":self.default, "values": list(self.values)}
        if self.low is not None:
            data["low"] = self.low
        if self.high is not None:
            data["high"] = self.high
        if self.optimize:
            data["optimize"] = True
        if self.best is not None:
            data["best"] = self.best
        if self.traverse_in_optimization:
            data["traverse_in_optimization"] = True
        return data

    def set_data(self, data):
        self.name = data["name"]
        self.default = data["default"]
        self.values = set(data["values"])
        if "low" in data:
            self.low = data["low"]
        if "high" in data:
            self.high = data["high"]
        if "best" in data:
            self.best = data["best"]
        if "optimize" in data:
            self.optimize = True
        if "traverse_in_optimization" in data:
            self.traverse_in_optimization = data["traverse_in_optimization"]

    def __str__(self):
        return str(self.__dict__)

    def set_values_from_default(self, space):
        self.values = {self.default}
        self.values.union(space * self.default)


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
    if "log" in kwargs.keys():
        for v in kwargs["log"]:
            results[v] = kwargs[v]
    results["seed"] = kwargs["seed"]
    t = time.time() - t
    results["task_time"] = t
    results["task_id"] = kwargs["task_id"]
    return results

class Experiment:

    rc = None
    lview = None
    parameters: Parameter

    def __init__(self, runs=31, seed=None, function=None, parameters=None, param_file=None, timeout=600, with_cluster=True):
        self.runs = runs
        self.seed = seed
        self.function = function
        self.reseed()
        self.tasks = []
        self.parameters = parameters
        self.timeout = timeout
        if param_file is not None:
            self.load_parameters(param_file)
        assert self.parameters is not None
        self.results = pd.DataFrame()
        self.parallel = with_cluster
        self._task_id = 0
        if with_cluster and Experiment.rc is None and Experiment.lview is None:
            Experiment.rc = ipp.Client()
            Experiment.lview = Experiment.rc.load_balanced_view()

    @property
    def default_kwargs(self):
        return {p.name: p.default for p in self.parameters}

    def queue_runs_for_kwargs(self, kwargs):
        for _ in range(self.runs):
            kwargs["seed"] = self.random.randint(2 ** 32)
            kwargs["task_id"] = self._task_id
            self._task_id += 1
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


    def queue_cross_product(self):
        for kwargs in self.cross_product_kwargs():
            self.queue_runs_for_kwargs(kwargs)

    def cross_product_kwargs(self):
        """generate tasks for the cross-product of the configuration space"""
        tasks = [self.default_kwargs]
        for param in self.parameters:
            print(f"\n\nbefore\n{tasks}\nparameter: {param.name}\nvalues:{param.values}")
            new = tasks.copy()
            tasks = []
            for v in param.values:
                for i, kwargs in enumerate(new):
                    kw = kwargs.copy()
                    kw[param.name] = v
                    new[i] = kw
                tasks = tasks + new
            print(f"after: {tasks}")
        return tasks

    def reseed(self):
        self.random = np.random.RandomState(seed=self.seed)

    def run_map(self, parallel: bool = None, interactive=True):
        if parallel is not None:
            self.parallel = parallel
        if self.parallel:
            results = Experiment.lview.map_async(functools.partial(run_task, self.function, self.parameters), self.tasks)
            if interactive:
                results.wait_interactive(timeout=self.timeout)
            else:
                results.wait(timeout=self.timeout)
            for result in results:
                if not type(result) == pd.DataFrame:
                    if not result.ready():
                        print("result not ready %s" %result)
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
        parameters = [param.get_data() for param in self.parameters]
        with open(filename, 'w') as json_file:
            json.dump(parameters, json_file)

    def load_parameters(self, filename="test.json", use_best_as_default=False):
        with open(filename, 'r') as json_file:
            p = []
            l = json.load(json_file)
            for data in l:
                x = Parameter()
                x.set_data(data)
                if x.best is not None and use_best_as_default:
                    x.default = x.best
                p.append(x)
            self.parameters = p

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

class Optimizer(Experiment):
    def __init__(self, evaluation_function=None, population_size=10, pso_c1=1.4, pso_c2=1.4, pso_w = 0.5, **kwargs):
        Experiment.__init__(self, **kwargs)
        self.evaluation_function = evaluation_function
        self.population = []
        self.velocity = []
        self.population_size = population_size
        self.mapping = [param for param in self.parameters if param.optimize]
        self.init_population(population_size)
        self.fitness = [np.inf for _ in range(population_size)]
        self.global_best = self.population[0]
        self.previous_best = self.population
        self.global_best_fitness = np.inf
        self.global_best_identifier = {}
        self.previous_best_fitness = [np.inf for _ in range(population_size)]
        self.generation = 0
        self.pso_c1 = pso_c1
        self.pso_c2 = pso_c2
        self.pso_w = pso_w

    def init_population(self, population_size):
        self.population = np.zeros((population_size, len(self.mapping)))
        # initialize each column of the population matrix with values randomly drawn from [param.min, param.max)
        for j, param in enumerate(self.mapping):
            self.population[:,j] = np.random.uniform(low=param.low, high=param.high, size=population_size)
        self.velocity = np.zeros_like(self.population)



    def run_generation(self):
        self.generation += 1
        self.tasks = []
        self.queue_tasks_for_generation()
        self.run_map()
        for i in range(self.population_size):
            df = self.results.loc[self.results["generation"] == self.generation]
            df = df.loc[df.individual == i]
            fitness = self.evaluation_function(df)
            if fitness <= self.previous_best_fitness[i]:
                self.previous_best_fitness[i] = fitness
                self.previous_best[i] = self.population[i]
                if fitness <= self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best = self.population[i]
                    self.global_best_identifier = {"individual": i, "generation": self.generation}
                    for j, v in enumerate(self.population[i]):
                        self.mapping[j].best = v
                    print(f"found new best: individual {i} @ gen {self.generation} with fitness {fitness}")
            self.fitness[i] = fitness
        # PSO update -- velocity
        for i in range(self.population_size):
            self.velocity[i] = self.pso_c1 * np.multiply(np.random.rand(len(self.mapping)), (self.global_best - self.population[i])) +\
                                self.pso_c2 * np.multiply(np.random.rand(len(self.mapping)), (self.previous_best[i] - self.population[i]))+\
                                self.pso_w  * self.velocity[i]

        # PSO update -- position
        self.old = self.population.copy()
        self.population = self.population + self.velocity

        # clamp values
        for j, param in enumerate(self.mapping):
            self.population[:,j] = np.clip(self.population[:,j], a_min=param.low, a_max=param.high)
        # calc last velocity
        self.velocity = self.population - self.old

    def queue_tasks_for_generation(self):
        #create the cross product of all variable parameters to find all possible scenarios
        variations = [{}]
        traverse_params = [param for param in self.parameters if param.traverse_in_optimization]
        for p in traverse_params:
            new = variations
            variations = []
            for v in p.values:
                for i in range(len(new)):
                    new[i][p.name] = v
                variations = variations + copy.deepcopy(new)

        #print(variations)
        for variation in variations:
            for i, values in enumerate(self.population):
                kwargs = self.default_kwargs
                kwargs.update(variation)
                kwargs["individual"] = i
                kwargs["generation"] = self.generation
                kwargs["log"] = ["individual", "generation"]
                for j, v in enumerate(values):
                    kwargs[self.mapping[j].name] = v
                self.queue_runs_for_kwargs(kwargs)


if __name__ == "__main__":
    def dummy_run(Foo=0, Bar=0, **_):
        return pd.DataFrame([{"Foobar": np.abs(Foo) + np.abs(Bar)}])

    def dummy_fitness(df):
        return np.abs(df["Foobar"].mean())

    parameters = [
        Parameter(name="Foo", values=range(3), low=-10, high=128, optimize=True),
        Parameter(name="Bar", values=np.linspace(-3, 3), low=0, high=5, optimize=True),
        Parameter(name="Baz", default="a", values=["a", "b", "c"], traverese_in_optimization=True),
        Parameter(name="Pling", values=range(2), traverese_in_optimization=True)
    ]
    optimizer = Optimizer(parameters=parameters, with_cluster=False, function=dummy_run, evaluation_function=dummy_fitness, runs=2)
    optimizer.queue_tasks_for_generation()
    for _ in range(50):
        optimizer.run_generation()
        print(f"fitness values: {optimizer.fitness} \n {optimizer.global_best_fitness}: {optimizer.global_best}\n{optimizer.population}\n\n")
    print([str(p) for p in optimizer.parameters])
    print(optimizer.global_best_identifier)
