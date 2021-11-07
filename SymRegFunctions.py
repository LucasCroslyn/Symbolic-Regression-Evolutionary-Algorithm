import operator
import math
import random
import numpy as np
from deap import algorithms, base, creator, tools, gp


def get_data(file_name):
    data = np.loadtxt(file_name, dtype=float, delimiter=',')
    return data


def protectedDiv(left, right):
    if right == 0:
        return 1
    return left/right


def make_prim_set(num_independent_variables):
    primset = gp.PrimitiveSet("MAIN", num_independent_variables)
    primset.addPrimitive(operator.add, 2)
    primset.addPrimitive(operator.sub, 2)
    primset.addPrimitive(operator.mul, 2)
    primset.addPrimitive(protectedDiv, 2)
    primset.addPrimitive(operator.neg, 1)
    primset.addPrimitive(math.cos, 1)
    primset.addPrimitive(math.sin, 1)
    primset.addEphemeralConstant("randfloat", lambda: round(random.uniform(0, 10), 1))
    return primset


def make_creator(primset):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)


def make_toolbox(primset, independent_variables, dependent_variables):
    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=primset, min_=1, max_=5)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=primset)
    toolbox.register("evaluate", eval_symbolic_regression, x_points=independent_variables, y_points=dependent_variables, toolbox=toolbox)
    toolbox.register("select", tools.selTournament, tournsize=4)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mutation", gp.genFull, min_=0, max_=2)  # Should go over this one
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mutation, pset=primset)
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
    return toolbox


def eval_symbolic_regression(individual, x_points, y_points, toolbox):
    callable_function = toolbox.compile(expr=individual)
    test = ((callable_function(*x_points[i]) - y_points[i])**2 for i in range(len(x_points)))
    return math.fsum(test) / len(x_points),


def statistics():
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("Avg", np.mean)
    mstats.register("Std", np.std)
    mstats.register("Min", np.min)
    mstats.register("Max", np.max)
    return mstats


def symReg(independent, dependent):
    primset = make_prim_set(independent.shape[1])
    make_creator(primset)
    toolbox = make_toolbox(primset, independent, dependent)
    test_indiv = toolbox.individual()
    mstats = statistics()
    print(test_indiv)
    pop = toolbox.population(n=50)
    hof = tools.HallOfFame(1)
    pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.1, 100, stats=mstats, halloffame=hof, verbose=True)
    best = hof.items[0]
    print(best)