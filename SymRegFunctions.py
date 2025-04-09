import operator
import math
import random
import numpy as np
from functools import partial
from deap import algorithms, base, creator, tools, gp

import CommonFunctions


def get_data(file_name):
    data = np.loadtxt(file_name, dtype=float, delimiter=',')
    return data

def make_prim_set(num_independent_variables):
    primset = gp.PrimitiveSet("MAIN", num_independent_variables)
    primset.addPrimitive(operator.add, 2)
    primset.addPrimitive(operator.sub, 2)
    primset.addPrimitive(operator.mul, 2)
    primset.addPrimitive(CommonFunctions.protectedDiv, 2)
    primset.addPrimitive(operator.neg, 1)
    primset.addPrimitive(math.cos, 1)
    primset.addPrimitive(math.sin, 1)
    primset.addEphemeralConstant(name="randInt", ephemeral=partial(random.randint, -10, 10))
    return primset


def make_creator(primset):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin, pset=primset)

def eval_symbolic_regression(individual, x_points, y_points, toolbox):
    callable_function = toolbox.compile(expr=individual)
    test = ((callable_function(*x_points[i]) - y_points[i])**2 for i in range(len(x_points)))
    return  math.fsum(test) / len(x_points), 

def make_toolbox(primset, independent_variables, dependent_variables):
    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=primset, min_=1, max_=3)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=primset)
    toolbox.register("evaluate", eval_symbolic_regression, x_points=independent_variables, y_points=dependent_variables, toolbox=toolbox)
    toolbox.register("select", tools.selTournament, tournsize=9)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("mutate", gp.mutNodeReplacement, pset=primset)
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter('height'), max_value=3))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter('height'), max_value=3))
    return toolbox

def symReg(independent, dependent):
    primset = make_prim_set(independent.shape[1])
    make_creator(primset)
    toolbox = make_toolbox(primset, independent, dependent)
    stats = CommonFunctions.statistics()
    pop = toolbox.population(n=200)
    hof = tools.HallOfFame(1)
    pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.1, 25, stats=stats, halloffame=hof, verbose=True)
    print(hof.items[0])
    best = hof.items[0]
    print(best)
    CommonFunctions.graph(best)
    return best, toolbox