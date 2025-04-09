import operator
import itertools
import random
import csv
import numpy as np
from functools import partial
from deap import algorithms, base, creator, tools, gp

import CommonFunctions

def get_data_typed(file_name):
    data = []
    with open(file_name) as file:
        file_reader = csv.reader(file)
        for row in file_reader:
            data.append([float(row[0]), float(row[1]), float(row[2]), float(row[3]), str(row[4])])
    return data

with open("spambase.csv") as spambase:
    spamReader = csv.reader(spambase)
    spam = list(list(float(elem) for elem in row) for row in spamReader)

def if_then_else(input, output1, output2):
    if input: return output1
    else: return output2


def make_prim_set_typed():
    primset = gp.PrimitiveSetTyped("MAIN", itertools.repeat(float, 4), str, "IN")

    primset.addPrimitive(operator.add, [float, float], float)
    primset.addPrimitive(operator.sub, [float, float], float)
    primset.addPrimitive(operator.mul, [float, float], float)
    primset.addPrimitive(CommonFunctions.protectedDiv, [float, float], float)

    primset.addPrimitive(operator.and_, [bool, bool], bool)
    primset.addPrimitive(operator.or_, [bool, bool], bool)
    primset.addPrimitive(operator.not_, [bool], bool)

    primset.addPrimitive(operator.lt, [float, float], bool)
    primset.addPrimitive(operator.gt, [float, float], bool)

    primset.addPrimitive(if_then_else, [bool, float, float], float)
    primset.addPrimitive(if_then_else, [bool, str, str], str)

    primset.addEphemeralConstant("randInt", partial(random.randint, -10, 10), int)
    primset.addTerminal(False, bool)
    primset.addTerminal(True, bool)
    
    primset.addTerminal("Iris-setosa", str)
    primset.addTerminal("Iris-versicolor", str)
    primset.addTerminal("Iris-virginica", str)
    return primset

def make_creator_typed(primset):
    creator.create("Fitness", base.Fitness, weights=(1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.Fitness, pset=primset)

def eval_typed(individual, toolbox, data):
    func = toolbox.compile(expr=individual)
    result = sum(func(*flower[:4]) == flower[4] for flower in data)
    return result,

def make_toolbox_typed(primset, data_set):
    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=primset, min_=1, max_=3)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=primset)
    toolbox.register("evaluate", eval_typed, toolbox=toolbox, data=data_set)
    toolbox.register("select", tools.selTournament, tournsize=9)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mutation", gp.genFull, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mutation, pset=primset)
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter('height'), max_value=5))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter('height'), max_value=5))
    return toolbox

def typed_reg(data):
    primset = make_prim_set_typed()
    make_creator_typed(primset)
    toolbox = make_toolbox_typed(primset, data)
    stats = CommonFunctions.statistics()
    pop = toolbox.population(n=200)
    hof = tools.HallOfFame(1)
    pop, log = gp.harm(pop, toolbox, 0.5, 0.1, ngen=50, alpha=0.05, beta=10, gamma=0.25, rho=0.9, stats=stats, halloffame=hof, verbose=True)
    best = hof.items[0]
    print(best)
    CommonFunctions.graph(best)

