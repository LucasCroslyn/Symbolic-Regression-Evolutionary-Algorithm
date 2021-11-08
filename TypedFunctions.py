import operator
import math
import itertools
import random
import numpy as np
from deap import algorithms, base, creator, tools, gp
import networkx as nx
import matplotlib.pyplot as plt

# Possibly make it do a boolean return for a tree and create a function/tree for each of the possible flowers.
# So one tree to create a function for when its versicolor, etc instead of one tree for all


def get_data_typed(file_name):
    data = np.genfromtxt(file_name, dtype=None, delimiter=',')
    return data


def protectedDiv(left, right):
    if right == 0:
        return 1
    return left/right


def if_then_else(input, output1, output2):
    if input: return output1
    else: return output2


def make_prim_set_typed():
    primset = gp.PrimitiveSetTyped("MAIN", itertools.repeat(float, 4), str, "IN")
    primset.addTerminal("Iris-setosa", str)
    primset.addTerminal("Iris-versicolor", str)
    primset.addTerminal("Iris-virginica", str)

    primset.addPrimitive(operator.add, [float, float], float)
    primset.addPrimitive(operator.sub, [float, float], float)
    primset.addPrimitive(operator.mul, [float, float], float)
    primset.addPrimitive(protectedDiv, [float, float], float)

    primset.addPrimitive(operator.and_, [bool, bool], bool)
    primset.addPrimitive(operator.or_, [bool, bool], bool)
    primset.addPrimitive(operator.not_, [bool], bool)

    primset.addPrimitive(operator.lt, [float, float], bool)
    primset.addPrimitive(operator.gt, [float, float], bool)

    primset.addPrimitive(if_then_else, [bool, str, str], str)

    primset.addEphemeralConstant("randfloat", lambda: round(random.uniform(0, 10), 1), float)
    primset.addTerminal(False, bool)
    primset.addTerminal(True, bool)
    return primset


def make_creator_typed(primset):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)


def make_toolbox_typed(primset, data_set):
    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=primset, min_=1, max_=2)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=primset)
    toolbox.register("evaluate", eval_typed, toolbox=toolbox, data=data_set)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mutation", gp.genFull, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mutation, pset=primset)
    toolbox.decorate("mate", gp.staticLimit(key=len, max_value=50))
    toolbox.decorate("mutate", gp.staticLimit(key=len, max_value=50))
    return toolbox


def eval_typed(individual, toolbox, data):
    func = toolbox.compile(expr=individual)
    result = sum(str(func(*list(flower)[:4])) == str(list(flower)[4].decode('UTF-8')) for flower in list(data))
    return result,


def statistics_typed():
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("Std", np.std)
    stats.register("Min", np.min)
    stats.register("Max", np.max)
    return stats


def typed_reg(data):
    primset = make_prim_set_typed()
    make_creator_typed(primset)
    toolbox = make_toolbox_typed(primset, data)
    stats = statistics_typed()
    pop = toolbox.population(n=100)
    hof = tools.HallOfFame(1)
    pop, log = algorithms.eaSimple(pop, toolbox, 0.7, 0.05, 20, stats=stats, halloffame=hof, verbose=True)
    best = hof.items[0]
    print(best)
    print(len(best))
    nodes, edges, labels = gp.graph(best)
    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    pos = nx.nx_agraph.graphviz_layout(g, prog="dot")
    nx.draw_networkx_nodes(g, pos)
    nx.draw_networkx_edges(g, pos)
    nx.draw_networkx_labels(g, pos, labels)
    plt.show()
