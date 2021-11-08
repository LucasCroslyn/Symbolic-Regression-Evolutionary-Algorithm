import operator
import math
import random
import numpy as np
from deap import algorithms, base, creator, tools, gp
import networkx as nx
import matplotlib.pyplot as plt


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
    primset.addEphemeralConstant("randint", lambda: random.randint(1, 11))
    return primset


def make_creator(primset):
    creator.create("FitnessMin", base.Fitness, weights=(-5.0, -1.0))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)


def make_toolbox(primset, independent_variables, dependent_variables):
    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=primset, min_=1, max_=3)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=primset)
    toolbox.register("evaluate", eval_symbolic_regression, x_points=independent_variables, y_points=dependent_variables, toolbox=toolbox)
    toolbox.register("select", tools.selNSGA2)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mutation", gp.genFull, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mutation, pset=primset)
    toolbox.decorate("mate", gp.staticLimit(key=len, max_value=50))
    toolbox.decorate("mutate", gp.staticLimit(key=len, max_value=50))
    return toolbox


def eval_symbolic_regression(individual, x_points, y_points, toolbox):
    callable_function = toolbox.compile(expr=individual)
    test = ((callable_function(*x_points[i]) - y_points[i])**2 for i in range(len(x_points)))
    return math.fsum(test) / len(x_points), len(individual)


def statistics():
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("Std", np.std, axis=0)
    stats.register("Min", np.min, axis=0)
    stats.register("Max", np.max, axis=0)
    return stats


def symReg(independent, dependent):
    primset = make_prim_set(independent.shape[1])
    make_creator(primset)
    toolbox = make_toolbox(primset, independent, dependent)
    test_indiv = toolbox.individual()
    stats = statistics()
    print(test_indiv)
    pop = toolbox.population(n=200)
    hof = tools.ParetoFront()
    pop, log = algorithms.eaSimple(pop, toolbox, 0.3, 0.05, 200, stats=stats, halloffame=hof, verbose=True)
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
    best_y = []
    bestFunction = toolbox.compile(expr=best)
    for i in range(len(independent)):
        best_y.append(bestFunction(independent[i])[0])
    plt.scatter(independent, best_y, color='orange', s=5)
    plt.scatter(independent, dependent, s=5)
    plt.show()