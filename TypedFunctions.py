import operator
import itertools
import random
import csv
import numpy as np
from functools import partial
from deap import algorithms, base, creator, tools, gp

import CommonFunctions

def get_data_typed(file_name):
    '''
    Read in the data from a csv file. Parameters must be in the correct format (no headers, 4 floats and then string)

    :param file_name: String for the file's name/location
    :return: Returns the data in a 2D Array
    '''
    data = []
    with open(file_name) as file:
        file_reader = csv.reader(file)
        for row in file_reader:
            data.append([float(row[0]), float(row[1]), float(row[2]), float(row[3]), str(row[4])])
    return data

def if_then_else(input, output1, output2):
    '''
    A function that is made to be added to the Tree's set of possible nodes. 

    :param input: The input node must be a Boolean.
    :param output1: Will be returned if the input Bool is True
    :param output2: Will be returned if the input Bool is False
    :return: output1/2 depending on input
    '''
    if input: return output1
    else: return output2

def make_prim_set_typed():
    '''
    Makes all of the possible nodes for the generated Trees. Input Nodes and the String terminals are preset to this specific file

    :return: Set of all possible nodes for a generated tree
    '''
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
    '''
    Makes the needed functions to define how individuals in the generated population are structured

    :param primset: A set of all possible nodes for the generated Trees
    '''
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax, pset=primset)

def eval_typed(individual, toolbox, data):
    '''
    Calculated the fitness value for an individual tree. Turns the tree into a callable function first
    The fitness is the number of correct classifications of flowers for the dataset

    :param individual: The tree to evaluate
    :param toolbox: The toolbox that contains various functions
    :param data: The dataset used to evaluate the individual tree
    :return: Returns the fitness value (in a tuple)
    '''
    func = toolbox.compile(expr=individual)
    result = sum(func(*flower[:4]) == flower[4] for flower in data)
    return result,

def make_toolbox_typed(primset, data_set):
    '''
    Makes the toolbox which contains various functions needed to perform the genetic algorithm

    :param primset: A set of all possible nodes for each generated tree.
    :param data_set: The dataset used to evaluate each generated tree.
    :return: Returns the toolbox with all of the various functions needed added
    '''
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

def typed_reg(data, n_gen, n_pop):
    '''
    Performs a simple genetic algorithm given the dataset to classify a flower.

    :param data: The dataset used to evaluate each generated tree.
    :return: Returns the best tree generated over the entire algorithm and the toolbox with the functions used in the genetic algorithm
    '''
    primset = make_prim_set_typed()
    make_creator_typed(primset)
    toolbox = make_toolbox_typed(primset, data)
    stats = CommonFunctions.statistics()
    pop = toolbox.population(n=n_pop)
    hof = tools.HallOfFame(1)
    # Harm is a bloat control algorithm to do a bit of limiting the number of nodes in the Trees
    pop, log = gp.harm(pop, toolbox, 0.5, 0.1, ngen=n_gen, alpha=0.05, beta=10, gamma=0.25, rho=0.9, stats=stats, halloffame=hof, verbose=True)
    best = hof.items[0]
    CommonFunctions.graph(best)
    return best, toolbox

