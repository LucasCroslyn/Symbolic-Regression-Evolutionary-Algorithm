import operator
import math
import random
import numpy as np
from functools import partial
from deap import algorithms, base, creator, tools, gp

import CommonFunctions


def get_data(file_name):
    '''
    Read in the data from a csv file

    :param file_name: String for the file's name/location
    :return: Returns the data in a 2D Array
    '''
    data = np.loadtxt(file_name, dtype=float, delimiter=',')
    return data

def make_prim_set(num_independent_variables):
    '''
    Makes all of the possible nodes for the generated Trees.

    :param num_independent_variables: Number of input variables
    :return: Returns the set of all possible nodes
    '''
    primset = gp.PrimitiveSet("MAIN", num_independent_variables)
    primset.addPrimitive(operator.add, 2)
    primset.addPrimitive(operator.sub, 2)
    primset.addPrimitive(operator.mul, 2)
    primset.addPrimitive(CommonFunctions.protectedDiv, 2)
    primset.addPrimitive(operator.neg, 1)
    primset.addPrimitive(math.cos, 1)
    primset.addPrimitive(math.sin, 1)
    primset.addEphemeralConstant(name="randInt", ephemeral=partial(random.randint, -10, 10)) #Able to add ints from -10 to 10 in the Tree
    return primset


def make_creator(primset):
    '''
    Makes the needed classes to define how individuals in the generated population are structured

    :param primset: A set of all possible nodes for the generated Trees
    '''
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin, pset=primset)

def eval_symbolic_regression(individual, x_points, y_points, toolbox):
    '''
    Calculates the fitness value for the individual Tree. Turns the Tree into a callable function first.

    :param individual: The Tree to evaluate
    :param x_points: The inputs for each sample. Outer Array is for the population. Inner Array is for each sample.
    :param y_points: The true result of what the actual equation makes given the inputs (independent x_points)
    :param toolbox: The toolbox that contains various functions
    :return: Returns the fitness value (in a tuple). The fitness value is the average mean squared error for each sample in the dataset
    '''

    callable_function = toolbox.compile(expr=individual)
    test = ((callable_function(*x_points[i]) - y_points[i])**2 for i in range(len(x_points)))
    return  math.fsum(test) / len(x_points), 

def make_toolbox(primset, independent_variables, dependent_variables):
    '''
    Makes the toolbox which contains various functions needed to perform the genetic algorithm

    :param primset: A set of all possible nodes for each generated tree.
    :param independent_variables: The inputs for each sample. Outer Array is for the population. Inner Array is for each sample.
    :param dependent_variables: The true result of what the actual equation makes given the inputs (independent variables)
    :return: Returns the toolbox with all of the various functions needed added
    '''
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

def symReg(independent, dependent, n_gen, n_pop):
    '''
    Performs a simple symbolic regression genetic algorithm given the dataset.

    :param independent: The inputs for each sample. Outer Array is for the population. Inner Array is for each sample.
    :param dependent: The true result of what the actual equation makes given the inputs (independent variables)
    :return: Returns the best tree generated over the entire algorithm and the toolbox with the functions used in the genetic algorithm
    '''
    primset = make_prim_set(independent.shape[1])
    make_creator(primset)
    toolbox = make_toolbox(primset, independent, dependent)
    stats = CommonFunctions.statistics()
    pop = toolbox.population(n=n_pop)
    hof = tools.HallOfFame(1)
    pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.1, ngen=n_gen, stats=stats, halloffame=hof, verbose=True)
    print(hof.items[0])
    best = hof.items[0]
    print(best)
    CommonFunctions.graph(best)
    return best, toolbox