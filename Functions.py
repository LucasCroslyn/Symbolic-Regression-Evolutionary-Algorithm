import operator
import math
import random
import numpy
import matplotlib.pyplot as plt
from deap import algorithms, base, creator, tools, gp


def make_prim_set(independent_variables):
    def protectedDiv(left, right):
        try:
            return left / right
        except ZeroDivisionError:
            return 1

    primset = gp.PrimitiveSet("MAIN", independent_variables)
    primset.addPrimitive(operator.add, 2)
    primset.addPrimitive(operator.sub, 2)
    primset.addPrimitive(operator.mul, 2)
    primset.addPrimitive(protectedDiv, 2)
    primset.addPrimitive(operator.neg, 1)
    primset.addPrimitive(math.cos, 1)
    primset.addPrimitive(math.sin, 1)
    primset.addEphemeralConstant("rand101", lambda: random.randint(-1, 1))
    return primset
