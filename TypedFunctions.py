import operator
import math
import random
import numpy as np
from deap import algorithms, base, creator, tools, gp


def make_prim_set_typed(num_independent_variables):
    primset = gp.PrimitiveSetTyped("MAIN", )