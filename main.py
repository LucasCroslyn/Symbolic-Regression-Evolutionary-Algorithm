from SymRegFunctions import *
from TypedFunctions import *

# This stuff is for the symbolic regression
data = get_data("d2.csv")
independent_variables = data[:, :-1]
dependent_variables = data[:, -1]
symReg(independent_variables, dependent_variables)

# This stuff is for the typed problem, in this case the iris classification problem
# Uncomment the two lines below to run (and comment out the section above as it doesn't need to run for it)
# data = get_data_typed("iris.csv")
# typed_reg(data, "Iris-versicolor")
