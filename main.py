from SymRegFunctions import *
from TypedFunctions import *

#data = get_data("d2.csv")
#independent_variables = data[:, :-1]
#dependent_variables = data[:, -1]
#symReg(independent_variables, dependent_variables)

data = get_data_typed("iris.csv")
typed_reg(data)
