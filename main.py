from SymRegFunctions import *

data = get_data("d1.csv")
independent_variables = data[:, :-1]
dependent_variables = data[:, -1]
print(independent_variables)
symReg(independent_variables, dependent_variables)
#plt.plot(independent_variables, dependent_variables, "ob")
#plt.show()