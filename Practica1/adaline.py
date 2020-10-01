import csv
import numpy as np
from numpy import random

# Cargar los datos
with open('inputs/training_set.csv') as file:
    data_training = list(csv.reader(file, delimiter=',', skipinitialspace=True))

with open('inputs/test_set.csv') as file:
    data_test = list(csv.reader(file, delimiter=',', skipinitialspace=True))

with open('inputs/validation_set.csv') as file:
    data_validation = list(csv.reader(file, delimiter=',', skipinitialspace=True))

# Variables del problema
learning_rate = 0.01

header = data_training[0]

w = random.rand(len(header)-1) * 2 - 1
b = random.rand() * 2 - 1

epoch = 1

stop_condition = False

list_training_errors = []
list_validation_errors = []

list_stop_condition = [np.inf, np.inf, np.inf, np.inf] # Se inicializa con infinitos para que siempre sean menores los errores

#Funciones auxiliares

def getOutput(row):
    x = np.array(list(map(lambda x: float(x), row[:-1])))

    return sum(x * w) + b

def getWeights(row, y):
    x = np.array(list(map(lambda x: float(x), row[:-1])))
    d = float(row[-1])

    return learning_rate * (d - y) * x

def getBias(row, y):
    d = float(row[-1])

    return learning_rate * (d - y)

# Se usa el error absoluto medio
def getError(dataframe, list=[]):
    total = 0
    for row in dataframe[1:]:
        d = float(row[-1])
        y = getOutput(row)

        list.append(y)

        total += abs(d - y)

    return total / (len(dataframe[1:]))

# Se para el bucle si el error nuevo es mayor o igual a los 4 últimos errores
# porque se considera que ya ha convergido el algoritmo
def setStopCondition(error):
    if error >= max(list_stop_condition):
        return True

    list_stop_condition.append(error)
    list_stop_condition.pop(0)

    return False

# Aprendizaje
while not stop_condition:

    # Cálculo de la salida y ajuste de pesos y umbral
    for row in data_training[1:]:
        y = getOutput(row)
        w += getWeights(row, y)
        b += getBias(row, y)

    # Apuntar errores
    list_training_errors.append(getError(data_training))

    error_val = getError(data_validation)
    list_validation_errors.append(error_val)

    stop_condition = setStopCondition(error_val)

    print("Epoch num: ", epoch)

    print("Training...", list_training_errors[-1])

    epoch += 1


# Error sobre el conjunto de test una vez finalizado el aprendizaje
data_test_output = []
print("Error in test: ", getError(data_test, data_test_output))


# Fichero salidas del adaline y deseadas
with open('outputs/ouputs_adaline_test.csv', 'w') as file:
    file.write("y,d,y-d\n")
    for i, y in enumerate(data_test_output):
        file.write(str(y) + "," +  data_test[i+1][-1] + "," + str(abs(y - float(data_test[i+1][-1]))) + '\n')

# Fichero pesos y umbral del mejor entrenamiento
with open('outputs/weights_bias_adaline.csv', 'w') as file:
    for i in range(len(w)):
        file.write("w" + str(i) + ",")
    file.write("b\n")

    for weight in w:
        file.write(str(weight) + ",")
    file.write(str(b) + '\n')

# Fichero evolución error de entrenamiento y de validación  del mejor entrenamiento
with open('outputs/training_validation_adaline.csv', 'w') as file:
    file.write("training_error,validation_error\n")
    for i, error_tr in enumerate(list_training_errors):
        file.write(str(error_tr) + "," + str(list_validation_errors[i]) + '\n')
