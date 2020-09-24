import csv
import numpy as np
from numpy import random

header = ["longitude","latitude","housing_median_age","total_rooms","total_bedrooms","population","households","median_income","median_house_value"]

w = random.rand(len(header)-1) * 2 - 1
b = random.rand() * 2 - 1
learning_rate = 0.5

epoch = 1

MAX_EPOCH = 50

list_test_errors = []
list_validation_errors = []

# Cargar los datos
with open('training_set.csv') as file:
    data_training = list(csv.reader(file, delimiter=',', skipinitialspace=True))

with open('test_set.csv') as file:
    data_test = list(csv.reader(file, delimiter=',', skipinitialspace=True))

with open('validation_set.csv') as file:
    data_validation = list(csv.reader(file, delimiter=',', skipinitialspace=True))

# Las entradas son las 9 columnas, por lo que tiene que haber 9 pesos y el umbral
# La salida es los pesos por las entradas, más el umbral

def getOutput(row):
    x = np.array(list(map(lambda x: float(x), row[:-1])))

    return sum(x * w) + b

def getWeights(row, y):
    x = np.array(list(map(lambda x: float(x), row[:-1])))
    d = float(row[-1])

    return learning_rate * (d - y) * x

# Voy a usar el error absoluto medio
def getError(dataframe):
    total = 0
    count = 0
    for row in dataframe[1:]:
        d = float(row[-1])
        y = getOutput(row)

        total += abs(d - y)
        count += 1

    return total / (len(dataframe[1:]))

while epoch <= MAX_EPOCH:

    for data in data_training[1:]:
        y = getOutput(data)
        w = getWeights(data, y)

    list_test_errors.append(getError(data_test))
    list_validation_errors.append(getError(data_validation))

    print("Epoch num: ", epoch)
    epoch += 1

print(list_test_errors)
print(list_validation_errors)
