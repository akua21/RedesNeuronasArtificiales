# ADALINE
import numpy as np
import pandas as pd
import random

LEARNING_RATE = 0.2

# Inicializar pesos y umbral (aleatorios entre -1 y 1)
def initializeWandT(num_w):
    threshold = random.uniform(-1, 1)

    weights = []
    num_weights = num_w - 1
    for i in range(num_weights):
        weights.append(random.uniform(-1, 1))

    return threshold, weights


# Entrada entrenamiento
with open('datosArtificiales_training.csv') as file:
    csvReader = pd.read_csv(file, sep=',', skipinitialspace=True)

    num_cols = csvReader.shape[1]
    threshold, weights = initializeWandT(num_cols)

    examples = [list(row) for row in csvReader.values]


# Bucle ciclos
error_v_init = 100
error_v = 0

while error_v != error_v_init: # Criterio de parada
    error_v_init = error_v

    # Bucle patrones
    for instance in examples:
        # salida estimada
        y = 0
        for count, col in enumerate(instance):
            if count < num_cols-1:
                y += weights[count] * col
        y += threshold

        # salida deseada
        d = instance[num_cols-1]

        # Se modifican los pesos y el umbral
        for count, w in enumerate(weights):
            w += LEARNING_RATE * (d - y) * instance[count]
            weights[count] = w

        threshold += LEARNING_RATE * (d - y)

    # Evaluar ERROR entrenamiento
    error_t = 0
    for instance in examples:
        # salida estimada
        y = 0
        for count, col in enumerate(instance):
            if count < num_cols-1:
                y += weights[count] * col
        y += threshold

        # salida deseada
        d = instance[num_cols-1]

        # error
        error_t += abs(d - y)

    error_t = error_t / csvReader.shape[0]

    # print("error_t ", error_t)

    # Evaluar ERROR validación
    with open('datosArtificiales_validation.csv') as file:
        csvReader2 = pd.read_csv(file, sep=',', skipinitialspace=True)

        examples2 = [list(row) for row in csvReader2.values]


    for instance in examples2:
        # salida estimada
        y = 0
        for count, col in enumerate(instance):
            if count < num_cols-1:
                y += weights[count] * col
        y += threshold

        # salida deseada
        d = instance[num_cols-1]

        # error
        error_v += abs(d - y)

    error_v = error_v / csvReader.shape[0]

    print("error_v ", error_v)
