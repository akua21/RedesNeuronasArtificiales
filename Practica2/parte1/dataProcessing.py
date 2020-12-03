import pandas as pd
import math
import numpy as np

# Normalizar
def normalize(column):
    maxValue = max(column)
    minValue = min(column)

    normalizedColumn = (column - minValue) / (maxValue - minValue)

    return normalizedColumn

# Coger datos del .txt
with open('datos/datosNubes.txt') as file:
    csvReader = pd.read_csv(file, sep='\t')
    header = list(csvReader.columns)

    # Normalizar
    for title in header[:-1]:
        csvReader[title] = normalize(csvReader[title])

    # Ordenar por clase
    csvReader = csvReader.sort_values(header[-1], ascending=False)
    countTotal = csvReader[header[-1]].value_counts()

    class_nube = csvReader.iloc[:countTotal[0]]
    class_multinube = csvReader.iloc[countTotal[0]:(countTotal[0] + countTotal[1])]
    class_cieloDespejado = csvReader.iloc[(countTotal[0] + countTotal[1]):]

    countFold = [math.ceil(i/4) for i in countTotal]

    # Dividir en clases
    division_nube = []
    division_multinube = []
    division_cieloDespejado = []

    for i in range(4):
        division_nube.append(class_nube.iloc[(countFold[0]*i):(countFold[0]*(i+1))])
        division_multinube.append(class_multinube.iloc[(countFold[1]*i):(countFold[1]*(i+1))])
        division_cieloDespejado.append(class_cieloDespejado.iloc[(countFold[2]*i):(countFold[2]*(i+1))])

    # Crear los folds para la VC estratificada
    folds = []

    for i in range(4):
        folds.append(np.concatenate((division_nube[i].values, division_multinube[i].values, division_cieloDespejado[i].values)))

    #Generar conjuntos de train
    for i in range(4):
        with open('datos/train' + str(i+1) + '.csv', 'w') as file:
            all_fold = []
            for j in range(4):
                if i != j:
                    all_fold.append(folds[j])

            dataframe = pd.DataFrame(np.concatenate(all_fold), columns=header)
            dataframe = dataframe.sample(frac=1).reset_index(drop=True)
            file.write(dataframe.to_csv(index=False))


    #Generar conjuntos de test
    for i in range(4):
        with open('datos/test' + str(i+1) + '.csv', 'w') as file:
            dataframe = pd.DataFrame(folds[i], columns=header)
            file.write(dataframe.to_csv(index=False))


    with open('datos/datosNubesProcesados.csv', 'w') as normFile:
        normFile.write(csvReader.to_csv(index=False))
