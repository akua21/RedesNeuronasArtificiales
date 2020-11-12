import pandas as pd

#FUNCIONES AUXILIARES

def normalize(column):

    maxValue = max(column)
    minValue = min(column)

    normalizedColumn = (column - minValue) / (maxValue - minValue)

    return normalizedColumn


with open('datos/datosNubes.txt') as file:
    csvReader = pd.read_csv(file, sep='\t')
    header = list(csvReader.columns)

    for title in header[:-1]:
        csvReader[title] = normalize(csvReader[title])

    with open('datos/datosNubesNormalizado.csv', 'w') as normFile:
        normFile.write(csvReader.to_csv(index=False))
