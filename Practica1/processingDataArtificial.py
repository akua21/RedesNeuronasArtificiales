import pandas as pd

def normalize(column):

    maxValue = max(column)
    minValue = min(column)

    normalizedColumn = (column - minValue) / (maxValue - minValue)

    return normalizedColumn

def randomize(dataframe):
    return dataframe.sample(frac = 1).reset_index(drop=True)


header = ["x.1", "x.2", "y"]

with open('datosArtificiales.csv') as file:
    csvReader = pd.read_csv(file, sep=',', skipinitialspace=True, usecols=header)

    for title in header[:-1]:
        csvReader[title] = normalize(csvReader[title])

    csvReader = randomize(csvReader)

    trainingSize = int(0.7 * len(csvReader))
    validationSize = int(0.3 * len(csvReader))

    trainingSet = csvReader.iloc[:trainingSize]
    validationSet = csvReader.iloc[trainingSize:trainingSize+validationSize].reset_index(drop=True)

    with open('datosArtificiales_training.csv', 'w') as trainingFile:
        trainingFile.write(trainingSet.to_csv(index=False))

    with open('datosArtificiales_validation.csv', 'w') as validationFile:
        validationFile.write(validationSet.to_csv(index=False))
