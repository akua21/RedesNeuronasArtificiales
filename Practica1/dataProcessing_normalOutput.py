import pandas as pd

#FUNCIONES AUXILIARES

def normalize(column):

    maxValue = max(column)
    minValue = min(column)

    normalizedColumn = (column - minValue) / (maxValue - minValue)

    return normalizedColumn

def randomize(dataframe):
    return dataframe.sample(frac = 1).reset_index(drop=True)



header = ["longitude","latitude","housing_median_age","total_rooms","total_bedrooms","population","households","median_income","median_house_value"]

with open('raw/california_housing.csv') as file:
    csvReader = pd.read_csv(file, sep=',', skipinitialspace=True, usecols=header)

    print("Min value: ", min(csvReader["median_house_value"]))
    print("Max value: ", max(csvReader["median_house_value"]))

    for title in header:
        csvReader[title] = normalize(csvReader[title])

    csvReader = randomize(csvReader)

    trainingSize = int(0.6 * len(csvReader))
    validationSize = int(0.2 * len(csvReader))
    testSize = int(0.2 * len(csvReader))

    trainingSet = csvReader.iloc[:trainingSize]
    validationSet = csvReader.iloc[trainingSize:trainingSize+validationSize].reset_index(drop=True)
    testSet = csvReader.iloc[trainingSize+validationSize:trainingSize+validationSize+testSize].reset_index(drop=True)


    with open('inputs/training_set_norm.csv', 'w') as trainingFile:
        trainingFile.write(trainingSet.to_csv(index=False))

    with open('inputs/validation_set_norm.csv', 'w') as validationFile:
        validationFile.write(validationSet.to_csv(index=False))

    with open('inputs/test_set_norm.csv', 'w') as testFile:
        testFile.write(testSet.to_csv(index=False))
