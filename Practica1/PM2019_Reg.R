library(RSNNS)


## funcion que calcula el error cuadratico medio
MSE <- function(pred,obs) {sum((pred-obs)^2)/length(obs)}

## funcion que calcula el error medio absoluto
MAE <- function(pred, obs) {sum(abs(pred-obs))/length(pred)}

#CARGA DE DATOS
# se supone que los ficheros tienen encabezados
trainSet <- read.csv("inputs/training_set_norm.csv", dec=".", sep=",", header = T)
validSet <- read.csv("inputs/validation_set_norm.csv", dec=".", sep=",", header = T)
testSet  <- read.csv("inputs/test_set_norm.csv", dec=".", sep=",", header = T)

 #trainSet <- read.table("trainParab.dat")
 #validSet <- read.table( "testParab.dat")
 #testSet <- read.table( "testParab.dat")


salida <- ncol (trainSet)   #num de la columna de salida




#SELECCION DE LOS PARAMETROS
topologia        <- c(30, 20) #PARAMETRO DEL TIPO c(A,B,C,...,X) A SIENDO LAS NEURONAS EN LA CAPA OCULTA 1, B LA CAPA 2 ...
razonAprendizaje <- 0.7 #NUMERO REAL ENTRE 0 y 1
ciclosMaximos    <- 10000 #NUMERO ENTERO MAYOR QUE 0

#EJECUCION DEL APRENDIZAJE Y GENERACION DEL MODELO

set.seed(1)
model <- mlp(x= trainSet[,-salida],
             y= trainSet[, salida],
             inputsTest=  validSet[,-salida],
             targetsTest= validSet[, salida],
             size= topologia,
             maxit=ciclosMaximos,
             learnFuncParams=c(razonAprendizaje),
             shufflePatterns = F
             )

#GRAFICO DE LA EVOLUCION DEL ERROR
plotIterativeError(model)

# DATAFRAME CON LOS ERRORES POR CICLo: de entrenamiento y de validacion
iterativeErrors <- data.frame(MAETrain= (model$IterativeFitError/ nrow(trainSet)),
                              MAEValid= (model$IterativeTestError/nrow(validSet)))

#SE OBTIENE EL N?MERO DE CICLOS DONDE EL ERROR DE VALIDACION ES MINIMO
nuevosCiclos <- which.min(model$IterativeTestError)

#ENTRENAMOS LA MISMA RED CON LAS ITERACIONES QUE GENERAN MENOR ERROR DE VALIDACION
set.seed(1)
model <- mlp(x= trainSet[,-salida],
            y= trainSet[, salida],
            inputsTest=  validSet[,-salida],
            targetsTest= validSet[, salida],
            size= topologia,
            maxit=nuevosCiclos,
            learnFuncParams=c(razonAprendizaje),
            shufflePatterns = F
)
# #GRAFICO DE LA EVOLUCION DEL ERROR
plotIterativeError(model)

iterativeErrors <- data.frame(MAETrain= (model$IterativeFitError/ nrow(trainSet)),
                              MAEValid= (model$IterativeTestError/nrow(validSet)))

#CALCULO DE PREDICCIONES
prediccionesTrain <- predict(model, trainSet[,-salida])
prediccionesValid <- predict(model, validSet[,-salida])
prediccionesTest  <- predict(model, testSet[,-salida])

#CALCULO DE LOS ERRORES
errors <- c(TrainMAE= MAE(pred= prediccionesTrain,obs= trainSet[,salida]),
            ValidMAE= MAE(pred= prediccionesValid,obs= validSet[,salida]),
            TestMAE=  MAE(pred= prediccionesTest ,obs=  testSet[,salida]))
errors

nuevosCiclos




#SALIDAS DE LA RED
outputsTrain <- data.frame(pred= prediccionesTrain,obs= trainSet[,salida])
outputsValid <- data.frame(pred= prediccionesValid,obs= validSet[,salida])
outputsTest  <- data.frame(pred= prediccionesTest, obs=  testSet[,salida])

minvalue <- 14999
maxvalue <- 500001
outputsTest <- outputsTest * (maxvalue - minvalue) + minvalue

#GUARDANDO RESULTADOS
saveRDS(model,"outputs_R/nnet.rds") #Red obtenida
write.csv2(errors,"outputs_R/finalErrors.csv")
write.csv2(iterativeErrors,"outputs_R/iterativeErrors.csv")  #Errores por ciclos
write.csv2(outputsTrain,"outputs_R/netOutputsTrain.csv")
write.csv2(outputsValid,"outputs_R/netOutputsValid.csv")
write.csv2(outputsTest, "outputs_R/netOutputsTest.csv") #Salidas deseadas y obtenidas en test (falta desnormalizar)





#############
colnames(trainSet)=c("x","y","z")
head(trainSet)
modelo=lm(z~x+y, trainSet)

summary(modelo)
maelin <- mean(modelo$residuals^2)
maelin   #error mae

#Fuente: https://www.i-ciencias.com/pregunta/89240/como-obtener-el-valor-del-error-cuadratico-medio-de-una-regresion-lineal-en-r
