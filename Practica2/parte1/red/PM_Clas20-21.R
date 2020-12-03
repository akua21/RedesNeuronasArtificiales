library(RSNNS)

# Funciones

graficaError <- function(iterativeErrors){
  plot(1:nrow(iterativeErrors),iterativeErrors[,1], type="l", main="Evolución del error",
       ylab="MSE (3 salidas)",xlab="Ciclos",
       ylim=c(min(iterativeErrors),max(iterativeErrors)))
  lines(1:nrow(iterativeErrors),iterativeErrors[,2], col="red")
}


accuracy <- function (cm) sum(diag(cm))/sum(cm)




folds <- c(1, 2, 3, 4)

for (i in folds) {

#CARGA DE LOS DATOS
# cambiar a fold 2 y 3
fold <- i

trainSet <- read.csv(paste("../datos/train",fold,".csv",sep=""),dec=".",sep=",",header = T)
testSet  <- read.csv(paste("../datos/test", fold,".csv",sep=""),dec=".",sep=",",header = T)

#SELECCION DE LA SALIDA. Num de columna del target. 
nTarget <- ncol(trainSet)    

#SEPARAR ENTRADA DE LA SALIDA
trainInput <- trainSet[,-nTarget]
testInput <-  testSet[,-nTarget]


#TRANSFORMAR LA SALIDA DISCRETA A NUMERICA (Matriz con columnas, una por etiqueta, hay un 1 por cada fila en la columna que pertenece a la clase)
trainTarget <- decodeClassLabels(trainSet[,nTarget])
testTarget <-  decodeClassLabels(testSet[,nTarget])

# transformar las entradas de dataframe en matrix para mlp: 
trainInput <- as.matrix(trainInput)
testInput  <- as.matrix(testInput )


#SELECCION DE LOS HIPERPARAMETROS DE LA RED
topologia        <- c(10, 10, 10)
razonAprendizaje <- 0.001
ciclosMaximos    <- 10000

## generar un nombre de fichero que incluya los hiperpar?metros
fileID <- paste("fold_",fold,"_topol",paste(topologia,collapse="-"),"_ra",
                razonAprendizaje,"_iter",ciclosMaximos,sep="")

set.seed(1)
#EJECUCION DEL APRENDIZAJE Y GENERACION DEL MODELO
model <- mlp(x= trainInput,
             y= trainTarget,
             inputsTest= testInput,
             targetsTest= testTarget,
             size= topologia,
             maxit=ciclosMaximos,
             learnFuncParams=c(razonAprendizaje),
             shufflePatterns = F
)

#TABLA CON LOS ERRORES POR CICLO de train y test correspondientes a las 4 salidas
iterativeErrors <- data.frame(MSETrain= (model$IterativeFitError/nrow(trainSet)),
                               MSETest= (model$IterativeTestError/nrow(testSet)))

graficaError(iterativeErrors)


#GENERAR LAS PREDICCIONES en bruto (valores reales)
trainPred <- predict(model,trainInput)
testPred  <- predict(model,testInput)

#poner nombres de columnas "cieloDespejado" "multinube" "nube" 
colnames(testPred)<-colnames(testTarget)
colnames(trainPred)<-colnames(testTarget)

# transforma las tres columnas reales en la clase 1,2,3 segun el maximo de los tres valores. 

trainPredClass<-as.factor(apply(trainPred,1,which.max))  
testPredClass<-as.factor(apply(testPred,1,which.max)) 

#transforma las etiquetas "1", "2", "3" en "cieloDespejado" "multinube" "nube"
levels(testPredClass)<-c("cieloDespejado","multinube", "nube")
levels(trainPredClass)<-c("cieloDespejado", "multinube", "nube")


#CALCULO DE LAS MATRICES DE CONFUSION
trainCm <- confusionMatrix(trainTarget,trainPred)
testCm  <- confusionMatrix(testTarget, testPred)

trainCm
testCm

#VECTOR DE PRECISIONES
accuracies <- c(TrainAccuracy= accuracy(trainCm), TestAccuracy=  accuracy(testCm))

accuracies


# calcular errores finales MSE
#MSEtrain <-sum((trainTarget - trainPred)^2)/nrow(trainSet)
#MSEtest <-sum((testTarget - testPred)^2)/nrow(testSet)




#GUARDANDO RESULTADOS
#MODELO
saveRDS(model,            paste("files/nnet_",fileID,".rds",sep=""))

#tasa de aciertos (accuracy)
write.csv(accuracies,     paste("files/finalAccuracies_",fileID,".csv",sep=""))

#Evoluci?n de los errores MSE
write.csv(iterativeErrors,paste("files/iterativeErrors_",fileID,".csv",sep=""))

#salidas esperadas de test con la clase (Target) (?ltima columna del fichero de test)
write.csv( testSet[,nTarget] ,      paste("files/TestTarget_",fileID,".csv",sep=""), row.names = TRUE)

#errores finales de entrenemiento y test
write.csv( c(tail(iterativeErrors, n=1), nuevosCiclos) ,      paste("files/FinalErrors_",fileID,".csv",sep=""), row.names = TRUE)


#salidas esperadas de test codificadas en tres columnas (Target)
write.csv(testTarget ,      paste("files/TestTargetCod_",fileID,".csv",sep=""), row.names = TRUE)


#salidas de test en bruto (nums reales)
write.csv(testPred ,      paste("files/TestRawOutputs_",fileID,".csv",sep=""), row.names = TRUE)

#salidas de test con la clase
write.csv(testPredClass,  paste("files/TestClassOutputs_",fileID,".csv",sep=""),row.names = TRUE)

# matrices de confusi?n
write.csv(trainCm,        paste("files/trainCm_",fileID,".csv",sep=""))
write.csv(testCm,         paste("files/testCm_",fileID,".csv",sep=""))

}

