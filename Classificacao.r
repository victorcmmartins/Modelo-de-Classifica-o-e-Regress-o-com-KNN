# DataSource
  setwd("C:/Users/Victor Martins/Documents/Estudos/Modelo-de-Classifica-o-e-Regress-o-com-KNN/DataSource") 

# Classificação
  train <- read.csv("trainTitanic.csv", header = T) 
  test <- read.csv("testTitanic.csv", header = T) 

## Visualização de categoria de dados
  # Variáveis
  # Survived: 0 = Não, 1 = Sim
  # SibSp: Número de irmãos / cônjuges a bordo
  # Parch: Número de pais / filhos a bordo
  # Fare: Tarifa
  # Embarked: Porto de embarque C = Cherbourg, Q = Queenstown, S = Southampton
  # Pclass: Classe do navio
  # Parametros de K
  str(train)
  str(test)

# Modelagem de Dados  
  train$Survived <- as.factor(train$Survived) 
  train$Embarked <- as.factor(train$Embarked) 
  train$Pclass <- as.factor(train$Pclass) 
  train$Sex <- as.factor(train$Sex) 
  train$X <- NULL 

  test$Survived <- as.factor(test$Survived) 
  test$Embarked <- as.factor(test$Embarked) 
  test$Pclass <- as.factor(test$Pclass) 
  test$Sex <- as.factor(test$Sex) 
  test$X <- NULL 



## Libs
  library(caret)
  library(Amelia)
  library(pROC)

# Definindo seed
set.seed(123)

#Utilizando validação cruzada 10-fold
  ctrl <- trainControl(method = "cv", 
                      number = 10)

  knnFit <- train(Survived ~ . ,
                  method     = "knn",
                  preProcess = c("center","scale"), 
                  tuneLength = 20, 
                  trControl  = ctrl,
                  metric     = "Accuracy", 
                  data       = train)
  knnFit
  plot(knnFit)
  knnFit$finalmodel

## Modelo de Predição
  predknn <- predict(knnFit, test, type = "prob")
  resultknn <- as.factor(ifelse(predknn[,2] > 0.5, 1, 0))

## Desempenho do modelo
  #Matriz de Confusão
  library(caret)
  confusionMatrix(resultknn, test$Survived, positive = "1")

  # Curba ROC e AUC
  library(pROC)
  aucknn <- roc(test$Survived, predknn[,2])
  plot.roc(aucknn, print.thres = T)

  # Utilizando o novo ponto de corte
  resultknn2 <- as.factor(ifelse(predknn[,2] > 0.536, 1, 0))
  confusionMatrix(resultknn2, test$Survived, positive = "1")


## Regressão
rm(list=ls(all=TRUE))

# Banco de dados mtcars 

# crim: taxa de criminalidade per capita por cidade
# nox: concentração de óxidos de nitrogênio (partes por 10 milhões).
# rm: número médio de quartos por habitação.
# rad: índice de acessibilidade às rodovias radiais.
# ...
# medv: valor mediano das casas ocupadas pelo proprietário em $1000.
library(MASS)
data(Boston)

str(Boston)

# Transformação de chas em categoricas
Boston$chas <- as.factor(Boston$chas)

# Verificaçãod de NA
any(is.na(Boston))

# Gerando base treino e teste
set.seed(123)
intrain <- createDataPartition(y = Boston$medv, p = 0.7, list = FALSE)
train <- Boston[intrain,]
test <- Boston[-intrain,]


## Ajustando Parametros
# Escolha de melhor K
set.seed(123)
ctrl <- trainControl(method = "cv", 
                    number = 10)

knnFit <- train(Survived ~ . ,
                method     = "knn",
                preProcess = c("center","scale"), 
                tuneLength = 20, 
                trControl  = ctrl,
                metric     = "RMSE", 
                data       = train)
knnFit
plot(knnFit)

# Predição
knnPredict <- predict(knnFit, newdata = test[, -14])

# Calculo de erro
cbind(knnPredict, test$medv)
plot(knnPredict, test$medv)
RMSE(knnPredict, test$medv)
MAE(knnPredict, test$medv)
