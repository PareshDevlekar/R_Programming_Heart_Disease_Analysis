#Loading Libraries
library(tidyverse)

#Reading the data set
data <- read.csv("C:/Users/Paresh Devlekar/Desktop/Paresh/R_Programming/R_Project/heart.csv")
head(data)
View(data)
tail(data)
str(data)
ncol(data)
nrow(data)
colnames(data)
summary(data)

#Data Transformation
data2 <- data %>%
  mutate(sex = if_else(sex == 1, "MALE", "FEMALE"),
         fbs = if_else(fbs == 1, ">120", "<=120"),
         exang = if_else(exang == 1, "YES" ,"NO"),
         cp = if_else(cp == 1, "ATYPICAL ANGINA",
                      if_else(cp == 2, "NON-ANGINAL PAIN", "ASYMPTOMATIC")),
         restecg = if_else(restecg == 0, "NORMAL",
                           if_else(restecg == 1, "ABNORMALITY", "PROBABLE OR DEFINITE")),
         slope = as.factor(slope),
         ca = as.factor(ca),
         thal = as.factor(thal),
         target = if_else(target == 1, "YES", "NO")
  ) %>% 
  mutate_if(is.character, as.factor) %>% 
  dplyr::select(target, sex, fbs, exang, cp, restecg, slope, ca, thal, everything())

View(data2)

#Data Visualization

# Bar Plot for target (heart disease)
ggplot(data=data2, aes(x=target, fill=target)) +
  geom_bar() +
  xlab("Heart Disease") +
  ylab("Count") + 
  ggtitle("Presence and Absence of Heart Disease") +
  theme(plot.title = element_text(hjust = 0.5)) +
  scale_fill_discrete(name="Heart Disease", labels=c("Absence","Presence"))
  
# prop.table(table(data2$target))

# Bar Plot analyzing frequency of heart disease w.r.t age
ggplot(data=data2, aes(x=ï..age)) + geom_bar(fill="palegreen4") +
  xlab("Age") +
  ylab("Count") +
  ggtitle("Risk of Heart Disease w.r.t Age") +
  theme(plot.title = element_text(hjust = 0.5))

#Compare blood pressure(continuous variable) across chest pain(categorical variable)
# Box-plot - Used for comparing one continuous and one categorical variable
ggplot(data=data2, aes(x=sex, y=trestbps, fill=cp)) + 
  geom_boxplot() +
  facet_grid(~cp) +
  xlab("Sex") +
  ylab("Blood Pressure") +
  ggtitle("Blood Pressure vs different chest pains") +
  theme(plot.title = element_text(hjust = 0.5)) +
  scale_fill_discrete(name="Chest Pain")

#Compare serum cholesterol(continuous variable) across chest pain(categorical variable)
ggplot(data=data2, aes(x=sex, y=chol, fill=cp)) +
  geom_boxplot() +
  facet_grid(~cp) +
  xlab("Sex") +
  ylab("Serum Cholesterol(mg/dl)") +
  ggtitle("Different chest pains w.r.t measure of serum cholesterol") + 
  theme(plot.title = element_text(hjust = 0.5)) +
  scale_fill_discrete(name="Chest Pain")

#Correlation
library(corrplot)
library(ggplot2)
# If the relationship is strong, it will be closer to positive 1. 
# If the relationship is inverse, (i.e. one variable increases resulting in decrease of another variable), that is a negative relationship.
# Anything closer to 0 implies that it is not related at all.
cor_heart <- cor(data2[, 10:14])
cor_heart
corrplot(cor_heart, method ='circle')
#-----------------------ANALYSIS AND VISUALIZATION END--------------------------
#Model Building
s = sum(is.na(data2))
#Splitting the data into training and testing set
library(caret)
library(e1071)
set.seed(10)

inTrainRows <- createDataPartition(data2$target,p=0.7,list=FALSE)
trainData <- data2[inTrainRows,]
testData <-  data2[-inTrainRows,]
nrow(trainData)/(nrow(testData)+nrow(trainData)) #check the percentage

AUC = list()
Accuracy = list()

#Logistic Regression Model
set.seed(10)
logRegModel <- train(target ~ ., data=trainData, method = 'glm', family = 'binomial')
logRegPrediction <- predict(logRegModel, testData)
logRegPredictionprob <- predict(logRegModel, testData, type='prob')[2]
logRegConfMat <- confusionMatrix(logRegPrediction, testData[,"target"])
#ROC Curve
library(pROC)
AUC$logReg <- roc(as.numeric(testData$target),as.numeric(as.matrix((logRegPredictionprob))))$auc
Accuracy$logReg <- logRegConfMat$overall['Accuracy']


#Support Vector Machine
fitControl <- trainControl(method = "repeatedcv",
                           number = 10,
                           repeats = 10,
                           ## Estimate class probabilities
                           classProbs = TRUE,
                           ## Evaluate performance using 
                           ## the following function
                           summaryFunction = twoClassSummary)

set.seed(10)
svmModel <- train(target ~ ., data = trainData,
                  method = "svmRadial",
                  trControl = fitControl,
                  preProcess = c("center", "scale"),
                  tuneLength = 8,
                  metric = "ROC")
svmPrediction <- predict(svmModel, testData)
svmPredictionprob <- predict(svmModel, testData, type='prob')[2]
svmConfMat <- confusionMatrix(svmPrediction, testData[,"target"])
#ROC Curve
AUC$svm <- roc(as.numeric(testData$target),as.numeric(as.matrix((svmPredictionprob))))$auc
Accuracy$svm <- svmConfMat$overall['Accuracy']


#Random Forest
library(randomForest)
set.seed(10)
RFModel <- randomForest(target ~ .,
                        data=trainData, 
                        importance=TRUE, 
                        ntree=200)

RFPrediction <- predict(RFModel, testData)
RFPredictionprob = predict(RFModel,testData,type="prob")[, 2]

RFConfMat <- confusionMatrix(RFPrediction, testData[,"target"])

AUC$RF <- roc(as.numeric(testData$target),as.numeric(as.matrix((RFPredictionprob))))$auc
Accuracy$RF <- RFConfMat$overall['Accuracy']



#Comparison of AUC and Accuracy between models
row.names <- names(Accuracy)
col.names <- c("AUC", "Accuracy")
cbind(as.data.frame(matrix(c(AUC,Accuracy),nrow = 3, ncol = 2,
                           dimnames = list(row.names, col.names))))


#Confusion Matrix
logRegConfMat
RFConfMat
svmConfMat

#A comparison of the area under the ROC and the accuracy of the model predictions shows that logistic regression performs best (accuracy of 0.87). 
#Tree-based methods shows low accuracy.
# Higher the AUC better the model is at seperating positives and negatives
