library(caret)
library(rpart)
library(rpart.plot)
library(rattle)
library(randomForest)
library(corrplot)

training <- read.csv("course-practical-machine-learning/assignment/pml-training.csv")
# 5 activity classifications
levels(training$classe)
