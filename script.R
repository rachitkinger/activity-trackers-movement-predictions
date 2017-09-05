library(caret)
library(rpart)
library(rpart.plot)
library(rattle)
library(randomForest)


trainingURL <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
testingURL <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
training_raw <- read.csv(url(trainingURL))
testing_raw <- read.csv(url(testingURL))
#split training into training and validation sets

inTrain <- createDataPartition(training_raw$classe, p = 0.8, list = FALSE)
training_unclean <- training_raw[inTrain,]
validation_unclean <- training_raw[-inTrain,]

#pre-processing
#remove NAs
colNAstatus <- sapply(training_unclean, function(x) mean(is.na(x)))
plot(colNAstatus) #can clearly see that cols are either complete or nearly incomplete
#all NA cols wil be removed. 
NAcols    <- colNAstatus > 0.95
training_noNAs <- training_unclean[,-NAcols]
validation_noNas <- validation_unclean[,-NAcols]


#remove nearZeroVar()
#PCA - try PCA with each of the models and also w/o PCA in each of the models
## to do this first break out your testing data into testing & staging data
