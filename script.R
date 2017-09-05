library(caret)
library(rpart)
library(rpart.plot)
library(rattle)
library(randomForest)
library(doParallel)
library(parallel)
library(iterators)
library(foreach)
library(corrplot)



trainingURL <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
testingURL <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
training_raw <- read.csv(url(trainingURL))
testing_raw <- read.csv(url(testingURL))
#split training into training and validation sets
set.seed(582)
inTrain <- createDataPartition(training_raw$classe, p = 0.8, list = FALSE)
training_unclean <- training_raw[inTrain,]
validation_unclean <- training_raw[-inTrain,]

#pre-processing
#remove NAs
colNAstatus <- sapply(training_unclean, function(x) mean(is.na(x)))
plot(colNAstatus) #can clearly see that cols are either complete or nearly incomplete
#all NA cols wil be removed. 
NAcols    <- colNAstatus > 0.95
training_noNAs <- training_unclean[,(NAcols == FALSE)]
validation_noNas <- validation_unclean[,(NAcols == FALSE)]
testing_noNAs <- testing_raw[,(NAcols == FALSE)]

#remove first 5 columns for analysis since they contain id variables
training_noNAs <- training_noNAs[,-(1:5)]
validation_noNas <- validation_noNas[,-(1:5)]
testing_noNAs <- testing_noNAs[,-(1:5)]

#remove nearZeroVar()
nzv <- nearZeroVar(training_noNAs, saveMetrics = TRUE)
training <- training_noNAs[,row.names(nzv[nzv$nzv == FALSE,])] #training set now contains 54 vars including classe
validation <- validation_noNas[,row.names(nzv[nzv$nzv == FALSE,])] #validation set now contains 54 vars including classe
testing <- testing_noNAs[,c(row.names(nzv[nzv$nzv == FALSE,])[1:53],
                            colnames(testing_noNAs)[88])] 
# further reduce variables by PCA
# first check if there are correlated variables
cor_matrix <- cor(training[,-54])
corrplot(cor_matrix, order = "FPC", method = "color", type = "lower", 
         tl.cex = 0.8, tl.col = rgb(0,0,0))
# there are some highly correlated varibles so for ease of computation we will run
# PCA to further reduce variables

prepca <- preProcess(training[,-54], method = "pca", thresh = 0.95)
trainingPCA <- predict(prepca, training)
validationPCA <- predict(prepca, validation)
testingPCA <- predict(prepca, testing)
dim(trainingPCA) # now reduced to 25 variables including classe variable

#make clusters for parallel processing
library(iterators)
library(parallel)
library(foreach)
library(doParallel)
cluster <- makeCluster(detectCores())
resgisterDOParallel(cluster)

#Model1 - linear discriminant analysis using parallel processing
set.seed(582)
intervalStart <- Sys.time()
mod1Control <- trainControl(method = "cv", number = 3, allowParallel = TRUE)
mod_lda <- train(classe ~ ., data = trainingPCA, method = "lda", trControl=mod1Control)
intervalEnd <- Sys.time()
paste("Train model1 took: ",intervalEnd - intervalStart,attr(intervalEnd - intervalStart,"units"))
#took 1.3 seconds on work laptop
#model's in sample accuracy is 65%. No point running a validation.

#Model2 - random forest
set.seed(583)
intervalStart2 <- Sys.time()
mod2Control <- trainControl(method = "cv", number = 3, allowParallel = TRUE)
mod_rf <- train(classe ~ ., data = trainingPCA, method = "rf", trControl=mod2Control)
intervalEnd2 <- Sys.time()
paste("Train model2 took: ",intervalEnd2 - intervalStart2,attr(intervalEnd2 - intervalStart2,"units"))
# took 6.1 mins on work laptop
# models in sample accuracy is 96%. Using this further.  

pred_rf_validation <- predict(mod_rf, validationPCA)
confusionMatrix(pred_rf_validation, validationPCA$classe)
## yayy this has 98% accuracy

pred_rf_test <- predict(mod_rf, testingPCA)
## 20/20 correct!!

saveRDS(mod_rf, "rf_model_assignment.rds")
saveRDS(mod_lda, "lda_model_assignment.rds")
saveRDS()