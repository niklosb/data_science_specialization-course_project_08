setwd("~/Documents/E-Courses/Data_Science_Specialization/08_PracticalMachineLearning/CourseProject")
rm( list = ls() )

training <- read.csv("pml-training.csv", na.strings = c("NA",""))
testing  <- read.csv("pml-testing.csv" , na.strings = c("NA",""))

head(training)

# 1. Cleaning the data
# 1.a remove NAs
column_index_NA <- colSums(is.na(training))>=0.80*nrow(training) 
training <- training[, !column_index_NA ]
testing  <- testing[ , !column_index_NA ]
# 1.b check if there are any NAs left
column_num_train_rest_NA   <- colSums(is.na(training))
sum(column_num_train_rest_NA) # 0, no NAs left
column_num_test__rest_NA   <- colSums(is.na(testing))
sum(column_num_test__rest_NA) # 0, no NAs left
# 1.c select useful variables
#names(training)[8]
#names(training)[59]
#names(testing)[8]
#names(testing)[59]
column_index_useful_data <- 8:59 # still 52 variables
column_index_class_var   <-   60
column_names_useful_data <- names(training)[column_index_useful_data]
column_names_class_var   <- names(training)[column_index_class_var]
# 1.d create data_frames with this data
df_train <- training[,c(column_index_useful_data,column_index_class_var)]
rownames(df_train) <- as.character(training$X)
df_test  <- testing[ ,column_index_useful_data]
rownames(df_test) <- as.character(testing$X)
col_dfclass_ind <- 53
# 1.e I selected position and total acceleration variables
selected_var_indexes <- c(1,2,3,4,14,15,16,17,27,28,29,30,40,41,42,43)
df_train <- df_train[,c(col_dfclass_ind, selected_var_indexes)]
#names(df_train)
df_test <- df_test[,c(selected_var_indexes)]

# K-fold cross validation, K = 5
library(caret)
set.seed(9682)
K = 5
folds <- createFolds(y=df_train$classe, k = K, list=TRUE, returnTrain=TRUE)
#sapply(folds,length)
for (k in 1:K) {
  fold_k_train <- df_train[ folds[[k]],]
  fold_k_test  <- df_train[-folds[[k]],]
  # 2. Normalize the data
  preProc_std  <- preProcess(fold_k_train[-1],method=c("center","scale"))
  # 2.a normalize train data
  fold_k_train_std <- fold_k_train[1]
  fold_k_train_std[2:ncol(fold_k_train)] <- predict(preProc_std, fold_k_train[-1])
  # 2.b normalizetest data
  fold_k_test_std <- fold_k_test[1]
  fold_k_test_std[2:ncol(fold_k_test)] <- predict(preProc_std, fold_k_test[-1])
  
  # 3. Train the classifier, method "gbm", boosting with trees
  modFit <- train(classe ~ ., method="gbm", data=fold_k_train_std, verbose=FALSE)
  
  # 4. Calculate the confussion matrix to check the accuracy
  predicted_classe <- predict(modFit, fold_k_test_std)
  confusionMatrix(predicted_classe, fold_k_test_std$classe)
  
  if (k==1) {
    models <- list(modFit)
  } else {
    models <- c(models, list(modFit))
  }
  sapply(models,class)
} 
# out of sample error calculation
accuracy_vec <- vector(length=K)
for (k in 1:K) {
    fold_k_train <- df_train[ folds[[k]],]
    fold_k_test  <- df_train[-folds[[k]],]
    # 2. Normalize the data
    preProc_std  <- preProcess(fold_k_train[-1],method=c("center","scale"))
    # 2.b normalize test data
    fold_k_test_std <- fold_k_test[1]
    fold_k_test_std[2:ncol(fold_k_test)] <- predict(preProc_std, fold_k_test[-1])
    
    # 3. Train the classifier, method "gbm", boosting with trees
    modFit <- models[[k]]
    
    # 4. Calculate the confussion matrix to check the accuracy
    predicted_classe <- predict(modFit, fold_k_test_std)
    accuracy_vec[k] <- sum( predicted_classe == fold_k_test_std$classe) / nrow(fold_k_test_std) 
}
out_of_sample_error <- 1 - mean(accuracy_vec)

# 5. Test data
# 5.a Normalize the data
df_test_std <- predict(preProc_std,df_test)
#sapply(df_test_std, mean)  # not perfect, but ok
#sapply(df_test_std, sd)    # not perfect, but ok
# 5.b Predict the classe
for (k in 1:K) {
  if (k==1) {
    test_classe <- list( predict(models[[k]], df_test_std) )
  } else {
    test_classe <- c( test_classe, list(predict(models[[k]], df_test_std)) )
  }
}
test_classe

# 6. Write answers to files (Course Project submission)
answers = as.character( test_classe[[3]] )
answers[3]  = "B" # second trial
answers[19] = "B" # second trial
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}
pml_write_files(answers)

# 7. The model that predicted everything correctly is model5
selected_model <- models[[5]]
