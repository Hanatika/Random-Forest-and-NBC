#import library
library(dplyr)
library(SnowballC)
library(tidytext)
library(tm)
library(caret)
library(e1071)
library(naivebayes)
library(randomForest)

#import data
df <- read.csv("E:/SPAM text message 20170820 - Data.csv")
df$Doc <- seq(1:nrow(df))
head(df)
df_copy <- df
df_copy$Category <- as.factor(df_copy$Category)

#preprocessing
#Merubah text ke lower case
df_copy$Message <- tolower(df_copy$Message)

#menghapus tanda baca
df_copy$Message <- gsub(pattern = "[[:punct:]]", replacement="", df_copy$Message)

#menghapus angka
#regex -> regular expression
df_copy$Message <- gsub(pattern= "[[:digit:]]", replacement = "", df_copy$Message)

#menghapus spesial karakter
df_copy$Message <- gsub(pattern= "[^[:alnum:]]", replacement = " ", df_copy$Message)

#tokenisasi
df_token <- df_copy
df_token$Doc <- seq(1:nrow(df_token))
df_token <- as_tibble(df_token)
df_tfidf <- df_token %>%
  unnest_tokens(output=word, Message, token = "words") %>%
  mutate(stem = wordStem(word)) %>%
  anti_join(stop_words) %>% 
  count(Doc, word) %>%
  bind_tf_idf(word, Doc, n)
#############################
df_dtm <- df_tfidf %>%
  cast_dtm(Doc, word, tf_idf)
df_dtm2 <- data.frame(as.matrix(df_dtm))
df_dtm2$Doc <- as.numeric(row.names(df_dtm2))

#Merge
df_merge <- (merge.data.frame(x=df_dtm2, y=df, by.y = "Doc"))
df_merge$Category <- as.factor(df_merge$Category)
df_merge <- df_merge[,-c(1.26)]

#pemodelan
#membagi data ke dalam training dan testing
#80:20, 60:40, 70:30, 90:10
train_index <- createDataPartition(df_merge$Category, p=0.8, list=F)
train <- df_merge[train_index,]
test <- df_merge[-train_index,]
table(test$Category)

#melatih model
#naif bayes
model_nbc <- naive_bayes(Category~., data=train)

#random forest
model_rf <- randomForest(Category~., data=train)

#evaluasi model training
fitted_values_nbc <- predict(model_nbc, train)
fitted_values_rf <- model_rf$predicted

confusionMatrix(fitted_values_nbc, train$Category) #model NBC
confusionMatrix(fitted_values_rf, train$Category) #model rf

#evaluasi model testing
pred_nbc <- predict(model_nbc, test, type="class")
pred_rf <- predict(model_rf, test)

confusionMatrix(pred_nbc, test$Category) #model NBC
confusionMatrix(pred_rf, test$Category) #model rf