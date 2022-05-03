#Sharif Amlani
#R 4.1.1
#Spring 2022

######################## Code Summary ##################

########################## Prelude #####################

rm(list=ls(all=TRUE))
options(stringsAsFactors = FALSE)
options(scipen = 3)
set.seed(1993)

######################### Functions ###################

######################### Library #####################
library(mlbench)
library(parallel)
library(doParallel)
library(foreach)
library(haven)
library(MASS)
library(ggplot2)
library(caret)
library(ranger)
library(pROC)
library(party)
library(dplyr)
library(ggraph)
library(igraph)
library(rpart.plot)

cl <- makeCluster((detectCores() - 2), setup_timeout = 0.5) # convention to leave 1 core for OS
registerDoParallel(cl)
######################## Upload Data ##################

#Set Working Directory
setwd("C:/Users/Shari/OneDrive/University of California, Davis/Research Projects/NYT - Interview/Data/Complete Data/District Level")


#Upload Data
load(file = "District Level - Complete Data - w1.RData")
House.Vote.Share.1 <- House.Vote.Share.Pure
House.dem_vote.1 <- House.dem_vote.Pure

######################## Examine Data ##################
head(House.Vote.Share.1)
colnames(House.Vote.Share.1)
sapply(House.Vote.Share.1, function(x) sum(is.na(x)))

######################## Machine Learning Models ##################
# Split data
set.seed(1993)
trainIndex <- createDataPartition(House.Vote.Share.1$dem_vote, p=0.7, list = FALSE, times = 1)
#
train <- House.Vote.Share.1[trainIndex,]
test <- House.Vote.Share.1[-trainIndex,]
#
############# Set control parameters for model training ############# 
fitCtrl <- trainControl(method = "repeatedcv",
                        number = 5,
                        repeats = 2,
                        #summaryFunction=twoClassSummary,
                        ## Estimate class probabilities
                        #classProbs = TRUE,
                        ## Search "grid" or "random"
                        search = "random",
                        ## Use cluster
                        allowParallel = TRUE)

############# Run the Model ############# 
rf.res <- caret::train(dem_vote ~ inc + po1 + redist + dexp + rexp + lagged_dpres + lagged_dem_vote + prcntUnemp + medianIncome + gini + cycle, 
                       data=train, 
                       method='rf', 
                       metric='RMSE', 
                      # tuneLength=10,
                       trControl=fitCtrl,
                       verbose = TRUE)
rf.res

############# Sleep to prevent crashing ############# 
Sys.sleep(10) #Sleep for 10 Seconds
############# Model Fit ############# 
train$y_hat <- predict(rf.res, train, type="raw")
test$y_hat <- predict(rf.res, test, type="raw")

############# Examine Out-of-Sample Performance ############# 
#Examine Fit Using Linear Model
test_results<- lm(dem_vote ~y_hat, data = test)
summary(test_results)

#Correlation
A_P_Cor <- cor(test$y_hat, test$dem_vote, use = "complete")

#Key Model Stats
mtry <- rf.res$bestTune
RMSE <- paste(round(subset(data.frame(rf.res$results), mtry == rf.res$bestTune[[1]])$RMSE,3) *100, "%", sep = "")
Rsquared <- round(subset(data.frame(rf.res$results), mtry == rf.res$bestTune[[1]])$Rsquared, 2)


Plot_1 <- ggplot(test, aes (x = y_hat, y = dem_vote)) +
  geom_point() +
  stat_smooth(method = "lm") +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed")+
  theme_minimal() +
  labs(x = "Predicted Democratic Vote Share",
       y = "Actual Democratic Vote Share",
       title = "Actual Verses Predicted Two-Party Democratic Vote Share",
       subtitle = "Estimates Generated From Out of Sample Predictions",
       caption = paste('Scatterplot reports out of sample comparison between actual and predicted two-party vote share.\nModel is fit using random forest algorithm.\nPlots fitted with a linear line with 95% confidence intervals.\nDotted line is fit to represent 1:1 relationship.\nCorrelation between actual and predicted values is', round(A_P_Cor, 2), "\nModel statistics are as follows:\nNumber of predictors sampled at each splits = ", mtry, "| Root Mean Squared Error = ", RMSE, " | R Squared = ", Rsquared, sep = " ")) +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
  scale_x_continuous(labels = scales::percent_format(accuracy = 1)) +
  theme(plot.title = element_text(hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5),
        plot.caption = element_text(hjust = 0.0),
        legend.position="bottom"); Plot1

  
############# Variable importance Graphs ############# 
# Variable importance
rfImp <- varImp(rf.res)
plot(rfImp)

rfImp.DF <- rfImp$importance
rfImp.DF$Variables <- rownames(rfImp.DF); rownames(rfImp.DF) <- NULL

rfImp.DF.2 <- rfImp.DF[order(-rfImp.DF$Overall),]
rfImp.DF.3 <- rfImp.DF.2 %>% mutate(Variables=factor(Variables, levels=Variables))  # This trick update the factor levels
rfImp.DF.4 <- head(rfImp.DF.3, 10)
rfImp.DF.4$Variables <- as.character(rfImp.DF.4$Variables)

rfImp.DF.4$Variables[rfImp.DF.4$Variables == "lagged_dem_vote"] <- "Lagged Dem Vote"
rfImp.DF.4$Variables[rfImp.DF.4$Variables == "lagged_dpres"] <- "Lagged Dem Pres Vote"
rfImp.DF.4$Variables[rfImp.DF.4$Variables == "incDemocratic.Incumbent"] <- "Democratic Incumbent"
rfImp.DF.4$Variables[rfImp.DF.4$Variables == "incGOP.Incumbent"] <- "Republican Incumbent"
rfImp.DF.4$Variables[rfImp.DF.4$Variables == "rexp"] <- "Republican Expenditures"
rfImp.DF.4$Variables[rfImp.DF.4$Variables == "dexp"] <- "Democratic Expenditures"
rfImp.DF.4$Variables[rfImp.DF.4$Variables == "diff_exp"] <- "Party Expenditure Differential"
rfImp.DF.4$Variables[rfImp.DF.4$Variables == "gini"] <- "Gini Coefficient"
rfImp.DF.4$Variables[rfImp.DF.4$Variables == "prcntUnemp"] <- "% Unemployment"
rfImp.DF.4$Variables[rfImp.DF.4$Variables == "medianIncome"] <- "Median Income"
rfImp.DF.4$Variables[rfImp.DF.4$Variables == "incDemocratic.Open.Seat"] <- "Democratic Open Seat"
rfImp.DF.4$Variables[rfImp.DF.4$Variables == "incGOP.Open.Seat"] <- "Republican Open Seat"
rfImp.DF.4$Variables[rfImp.DF.4$Variables == "po1Quality.Challenger"] <- "Quality Challenger"

rfImp.DF.5 <- rfImp.DF.4 %>% mutate(Variables=factor(Variables, levels=rev(Variables)))  # This trick update the factor levels

library(ggplot2)
Plot_VI <- ggplot(rfImp.DF.5, aes(x=Variables, y=Overall)) +
  geom_segment(aes(xend=Variables, yend=0)) +
  geom_point( size=4, color="#1976D2") +
  coord_flip() +
  theme_bw() +
  labs(x = "", 
       y = "Scaled Level of Importance",
       title = "Scaled Variable Importance for Top 10 Predictors",
       caption = "Note: Figure shows the scaled importance of each variable in contributing to the model's predictive power.\nModel is fit using random forest algorithm.") +
  theme(plot.title = element_text(hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5),
        plot.caption = element_text(hjust = 0.0),
        legend.position="bottom");Plot_VI


############################# Save Plots #########################
ggsave(Plot_1, file = "Actual v Predicted - v1 - House Vote Share.png",
       width=7, height=6,  dpi = 300)

ggsave(Plot_VI, file = "Variable Importance Plot - v1 - House Vote Share.png",
       width=8, height=6,  dpi = 300)
