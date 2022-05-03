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
House.PWin.1 <- House.PWin.Pure

######################## Examine Data ##################
head(House.PWin.1)
colnames(House.PWin.1)
sapply(House.PWin.1, function(x) sum(is.na(x)))

######################## Data Management ##################
House.PWin.1$diff_exp <- House.PWin.1$dexp - House.PWin.1$rexp 

######################## Machine Learning Models ##################
# Split data
set.seed(1993)
trainIndex <- createDataPartition(House.PWin.1$pwin, p=0.7, list = FALSE, times = 1)
#
train <- House.PWin.1[trainIndex,]
test <- House.PWin.1[-trainIndex,]
#
############# Set control parameters for model training ############# 
fitCtrl <- trainControl(method = "repeatedcv",
                        number = 5,
                        repeats = 2,
                        #summaryFunction=twoClassSummary,
                        ## Estimate class probabilities
                        classProbs = TRUE,
                        ## Search "grid" or "random"
                        search = "random",
                        ## Use cluster
                        allowParallel = TRUE)

#tuneGrid=expand.grid(mtry=c(5,10,11))

############# Run the Model ############# 
#ranger
#xgbTree
set.seed(1993)
rf.res <- caret::train(pwin ~ inc + po1 + redist + diff_exp + lagged_dpres + lagged_dem_vote + prcntUnemp + medianIncome + gini + cycle, 
                    data=train, 
                    method='rf', 
                    metric='Accuracy',
                    tuneLength=10,
                    trControl=fitCtrl, 
                    verbose = TRUE)
rf.res

############# Sleep to prevent crashing ############# 
Sys.sleep(10) #Sleep for 10 Seconds

############# Extract predictions ############# 

# Extract predictions
confusionMatrix(predict(rf.res, train, type="raw"), train$pwin)
confusionMatrix(predict(rf.res, test, type="raw"), test$pwin)

pred.train <- predict(rf.res, train, type="prob")[,"Democratic.Win"]

pred.test <- predict(rf.res, test, type="prob")[,"Democratic.Win"]


################## Variable Importance Graphs ###################
#Extract Importance
rfImp <- varImp(rf.res)
plot(rfImp)

#Data Management
rfImp.DF <- rfImp$importance
rfImp.DF$Variables <- rownames(rfImp.DF); rownames(rfImp.DF) <- NULL

rfImp.DF.2 <- rfImp.DF[order(-rfImp.DF$Overall),]
rfImp.DF.3 <- rfImp.DF.2 %>% mutate(Variables=factor(Variables, levels=Variables))  # This trick update the factor levels
rfImp.DF.4 <- head(rfImp.DF.3, 10)
rfImp.DF.4$Variables <- as.character(rfImp.DF.4$Variables)

#Rename Terms
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

rfImp.DF.5 <- rfImp.DF.4 %>% mutate(Variables=factor(Variables, levels=rev(Variables)))  # This trick update the factor levels

#Plot
library(ggplot2)
Plot_VI <- ggplot(rfImp.DF.5, aes(x=Variables, y=Overall)) +
  geom_segment(aes(xend=Variables, yend=0)) +
  geom_point( size=4, color="#1976D2") +
  coord_flip() +
  theme_bw() +
  labs(x = "", 
       y = "Level of Importance",
       title = "Variable Importance for Top 10 Predictors",
       caption = "Note: Figure shows the scaled importance of each variable in contributing to the model's predictive power.\nModel is fit using random forest algorithm.") +
  theme(plot.title = element_text(hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5),
        plot.caption = element_text(hjust = 0.0),
        legend.position="bottom");Plot_VI

################## Accuracy Graph ###################

Accuracy.DF.1 <- data.frame(confusionMatrix(predict(rf.res, test, type="raw"), test$pwin)$table)
Accuracy.DF.1

#Data Management 
Accuracy.DF.1$Correct <- "Incorrect"
Accuracy.DF.1$Correct[Accuracy.DF.1$Prediction == "Democratic.Win" & Accuracy.DF.1$Reference  == "Democratic.Win"] <- "Correct"
Accuracy.DF.1$Correct[Accuracy.DF.1$Prediction == "Independent.Win" & Accuracy.DF.1$Reference  == "Independent.Win"] <- "Correct"
Accuracy.DF.1$Correct[Accuracy.DF.1$Prediction == "Republican.Win" & Accuracy.DF.1$Reference  == "Republican.Win"] <- "Correct"
table(Accuracy.DF.1$Correct)

#Calulate Within Frequency Percentage
Accuracy.DF.2 <- NULL
for(i in unique(Accuracy.DF.1$Reference)){
 Data.Loop.1 <- subset(Accuracy.DF.1, Reference == i)
 Data.Loop.1$Within_Freq<- Data.Loop.1$Freq / sum(Data.Loop.1$Freq)
 
 Accuracy.DF.2 <- rbind(Accuracy.DF.2,Data.Loop.1 )
 
 
}
Accuracy.DF.2$Between_Freq<- Accuracy.DF.2$Freq / sum(Accuracy.DF.2$Freq)

#Subset Away Independent
Accuracy.DF.3 <- subset(Accuracy.DF.2, Reference != "Independent.Win" & Prediction  != "Independent.Win")

#Change Factor Level
Accuracy.DF.3$Reference <- gsub("[.]", " ", Accuracy.DF.3$Reference)

Accuracy.DF.3$Fill <- paste("Actual Result: ", Accuracy.DF.3$Reference, "\nModel Prediction: ", Accuracy.DF.3$Correct, sep = "")
Accuracy.DF.3$Fill <- factor(Accuracy.DF.3$Fill, levels = rev(levels(factor(Accuracy.DF.3$Fill))))

#https://github.com/tidyverse/ggplot2/issues/3612

Plot_1 <- ggplot2::ggplot(Accuracy.DF.3, aes(x = Reference, y =  Within_Freq, fill = Fill, label=paste0(round(Within_Freq,3)*100,"%")," ")) +
  geom_bar(stat = "identity") +
  geom_text(size = 5,
            position = position_stack(vjust = 0.5)) +
  labs(x = "Actual Race Outcome",
       y = "Percent of Party Races",
       title = "Distribution of Correct and Incorrect Classifications of U.S. House Winners",
       subtitle = "Estimates Generated From Out of Sample Predictions",
       caption = "Note: Figure shows out of sample classification of U.S. House winners.\nModel is fit using random forest algorithm.\nPlot omits Independent winning races.") +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
  scale_fill_manual("", values  = rev(c("#1976D2", "#BBDEFB", "#D32F2F", "#FFCDD2"))) +
  theme_minimal() +
  guides(fill=guide_legend(nrow=2,byrow=TRUE)) +
  theme(plot.title = element_text(hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5),
        plot.caption = element_text(hjust = 0.0),
        legend.position="bottom"); Plot_1

############################# Save Plots #########################

ggsave(Plot_1, file = "Confusion Plot - v1 - House Winner.png",
       width=7, height=8,  dpi = 300)

ggsave(Plot_VI, file = "Variable Importance Plot - v1 - House Winner.png",
       width=7, height=6,  dpi = 300)
