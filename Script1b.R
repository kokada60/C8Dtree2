library(parallel)
library(parallelMap)
library(tidyverse)
library(dplyr)
library(mlr)

# First filter out ozone records with NAs for target field. 
data(Ozone, package="mlbench")
ozoneTib <- as.tibble(Ozone)
names(ozoneTib) <- c("Month", "Date", "Day", "Ozone", "Press_height", "Wind", "Humid", "Temp_Sand", "temp_Monte", 
                     "Inv_height", "Press_grad", "Inv_temp", "Visib")
glimpse(ozoneTib)
summary(ozoneTib)

# In order to further cleaning the list, filter out records with NA for Ozone, the target variable, and also 
#   to numericize factorial fields Month, Date, and Day. 
map_dbl(ozoneTib, ~sum(is.na(.)))
check_isAnyValIsFactorial <- function(c) {
  any(is.factor(c))
}
check_isAnyValIsNA <- function(c) {
  any(is.na(c))
}
#ozoneClean <- mutate_all(data.frame(ozoneTib), as.numeric) %>% filter(is.na(Ozone)==FALSE) %>% as.tibble()
ozoneClean <- mutate_if(ozoneTib, check_isAnyValIsFactorial, as.numeric) %>% filter(is.na(Ozone)==FALSE) %>% as.tibble()
check_isAnyValIsFactorial(ozoneTib[[2]])
map_dbl(ozoneClean, ~sum(is.na(.)))
glimpse(ozoneClean)

# all rows with an NA in any field except for Ozone target field. 
ozoneTib[!is.na(ozoneTib$Ozone) & !complete.cases(ozoneTib), ]  

# Tidy up the dataframe to chart for each independent variable to illustrate its relation to the target variable. 
gather(ozoneClean, key="Variable", value="Value", -Ozone) %>%
  ggplot(aes(x=Value, y=Ozone)) + 
  facet_wrap(~Variable, scale="free_x") + 
  geom_point() + 
  geom_smooth() +
  geom_smooth(method="lm", col="red") + 
  theme_bw()


imputeMethod <- imputeLearner("regr.rpart") 
ozoneImp <- impute(as.data.frame(ozoneClean), #classes=list(numeric=imputeMethod))
                   cols=list(Press_height=imputeMethod, Humid=imputeMethod, Temp_Sand=imputeMethod, temp_Monte=imputeMethod, 
                             Inv_height=imputeMethod, Press_grad=imputeMethod, Inv_temp=imputeMethod)) 

# Set records with NAs as a separate dataframe.
ozoneIndex_NAs <- which(!complete.cases(ozoneClean))
ozoneNAs <- ozoneClean[ozoneIndex_NAs, ]

head(ozoneNAs)
head(ozoneImp$data[ozoneIndex_NAs, ])

  # Just a few of points here and there. But overall regression shape does not change all that much for each variable.
gather(ozoneImp$data, key="Variable", value="Value", -Ozone) %>%
  ggplot(aes(x=Value, y=Ozone)) + 
  facet_wrap(~Variable, scale="free_x") + 
  geom_point() + 
  geom_smooth() +
  geom_smooth(method="lm", col="red") + 
  theme_bw()

which(ozoneTib) 

ozoneTask <- makeRegrTask(data=ozoneImp$data, target="Ozone")
lin <- makeLearner("regr.lm")

listFilterMethods() %>% View()
filterVals <-generateFilterValuesData(task=ozoneTask, method="linear.correlation") 
filterValsRFSRC <- generateFilterValuesData(task=ozoneTask, method="randomForestSRC_importance")
filterValsRF <- generateFilterValuesData(task=ozoneTask, method="randomForest_importance")
filterValsChiSq <- generateFilterValuesData(task=ozoneTask, method="FSelector_chi.squared")
filterVals
filterValsRF
filterValsRFSRC
filterValsChiSq 
plotFilterValues(filterVals) + theme_bw()
plotFilterValues(filterValsRF) + theme_bw()
plotFilterValues(filterValsRFSRC) + theme_bw()
plotFilterValues(filterValsChiSq) + theme_bw()

# Still continuing with Filter method of Feature Selection... Simply wrapping the filter method with a learner.
filterWrapper = makeFilterWrapper(learner = lin, fw.method = "linear.correlation")
# Within the wrapper just created of learner and filter feature selector, each hyperparams of the steps become available for tuning...
# In this case the abs parameter of filter feature selector will be  tuned. Should the learner select only the Top 1 rank or all 12 features? 
# The first step is to create Parameter Space. 
lmParamSpace <- makeParamSet(
  makeIntegerParam("fw.abs", lower=1, upper = 12)    #Should the learner select only the Top 1 rank or all 12 features?
)
gridSearch <- makeTuneControlGrid()  # This grid control will go through all values within specified param value space and evaluate rank for each. 
kFold <- makeResampleDesc(method="CV", iters=10)
tunedFeatures <- tuneParams(filterWrapper, task=ozoneTask, resampling=kFold, par.set=lmParamSpace, control=gridSearch)
tunedFeatures # Selects 10 as the value of fw.abs, ie TOP 10 ranked features will be selected. 

# Let's try the wrapper of learner and filter feature selection with different filter methods 
filterWrapper_RF <- makeFilterWrapper(learner=lin, fw.method="randomForest_importance")
lmParamSpace <- makeParamSet(
  makeIntegerParam("fw.abs", lower=1, upper=12) 
)
gridSearch <- makeTuneControlGrid()
kFold <- makeResampleDesc(method="CV", iters=15)
tunedFeatures_RF <- tuneParams(learner=filterWrapper_RF, task=ozoneTask, resampling=kFold, par.set=lmParamSpace, control=gridSearch) 
tunedFeatures_RF

filterWrapper_RFSRC <- makeFilterWrapper(learner=lin, fw.method="randomForestSRC_importance")
lmParamSpace <- makeParamSet(
  makeIntegerParam(id="fw.abs", lower=1, upper=12) 
)
gridSearch <- makeTuneControlGrid()
kFold <- makeResampleDesc(method="CV", iters=10)
tunedFeatures_RFSRC <- tuneParams(learner=filterWrapper_RFSRC, task=ozoneTask, resampling=kFold, par.set=lmParamSpace, control=gridSearch)
tunedFeatures_RFSRC

filterWrapper_ChiSq <- makeFilterWrapper(learner=lin, fw.method="FSelector_chi.squared")
lmParamSet <- makeParamSet(
  makeIntegerParam(id="fw.abs", lower=1, upper=12)
)
gridSearch <- makeTuneControlGrid()
kFold <- makeResampleDesc(method="CV", iters=10)
tunedFeatures_ChiSq <- tuneParams(learner=filterWrapper_ChiSq, task=ozoneTask, resampling=kFold, par.set=lmParamSet, control=gridSearch)

# We'll pick the linear correlation tune methods for now, though it appears randomFresetSRC is the best performing by a slim margin, 
# with fw.abs of 8. LC filter method picked fw.abs to 10.
# Now constructing a model with a tunedParam. 
filteredTask_LM <- filterFeatures(ozoneTask, fval=filterVals, abs=unlist(tunedFeatures$x))
filteredModel_LM <- train(learner=lin, task=filteredTask_LM)

pd_LM <- predict(filteredModel_LM, newdata = as.data.frame(ozoneImp$data))
pd_LM_MSE <-(pd_LM$data$truth - pd_LM$data$response) ^ 2 %>% mean()

filteredTask_RM <- filterFeatures(ozoneTask, fval=filterValsRF, abs=unlist(tunedFeatures_RF$x))
filteredModel_RM <- train(learner=lin, task=filteredTask_RM)
pd_RF <- predict(filteredModel_RM, newdata = as.data.frame(ozoneImp$data))
pd_RF_MSE <- (pd_RF$data$truth - pd_RF$data$response) ^ 2 %>% mean()

filteredTask_ChiSq <- filterFeatures(task=ozoneTask, fval=filterValsChiSq, abs=unlist(tunedFeatures_ChiSq$x))
filteredModel_ChiSq <- train(learner=lin, task=filteredTask_ChiSq)
pd_CS <- predict(filteredModel_ChiSq, newdata=as.data.frame(ozoneImp$data))
pd_CS_MSE <- (pd_CS$data$truth - pd_CS$data$response)^2 %>% mean()

# Now the WRAPPER feature selection method... 
?FeatSelControl

featSelControl <- makeFeatSelControlSequential(method="sfbs")
selFeats <- selectFeatures(learner=lin, task=ozoneTask, resampling=makeResampleDesc(method="CV", iters=15), control = featSelControl)
ozoneSelFeat <- ozoneImp$data[, c("Ozone", selFeats$x)]
ozoneSelFeatTask <- makeRegrTask(data=ozoneSelFeat, target="Ozone")
ozoneSelWrapperModel <- train(lin, ozoneSelFeatTask)
pd_SelFeatWrapper <- predict(ozoneSelWrapperModel, newdata=ozoneImp$data)
pd_SelFeatWrapper_MSE <- (pd_SelFeatWrapper$data$truth - pd_SelFeatWrapper$data$response) ^ 2 %>% mean()




