install.packages("tidyverse")
install.packages("mlr")
install.packages("data.table")
install.packages("rpart.plot")

library(tidyverse)
library(mlr)
library(data.table) 
library(parallel)
library(parallelMap)
library(rpart)
library(rpart.plot)


data(Zoo, package="mlbench")
zooTib <- as_tibble(Zoo)
zooTib
glimpse(zooTib)

zooTib <- mutate_if(zooTib, is.logical, as.factor)
glimpse(zooTib)

## create classification task obj and a learner with rpart...

zooTask <- makeClassifTask(data=zooTib, target="type")
tree <- makeLearner(cl="classif.rpart")
tree

## First thing is to create hyper parameter space, in order to tune, and to crossvalidate. 
getParamSet(tree) 

treeParamSpace <- makeParamSet(
  makeIntegerParam("minsplit", lower=5, upper=20), 
  makeIntegerParam("minbucket", lower=3, upper=10), 
  makeNumericParam("cp", lower=0.01, upper=0.1),
  makeIntegerParam("maxdepth", lower=3, upper=10)
)

## Develop CV strategy for tuning. 
dtZoo<-zooTib %>% data.table()
summary(dtZoo) # first glance at each feature. [venomous] only has 8 cases. Is this enough to stratify for 7 classes? (mammal, bird, reptile, etc.)
               # Another fact is that each class particularly for reptil, amphibian, and insect, has very low few cases. 

randSearch <- makeTuneControlRandom(maxit=200)
cvFOrTuning <- makeResampleDesc(method="CV", stratify=FALSE, iters=5)

## tuning param space
parallelStartSocket(cpus=detectCores())
tunedTreeParams <- tuneParams(learner=tree, task=zooTask, resampling=cvFOrTuning, par.set=treeParamSpace, control=randSearch)
parallelStop()

tunedTreeParams

# Build model with the tuned hyper parameters...
  # tuned Tree Learner
tunedTree <- setHyperPars(learner=tree, par.vals=tunedTreeParams$x)
  # train model with tuned params.
tunedTreeModel <- train(learner=tunedTree, zooTask)
treeModelData <- getLearnerModel(tunedTreeModel)

# Cross Validating the newly created model...
outer <- makeResampleDesc(method="CV", iters=5)
treeWrapper <- makeTuneWrapper(learner="classif.rpart", resampling=cvForTuning, par.set=treeParamSpace, control=randSearch)
parallelStartSocket(cpus=detectCores())
cvWithTuning <- resample(treeWrapper, zooTask, resampling=outer)
parallelStop()

dtZoo[type=="amphibian",]

cvWithTuning$aggr
treeModelData$y
rpart.plot(treeModelData, roundint=FALSE, box.palette="BuBn", type=5)


outer <- makeResampleDesc(method="CV", iters=5)
parallelStartSocket(cpus=detectCores())
cvtunedAlready <- resample(tunedTree, zooTask, resampling=outer)
parallelStop()


zooXgb <- 






















data(Ozone, package="mlbench")
ozoneTib <- as.tibble(Ozone)
names(ozoneTib) <- c("Month", "Date", "Day", "Ozone", "Press_height", "Wind", "Humid", "Temp_Sand", "Temp_Monte", 
                     "Inv_height", "Press_grad", "Inv_temp", "Visib")
ozoneClean <- ozoneTib %>% mutate_if(is.factor, as.numeric) %>% filter(!is.na(Ozone))


lin <- mlr::makeLearner("regr.lm")
kFold <- makeResampleDesc(method="CV", iters=10)
featSelControl <- makeFeatSelControlSequential(method="sfbs")
imputeMethod <- imputeLearner("regr.rpart")
imputeWrapper <- makeImputeWrapper(learner=lin, classes=list(numeric=imputeMethod))
featSelWrapper <- makeFeatSelWrapper(learner=imputeWrapper, resampling=kFold, control=featSelControl)

ozoneTaskWithNAs <- makeRegrTask(data=ozoneClean, target="Ozone")
kFold3 <- makeResampleDesc("CV", iters = 3)

parallelStartSocket(cpu=detectCores())
lmCV <- resample(learner=featSelWrapper, task=ozoneTaskWithNAs, resampling=kFold3)
parallelStop()
lmCV 

lmCV$task.desc


wrapperModel <- train(featSelWrapper, ozoneSelFeatTask)





wrapperModel <- train(featSelWrapper, task=ozoneTaskWithNAs)
wrapperModelData <- getLearnerModel(wrapperModel)
summary(wrapperModelData)
