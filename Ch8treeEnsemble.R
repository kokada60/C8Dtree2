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
cvForTuning <- makeResampleDesc(method="CV", stratify=FALSE, iters=5)

## tuning param space
parallelStartSocket(cpus=detectCores())
tunedTreeParams <- tuneParams(learner=tree, task=zooTask, resampling=cvForTuning, par.set=treeParamSpace, control=randSearch)
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



rpart.plot(treeModelData, roundint=FALSE, box.palette="BuBn", type=5)

