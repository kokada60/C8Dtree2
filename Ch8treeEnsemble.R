install.packages("tidyverse")
install.packages("mlr") 
install.packages("data.table")

library(tidyverse)
library(mlr)
library(data.table) 

data(Zoo, package="mlbench")
zooTib <- as_tibble(Zoo)
zooTib
glimpse(zooTib)

zooTib %>% mutate_if(is.logical, as.factor) %>% glimpse() -> zooTib
glimpse(zooTib)

## create classification task obj and a learner with rpart...
zooTask <- makeClassifTask(data = zooTib, target="type") 
tree <- makeLearner("classif.rpart")

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
cvForTuning <- makeResampleDesc(method="CV", stratify=TRUE, iters=5)

library(parallel)
library(parallelMap)

