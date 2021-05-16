library(tidyverse)
library(mlr)


data("fuelsubset.task")
fuel <- getTaskData(fuelsubset.task)
fuelTib <- as_tibble(fuel)
names(fuelTib)

fuelTib <- fuelTib %>% mutate(rowId=1:nrow(.))
fuelUntidy <- fuelTib %>% gather(key="variable", value="absorbance", c(-heatan, -h20, -rowId)) %>% 
  mutate(spectrum=str_sub(variable, 1, 3), wavelength=as.numeric(str_extract(variable, "(\\d)+")))

str_extract("hello123", "(\\d)")
str_extract("hello123e456", "(\\d)+(e)(\\d)+")
str_extract("hello123e456", "(\\d)+(ef)(\\d)+")

fuelUntidy


#filter(fuelUntidy, wavelength %in% c(227, 228, 229)) %>% 
fuelUntidy %>% 
  ggplot(aes(absorbance, heatan, col=as.factor(wavelength))) + 
  facet_wrap(~ spectrum, scales="free_x") + 
  geom_smooth(se=FALSE, size=0.2) + 
  ggtitle("Absorbance vs heatan for each wavelength") + 
  theme_bw() + 
  theme(legend.position = "none")

  fuelUntidy %>% 
    ggplot(aes(absorbance, heatan, col=as.factor(wavelength))) + 
    facet_wrap(~ spectrum, scales="free_x") + 
    #geom_smooth(se=FALSE, size=0.2) + 
    geom_smooth(group=1, col="blue") + 
    ggtitle("Absorbance vs heatan for each wavelength") + 
    theme_bw() + 
    theme(legend.position = "none")
  
fuelUntidy %>%
  ggplot(aes(wavelength, absorbance, group=rowId, col=heatan)) + 
  facet_wrap( ~ spectrum, scales="free_x") + 
  geom_smooth(se=FALSE, size=0.2) + 
  ggtitle("Wavelength vs absorbance for each batch") + 
  theme_bw()

fuelUntidy %>% 
  ggplot(aes(h20, heatan)) + 
  #facet_wrap( ~ wavelength, scales="free_x") + 
  geom_point() +
  geom_smooth(se=FALSE) + 
  ggtitle("Humidity vs heatan") + 
  theme_bw()+ theme(legend.position="none")

fuelTask <- makeRegrTask(data=fuelTib, target="heatan")
kknn <- makeLearner(cl="regr.kknn")
kknnParamSpace <- makeParamSet(
  makeDiscreteParam(id="k", values=1:12)
)
gridSearch <- makeTuneControlGrid()
kFold <- makeResampleDesc(method="CV", iters=10)
library(parallel)
library(parallelMap)
parallelStart(cpus=detectCores())
tunedK <- tuneParams(kknn, task=fuelTask, resampling=kFold, par.set=kknnParamSpace, control=gridSearch)
parallelStop()
tunedK
knnTuningData <- generateHyperParsEffectData(tunedK)
knnTuningData
plotHyperParsEffect(knnTuningData, x="k", y="mse.test.mean", plot.type = "line") +
  theme_bw()
tunedkknn <- setHyperPars(learner=kknn, par.vals=tunedK$x)
# Now train with the tuned kknn learner. 
tunedKnnModel <- train(learner=tunedkknn, task=fuelTask)


# train randomForest model... Will skip rpart model because almost always it's outperformed by 
#  bagged and boosted learners. 
rForest <- makeLearner(cl="regr.randomForest")
forestParamSpace <- makeParamSet(
  makeIntegerParam(id="ntree", lower=50, upper=100),
  makeIntegerParam(id="mtry", lower=100, upper=367),
  makeIntegerParam(id="nodesize", lower=1, upper=10), 
  makeIntegerParam(id="maxnodes", lower=5, upper=30)
)
randSearch <- makeTuneControlRandom(maxit=100)
parallelStart(cpus=detectCores())
tunedForstParams <- tuneParams(rForest, task=fuelTask, par.set=forestParamSpace, resampling=kFold, control=randSearch)
parallelStop()

tunedRForest <- setHyperPars(learner=rForest, par.vals=tunedForstParams$x)
tunedRForestModel <- train(learner=tunedRForest, task=fuelTask)
tunedRForestModelData <- getLearnerModel(model=tunedRForestModel)

plot(tunedRForestModelData)


#Now training a XGBoost model...
xgbLearner <- makeLearner(cl="regr.xgboost")
xgbParamSpace <- makeParamSet(
  makeNumericParam(id="eta", lower=0, upper=1),
  makeNumericParam(id="gamma", lower=0, upper=10),
  makeIntegerParam(id="max_depth", lower = 1, upper=20),
  makeNumericParam(id="min_child_weight", lower=1, upper=10), 
  makeNumericParam(id="subsample", lower=0.5, upper=1),
  makeNumericParam(id="colsample_bytree", lower=0.5, upper=1), 
  makeIntegerParam(id="nrounds", lower=38, upper=43)
)

parallelStartSocket(cpus=detectCores())
tunedXgbParams <- tuneParams(learner=xgbLearner, task=fuelTask, resampling=makeResampleDesc(method="CV", iters=10),
                             par.set=xgbParamSpace, control=randSearch)
parallelStop()
tunedXgbLearner <- setHyperPars(learner=xgbLearner, par.vals=tunedXgbParams$x)
tunedXgbModel <- train(tunedXgbLearner, fuelTask)
tunedXgbModelData <- getLearnerModel(tunedXgbModel)
tunedXgbModelData
tunedXgbModelData$evaluation_log

class(tunedXgbModelData$evaluation_log)
ggplot(tunedXgbModelData$evaluation_log, aes(x=iter, y=train_rmse)) + 
  geom_line() + 
  geom_point() + theme_bw()

# All three model types are set up. ( kknn, random forest, xgboost ) Now build a benchmark comparison platform. 
