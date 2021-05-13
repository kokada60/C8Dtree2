library(mlr) 
library(tidyverse)
library(glmnet)

data(Iowa, package="lasso2")
iowaTib <- as.tibble(Iowa)
iowaTib
iowaUntidy <- gather(iowaTib, key="Variable", value="Value", -Yield)
ggplot(iowaUntidy,aes(x=Value, y=Yield)) + 
  facet_wrap(~Variable, scales="free_x") + 
  geom_point() + 
  geom_smooth(method="lm", formula=y ~ x) + 
  theme_bw()

  # A pure ridge regularization ridge learner, 
  # as indicated by alpha set to 0...
iowaTask <- makeRegrTask(data=iowaTib, target="Yield")
ridgeLearner <- makeLearner(cl="regr.glmnet", alpha=0, id="ridgeLearningisFun")

  # now filter feature selection on the iowa dataset...
filterVals_RFSRC <- generateFilterValuesData(task=iowaTask, 
                                       method="randomForestSRC_importance")
plotFilterValues(filterVals_RFSRC)
filterVals_LM <- generateFilterValuesData(task=iowaTask, 
                                       method="linear.correlation")
plotFilterValues(filterVals_LM)
filterVals_RF <- generateFilterValuesData(task=iowaTask, 
                                          method="randomForest_importance")
plotFilterValues(filterVals_RF)

# Above are plots of feature ranking done in the usual RF_SRC, RF_importance, Linear_Correlation
#methods. But feature filtering will not be done here. The Ridge regularization algorithm will determine 
#how big of penalties will be applied toward non-contributing features. Now the next step is to 
#tune for alpha hyp-param that controls just how big a penalty to apply to the parameter estimates. 
# Lambda=0 applies no penalty, larget the lambda more the parameters are shrunk toward zero. Now must find the 
# optimal lambda value. CV of course. 

# Build Hyper Parameter space.. Ceiling is set arbitrary to 15...
ridgeParamSpace <- makeParamSet(
  makeNumericParam(id="s", lower=0, upper=30) # LAMBDA... Seems MSE just keeps increasing past 15... 5 appears to be the global minumum...
)
randomSearch <- makeTuneControlRandom(maxit = 200)
cvForTuning <- makeResampleDesc(method="RepCV", folds=3, reps=15)
library(parallelMap)
library(parallel)
parallelStartSocket(cpus=detectCores())
tunedRidgeParams <- tuneParams(learner=ridgeLearner, task=iowaTask, 
                               par.set=ridgeParamSpace,
                               resampling=cvForTuning, control=randomSearch)
parallelStop()

tunedRidgeParams

ridgeTuneData <- generateHyperParsEffectData(tunedRidgeParams)
plotHyperParsEffect(ridgeTuneData, x="s", y="mse.test.mean", plot.type="line") + theme_bw()

# Now the best performing hyp-param lambda was apprx to 6, a model will be built on that premise.
# First, take the ridgeLearner and set its hyper-parameter space to tunedRidgeParams, a returned object 
#from parameter tuning from earlier. 
tunedRidgeParams$learner$id
tunedRidgeParams$x

tunedRidgeLearner <- setHyperPars(learner=ridgeLearner, par.vals=tunedRidgeParams$x)
tunedRidgeModel <- train(tunedRidgeLearner, task=iowaTask)
tunedRidgeModelData <- getLearnerModel(tunedRidgeModel)
ridgeCoefs <- coef(tunedRidgeModelData, s=tunedRidgeParams$x$s)
View(ridgeCoefs)

Now we hav 