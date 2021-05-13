library(mlr) 
library(tidyverse)
library(glmnet)
library(tidytext)


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

#Now we have this model built, let's compare this side-by-side with unregularized lm OLS model. 
lm_Model <- lm(formula=Yield ~ ., data=iowaTib)
lmCoefs <- coef(lm_Model)

coefTib <- tibble(Coef=rownames(ridgeCoefs)[-1], Ridge=as.vector(ridgeCoefs)[-1], Lm=as.vector(lmCoefs[-1]))
coefUntidy <- gather(coefTib, key="Model", value=Beta, -Coef)
# Plot the predictors of both models

ggplot(coefUntidy, aes(x=Coef, y=Beta, fill=Model)) + 
  geom_bar(stat="identity", col="black") + 
  facet_wrap(~Model) + 
  theme_bw() + 
  theme(legend.position="none")

coefUntidy
coefUntidy %>% 
  ggplot(aes(reorder(Coef, Beta), Beta, fill=Model)) + 
  geom_bar(stat="identity", col="black") + 
  facet_wrap(~Model) + 
  
  theme_bw() + 
  theme(legend.position="none")

lassoLearner <- makeLearner(cl="regr.glmnet", alpha=1, id="lasso")
parallelStartSocket(cpus=detectCores())
tunedLassoParms <- tuneParams(learner=lassoLearner, task=iowaTask, 
                              par.set=ridgeParamSpace,
                              resampling=makeResampleDesc(method="RepCV", folds=3, reps=15), 
                              control=makeTuneControlRandom(maxit=200))
parallelStop()
lassoTuningData <- generateHyperParsEffectData(tunedLassoParms)
plotHyperParsEffect(lassoTuningData, x="s", y="mse.test.mean", plot.type="line") + 
  theme_bw()
tunedLassoLearner <- setHyperPars(learner=lassoLearner, par.vals=tunedLassoParms$x)
tunedLassoModel <- train(tunedLassoLearner, task=iowaTask)
tunedLassoModelData <- getLearnerModel(tunedLassoModel)
lassoCoefs <- coef(tunedLassoModelData, s=tunedLassoParms$x$s)
lassoCoefs
ridgeCoefs
coefTib <- coefTib %>% mutate(Lasso=as.vector(lassoCoefs)[-1])
coefUntidy <- coefTib %>% gather(key="Model", value="Beta", -Coef)
ggplot(coefUntidy, aes(reorder(Coef, Beta), Beta, fill=Model)) + 
  geom_bar(stat="identity", col="black") + 
  facet_wrap(~Model) + 
  theme_bw()

#Now that the Lasso model is complete, now lets train an elastic net model. This 
# will be done by including "Alpha" hyper param to the set of parameter to be tuned by cross validation. 
elasticLearner <- makeLearner(cl="regr.glmnet", id="elastic") 
elasticParamSet <- makeParamSet(
  makeNumericParam(id="s", lower=0, upper=15), 
  makeNumericParam(id="alpha", lower=0, upper=1)
)
randomSearch <- makeTuneControlRandom(maxit=400)
repCV <- makeResampleDesc(method="RepCV", fold=3, reps=15)
parallelStartSocket(cpus=detectCores())
tunedElasticParams <- tuneParams(learner=elasticLearner, task=iowaTask, 
                                 resampling=repCV, par.set=elasticParamSet, 
                                 control=randomSearch)
parallelStop()
tunedElasticParamData <- generateHyperParsEffectData(tunedElasticParams)
plotHyperParsEffect(tunedElasticParamData, x="s", y="alpha", z="mse.test.mean", 
                    interpolate = "regr.kknn", 
                    plot.type="heatmap",
                    ) +
  scale_fill_gradientn(colours=terrain.colors(5)) + 
  geom_point(x=tunedElasticParams$x$s, y=tunedElasticParams$x$alpha, col="white") + 
  theme_bw()


plotHyperParsEffect(tunedElasticParamData, x="s", y="alpha", z="mse.test.mean", 
                    interpolate = "regr.kknn", 
                    plot.type="contour",
                    show.experiments = TRUE) +
  scale_fill_gradientn(colours=terrain.colors(5)) + 
  geom_point(x=tunedElasticParams$x$s, y=tunedElasticParams$x$alpha, col="white") + 
  theme_bw()

# Not illustrative at all...
plotHyperParsEffect(tunedElasticParamData, x="s", y="alpha", z="mse.test.mean", 
                    #interpolate = "regr.kknn", 
                    plot.type="scatter"
                    #show.experiments = TRUE
                    ) +
  #scale_fill_gradientn(colours=terrain.colors(5)) + 
  geom_point(x=tunedElasticParams$x$s, y=tunedElasticParams$x$alpha, col="white") + 
  theme_bw()

tunedElasticLearner <- setHyperPars(elasticLearner, par.vals = tunedElasticParams$x)
tunedElasticModel <- train(tunedElasticLearner, task=iowaTask)
tunedElasticModelData <- getLearnerModel(model=tunedElasticModel)
tunedElasticCoefs <- coef(tunedElasticModelData, s=tunedElasticParams$x$s)
coefTib$Elastic <- as.vector(tunedElasticCoefs)[-1]
coefUntidy <- coefTib %>% gather(key="Model", value="Beta", -Coef)

ggplot(coefUntidy, aes(x=reorder(Coef, Beta), y=Beta, fill=Model)) + 
  geom_bar(stat="identity", col="black") + 
  facet_wrap(~Model) + 
  theme_bw() 

library(plotmo)
plotres(tunedRidgeModelData)
plotres(tunedLassoModelData)
plotres(tunedElasticModelData)
