library(tidyverse)
library(mlr)

data(banknote, package="mclust")
swissTib <- as_tibble(banknote)
swissTib
swissUntidy <- swissTib %>% gather(key="Variable", value="Value", -Status)
swissUntidy %>% ggplot(aes(x=Value)) + 
  facet_wrap(~Variable, scales="free_x") + 
  geom_histogram()

library(GGally)
ggpairs(swissTib, mapping=aes(col=Status)) + theme_bw()

# Now pick out PCAs. 
pca <- select(swissTib, -Status) %>% 
  prcomp(center=TRUE, scale=TRUE)
pca$x
summary(pca)
