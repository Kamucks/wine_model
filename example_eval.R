source("./wine_analysis.R")
wines <- load_wines()
newdata <- split(wines, holdout_prop = 0.2)$testing
predict(model, newdata = newdata)
