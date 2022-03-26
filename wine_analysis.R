library(tidyverse)
library(furrr)
library(GGally)
library(DirichletReg)
library(xgboost)
library(plsRglm)
library(parallel)

# Load the pre-fitted model in this directory.
try(load("./fitted_model.RData"),
    function(e) message("No fitted model available! Is this being run from a directory with fitted_model.RData?"))

load_wines <- function(reds = read.csv('./data/winequality-red.csv', sep=";"),
                       whites = read.csv('./data/winequality-white.csv', sep=";")) {
  rbind(mutate(reds, color = 'red'),
        mutate(whites, color = 'white'))
}

preproc <- function(wines) {
  # All cols are non-negative except color, so log transform them before standard scaling.
  #This is more valid from a prior theoretical standpoint, but also combats skews violating normal-ish assumptions.
  wines %>%
    mutate(across(c(-color,-quality), ~ scale(log(.x + 1e-5)))) %>%
    mutate(quality = factor(quality, ordered=TRUE)) %>%
    mutate(color = if_else(color == 'red', -1, 1)) %>%
    relocate(quality, .after=color)
}

plot_tsne_stack <- function(params, plot_data) {
  params$nt <- 3
  Rtsne(select(df, -quality), check_duplicates = FALSE)$Y %>%
    as.data.frame() %>%
    cbind(df) %>%
    ggplot(aes(x=V1, y=V2, color=quality)) +
      geom_point()
}


# Pair plots of some of the more important features.
wine_pairings <- function(df,
                          cols = c('color', 'alcohol', 'density', 'total.sulfur.dioxide', 'chlorides', 'quality'),
                          corrs_by='color') {
  # Default geom_point is hard to read here.
  lower_fn <- function(data, mapping) {
    ggplot(data=data, mapping=mapping) +
      geom_density_2d(bindwidth = 0.25)
  }
  upper_fn <- function(data, mapping) {
    #corrs <- data %>% #group_by(corrs_by) %>%
    ggplot(rm.na=TRUE) + annotate("text", x=1, y=1, label=cor(.x,.y))
  }
  df %>% select(all_of(cols)) %>%
    ggpairs(df,
            lower=list(continuous=wrap(lower_fn),
                       combo = wrap("facethist", binwidth=0.5)))
}

mae <- function (predicted, truth) {
    mean(abs(predicted - truth))
}

#Assign fold numbers to row indices.
partition_folds <- function (n, k_folds) {
  n_fold <- ceiling(n/k_folds)
  rep(1:k_folds, each = n_fold)[1:n] %>% sample()
}


xgb_preproc <- function(data, weight = rep(1, nrow(data))) {
  x <- data %>% select(-quality) %>% mutate(color = if_else(color == "red", -1, 1)) %>% as.matrix() 
  y <- data$quality %>% as.numeric()
  list(dmatrix=xgb.DMatrix(data = x, label = y, weight = weight),
       X = x,
       y = y)
}

dirichlet_reweight <- function(alpha, n_obs, n_resamples) {
  rdirichlet(n_resamples, rep(alpha, n_obs)) * n_obs
}

xgb_ensemble <- function(models) {
  class(models) <- "xgb_ensemble"
  models
}

train_xgb_ensemble <- function(data,
                               n_models = 2,
                               # Weight concentration of Dirichlet distribution on the nrow(data)-dimensional probability simplex.
                               resample_alpha = 1,
                               eff_trees = 100,
                               eta = eff_trees/nrounds,
                               max_depth = 6,
                               colsample_bytree = 0.5,
                               gamma = 1e-3,
                               l1_alpha = 1e-3,
                               # Number of booster iterations/trees per model.
                                nrounds = 100,
                                ...) {
  map(1:n_models,
      function(x) {
        xgb.train(data = xgb_preproc(data, weight = dirichlet_reweight(resample_alpha, nrow(data), 1))$dmatrix,
                 nrounds = nrounds,
                 params = list(objective = 'reg:squarederror',
                               alpha = l1_alpha,
                               max_depth = max_depth,
                               eta = eta))
       }) %>%
    xgb_ensemble(.)
}

# Generic fn for operators that can project as R matrices, as opposed to vectors with `predict`.
transform <- function(self, ...) {
  UseMethod("transform")
}

transform.xgb_ensemble <- function(self, newdata, ...) {
  if(is.data.frame(newdata)) { newdata <- xgb_preproc(newdata)$X}
  self %>%
    map(predict, newdata) %>%
    reduce(., rbind) %>%
    t()
}


# Split data into test/train partitions.
split <- function(data, holdout_prop = 0.2) {
  n <- nrow(data)
  split_i <- holdout_prop * n
  shuffled <- sample(1:n)
  
  ix_testing <- shuffled[1:split_i]
  ix_training <- shuffled[split_i:n]
  
  training <- data[ix_training,]
  testing <- data[ix_testing,]
  structure(list(training = training, testing = testing), class = "data_split")
}

# Segment data into nested folds.
segment_folds <- function(data, k_folds = 3) {
  data$fold <- partition_folds(nrow(data), k_folds)
  data
}

failed_fit <- function(err) {
  structure(err, class = "failed_fit")
}
predict.failed_fit <- function(self, newdata, ...) {
  rep(Inf, nrow(newdata))
}

model_spec <- function(fit_func, params, ...) {
  structure(list(fit = fit_func, params = params), class="model_spec")
}

fit <- function(self, ...) {
  UseMethod("fit")
}

fit.function <- function(self, ...) self(...)

fit.model_spec <- function(self, ...) {
  args <- list(...)
  # append the self$params whose names don't appear in fit args.
  fit_params <- append(args, self$params[setdiff(names(self$params), names(args))])
  fit(self$fit_func, fit_params)
}

fit_stack <- function(data,
                      #param_grid = xgb_grid,
                      nrounds = 100,
                      n_epochs = 3,
                      n_models = 10,
                      nt = 4,
                      lambda = 1,
                      eff_trees = 10,
                      resample_alpha = 1,
                      gamma = 0L,
                      l1_alpha = 1e-3,
                      alpha.pvals.expli = 0.05,
                      min_child_weight = 1,
                      backoff_resample_alpha = FALSE,
                      ...) {
  args <- c(as.list(environment()), list(...))
  xgb_models <- train_xgb_ensemble(data,
                                   n_models = n_models,
                                   nrounds=nrounds,
                                   eta = eff_trees/nrounds,
                                   gamma=gamma,
                                   lambda=lambda,
                                   l1_alpha = l1_alpha,
                                   resample_alpha=resample_alpha)
  transformed <- transform.xgb_ensemble(xgb_models, data) 
  # Per-point inverse variance reweightings.
  reweights <- 1L/(apply(transformed, 1, var))
  # plsRglm is very temperamental and won't converge for many param combinations, so just catch and wrap errors.
  
  pls <- tryCatch({
    plsRglm(dataX = transformed,
              dataY = data$quality,
              weights = reweights,
              # ordinal
              modele = "pls-glm-polr",
              nt = nt,
              pvals.expli = TRUE,
              alpha.pvals.expli = alpha.pvals.expli)
      },
      error = failed_fit)
  structure(list(xgb_models = xgb_models, pls = pls), class = c("stack", "list"))
}

# Returns list of predictions of both layers, so maybe don't have it as a predict.stack method.
predict_stack <- function(self, newdata=newdata, pls_prediction_type = "class", numeric_preds = TRUE) {
  tr_xgb <- transform(self$xgb_models,
                      newdata=newdata)
  
  pred_pls <- predict(self$pls,
              type = pls_prediction_type,
              newdata = tr_xgb)
  pred_xgb <- apply(tr_xgb, 1, mean)
  if (numeric_preds & is.factor(pred_pls)) {
    pred_pls <- as.numeric(as.character(pred_pls))
  }
  list(pred_pls = pred_pls,
       pred_xgb = pred_xgb)
}

predict.stack <- function(self, ...) {
  predict_stack(self, ...)$pred_pls
}

transform.plsRglmmodel <- function(self, newdata, ...) {
  newdata %*% self$wwetoile
}

transform.stack <- function(self, newdata, ...) {
  rf <- function(acc, model, ...) {
    transform(model, acc, ...)
  }
  Reduce(rf, self, newdata, ...)
}

fit.stack <- function(model_fns, ...) {
  init <- 
  rf <- function(acc) {
    fit(fitable, transform)
  }
  init <- fit(first(self), ...)
  
}

# Map over every kfold split.
map_folds <- function(folds, on_fold) {
  folds$fold %>%
    unique() %>%
    sort() %>%
    map(on_fold)
}

eval_fold_grid <- function(..., folded, fit = fit_stack, on_result = identity) {
  fparams <- list(...)
  
  holdout <- select(filter(folded, fold == fparams$fold), -fold)
  training <- select(filter(folded, fold != fparams$fold), -fold)
  args <- append(list(data=training), fparams)
  fitted <- do.call(fit, args)
  
  oof_preds <- predict_stack(fitted, holdout)
  fold_preds <- predict_stack(fitted, training)
  
  fparams$oof_mae_pls <- mae(holdout$quality, oof_preds$pred_pls)
  fparams$oof_mae_xgb <- mae(holdout$quality, oof_preds$pred_xgb)
  
  fparams$in_fold_mae_pls <- mae(training$quality, fold_preds$pred_pls)
  fparams$in_fold_mae_xgb <- mae(training$quality, fold_preds$pred_xgb)
  
  result <- as.data.frame(fparams)
  on_result(result)
  result
}
cv_grid <-  list(n_models = 20,
                 nrounds = 500,
                 l1_alpha = 0.,
                 lambda = c(0., 1.),
                 colsample_bytree = c(1., 0.5),
                 max_depth = c(7L, 3L),
                 eff_trees = c(200, 50),
                 nt = 5,
                 # Upper range is limited by plsRglm's numerical stability.
                 resample_alpha = c(1e-1, 0.2, 1.))

cv_grid_search <- function(grid, training_data, k_folds = 3, save_loc=NULL, ...) {
  folded <- segment_folds(training_data, k_folds)
  fgrid <- grid %>% append(list(fold = 1:k_folds)) %>% expand.grid(.) %>% sample()
  # train each param/fold, summarizing MAE for both xgb ensemble and final PLS.
  cv_results <- pmap(fgrid, eval_fold_grid, folded = folded, ...) %>% reduce(rbind)
  if(is.character(save_loc)) {
    write.csv(cv_results, row.names=FALSE, quote=FALSE)
  }
  cv_results
}

collect_folds <- function(folds,
                          is_metric_col = function(s) grepl("^(oof_|in_fold_)", s),
                          is_param_col = function(s) !is_metric_col(s) & (s != "fold"), s) {
  cols <- colnames(folds)
  metric_colnames <- Filter(is_metric_col, cols)
  param_colnames <- Filter(is_param_col, cols)
  folds %>%
    group_by(across(all_of(param_colnames))) %>%
    summarise(across(all_of(metric_colnames), mean)) %>%
    arrange(oof_mae_pls) %>%
    ungroup
}

fit_best <- function(cvfits, data, ..., method = fit_stack) {
  best <- collect_folds(cvfits)[1,] %>%  as.list()
  do.call(method, append(list(data = data), append(best, list(...))))
}
nice_metrics <- function(model, split_data) {
  te <- split_data$testing$quality
  tr <- split_data$training$quality
  tr_pred <- predict(model, split_data$training)
  test_pred <- predict(model, split_data$testing)
  tibble("kind"=c("testing", "training"),
         "MAE"= c(mae(test_pred, te),
                  mae(tr_pred, tr)),
         "label accuracy" = c(mean(te == test_pred),
                              mean(tr == tr_pred)))
}
