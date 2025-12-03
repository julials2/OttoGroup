library(tidyverse)
library(tidymodels)
library(vroom)
library(embed)


#Otto Products < 0.55

train <- vroom("train.csv") %>% 
  mutate(target = factor(target))
  
test <- vroom("test.csv")
#####
## Recipe
#####
otto_recipe <- recipe(target~., data = train) %>% 
  step_rm(id) %>% 
  step_normalize(all_numeric_predictors()) 

prep <- prep(otto_recipe)
baked <- bake(prep, new_data = train)

#####
## Models
#####

#####
## random forest
#####
forest_mod <- rand_forest(mtry = tune(),
                          min_n = tune(),
                          trees = tune()) %>%
  set_engine("ranger") %>%
  set_mode("classification")

#####
## knn model
#####
# knn_model <- nearest_neighbor(neighbors = tune()) %>%
#   set_mode("classification") %>%
#   set_engine("kknn") 

##########
## Put into a workflow here
##########

#####
## random forest
#####
forest_workflow <- workflow() %>%
  add_recipe(otto_recipe) %>%
  add_model(forest_mod) 

#####
## knn 
#####
# knn_wf <- workflow() %>%
#   add_recipe(otto_recipe) %>%
#   add_model(knn_model) 

##########
## CV
##########

#####
## Grid for forest
#####
tuning_grid <- grid_regular(mtry(range = c(1, 9)),
                            min_n(),
                            trees(range(100,1000)),
                            levels = 5)

#####
## Grid for knn
#####
# tuning_grid <- grid_regular(neighbors(),
#                             levels = 5)

#####
## Datarobot or boosting
#####

#####
## Split data for CV
#####
folds <- vfold_cv(train, v = 5, repeats = 1)

#####
## Run CV
#####
CV_results <- tune_grid(
  forest_workflow,
  resamples = folds,
  grid = tuning_grid,
  metrics = metric_set(roc_auc))

#####
## Find best tuning parameters
#####
bestTune <- CV_results %>%
  select_best(metric="roc_auc")


##########
## Predictions
##########

#####
## Finalize workflow and fit it
#####
final_wf <-forest_workflow %>%
  finalize_workflow(bestTune) %>%
  fit(data = train)

#####
## Make predictions
#####
otto_predictions <- predict(final_wf,
                              new_data=test,
                              type="prob")

#####
## Format kaggle predictions
#####
kaggle_predictions <- otto_predictions %>% 
  bind_cols(., test) %>% 
  rename(Class_1 = `.pred_Class_1`,
         Class_2 = `.pred_Class_2`,
         Class_3 = `.pred_Class_3`,
         Class_4 = `.pred_Class_4`,
         Class_5 = `.pred_Class_5`,
         Class_6 = `.pred_Class_6`,
         Class_7 = `.pred_Class_7`,
         Class_8 = `.pred_Class_8`,
         Class_9 = `.pred_Class_9`) %>% 
  select(id, Class_1:Class_9)
  

write_csv(kaggle_predictions, file = "ottoRForest.csv")
