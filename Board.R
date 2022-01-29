### Loading in Tidy Tuesday Data for this week:
# ratings <- readr::read_csv('https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2022/2022-01-25/ratings.csv')
# details <- readr::read_csv('https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2022/2022-01-25/details.csv')

# save(ratings, file="ratings.rda")
# save(details, file = "details.rda")

load(file = "ratings.rda")
load(file = "details.rda")
pacman::p_load(tidymodels,tidyverse,tidytext,textrecipes)
analysis <- left_join(details, ratings, by = c("id"))
analysis$boardgamecategory <- gsub("\\[|\\]","",analysis$boardgamecategory)
analysis$boardgamecategory <- gsub("\\'","", analysis$boardgamecategory)
theme_set(theme_minimal())

### Explore the data:

analysis %>% ggplot(aes(average))+
  geom_histogram(fill="midnightblue", alpha=.8, bins=40)

## Most common board game categories:

tidy_board <- 
  analysis %>% 
  unnest_tokens(word,boardgamecategory)

tidy_board %>% 
  count(word,sort = TRUE)

## What is the mean rating for these game categories:

tidy_board %>%
  group_by(word) %>%
  summarise(
    n = n(),
    rating = mean(average)
  ) %>%
  ggplot(aes(n, rating)) +
  geom_hline(
    yintercept = mean(analysis$average), lty = 2,
    color = "gray50", size = 1.5
  ) +
  geom_jitter(color = "midnightblue", alpha = 0.7) +
  geom_text(aes(label = word),
            check_overlap = TRUE, 
            vjust = "top", hjust = "left"
  ) +
  scale_x_log10()


## Building Models:

set.seed(123)
board_split <- initial_split(analysis, strata = average)
board_train <- training(board_split)
board_test <- testing(board_split)

set.seed(234)
board_folds <- vfold_cv(board_train, strata = average)
board_folds

# Set up feature engineering steps:
board_rec <-
  recipe(average ~ boardgamecategory, data = analysis) %>%
  step_tokenize(boardgamecategory) %>%
  step_tokenfilter(boardgamecategory) %>%
  step_tfidf(boardgamecategory)

## just to check this works
prep(board_rec) %>% bake(new_data = NULL)


## Two model specifications to compare:

rf_spec <- 
  rand_forest(trees=500) %>% 
  set_mode("regression")

rf_spec

svm_spec <-
  svm_linear() %>%
  set_mode("regression")

svm_spec

## join up the specifications with a workflow:

svm_wf <- workflow(board_rec, svm_spec)
rf_wf <- workflow(board_rec, rf_spec)


### Evaluate the models:

doParallel::registerDoParallel()
contrl_preds <- control_resamples(save_pred = TRUE)

svm_rs <- fit_resamples(
  svm_wf,
  resamples = board_folds,
  control = contrl_preds
)

ranger_rs <- fit_resamples(
  rf_wf,
  resamples = board_folds,
  control = contrl_preds
)

# How do these 2 models compare?:

collect_metrics(svm_rs)
collect_metrics(ranger_rs)

# Visualise the results by comparing the predicted rating with the 
# true value:

bind_rows(
  collect_predictions(svm_rs) %>%
    mutate(mod = "SVM"),
  collect_predictions(ranger_rs) %>%
    mutate(mod = "ranger")
) %>%
  ggplot(aes(average, .pred, color = id)) +
  geom_abline(lty = 2, color = "gray50", size = 1.2) +
  geom_jitter(width = 0.5, alpha = 0.5) +
  facet_wrap(vars(mod)) +
  coord_fixed()

# not the best models but lets go with the random forest:

final_fitted <- last_fit(rf_wf, board_split)
collect_metrics(final_fitted)

# This is now a fitted workflow that we can use for prediction:

final_wf <- extract_workflow(final_fitted)
predict(final_wf, board_test[100, ])

# Directly inspect the coefficients, for each term, which one
# is more associated with higher ratings:

extract_workflow(final_fitted) %>%
  tidy() %>%
  filter(term != "Bias") %>%
  group_by(estimate > 0) %>%
  slice_max(abs(estimate), n = 10) %>%
  ungroup() %>%
  mutate(term = str_remove(term, "tfidf_boardgamecategory_")) %>%
  ggplot(aes(estimate, fct_reorder(term, estimate), fill = estimate > 0)) +
  geom_col(alpha = 0.8) +
  scale_fill_discrete(labels = c("low averages", "high averages")) +
  labs(y = NULL, fill = "Averages") 
















