##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# # if using R 3.6 or earlier:
# movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
#                                             title = as.character(title),
#                                             genres = as.character(genres))
# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

##########################################################
# PROJECT
##########################################################

# we will be judging the accuracy of our predictions using RMSE,
# just as we did in the 'Recommendation Systems' section of the textbook
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

# we split the edx data into train and test sets
# get sample size so we can determine how big the test set will need to be
nrow(edx)
# we have over 9 million rows - test set can be as low as 10% of total
# and still have close to a million rows of test data

# set random seed and use data partitioning to create train and test sets from the edx dataset
set.seed(7623, sample.kind = "Rounding")
test_ind <- createDataPartition(edx$rating, times = 1, p = 0.1, list = FALSE)
train_set <- edx %>% slice(-test_ind)
temp <- edx %>% slice(test_ind)

# Make sure userId and movieId in test set are also in train set
test_set <- temp %>% 
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

# Add rows removed from test set back into train set
removed <- anti_join(temp, test_set)
train_set <- rbind(train_set, removed)
rm(temp, removed)

# We start similarly to the comprehension check in section 6.2, first by calculating
# the mean of all ratings and fitting a naive Bayesian model
# (as in 6.2, we won't use 'hat' notation in the code)
# Y_u,i = mu + e_u,i
mu <- mean(train_set$rating)
mu

# As in section 6.2, we will be comparing different approaches as we go along
rmse_results <- tibble(method = "Just the average", RMSE = RMSE(test_set$rating, mu))
rmse_results

# we then calculate the bias for each movie to fit the model
# Y_u,i = mu + b_i + e_u,i

# we will use regularisation (penalised least squares) to account for movies
# with relatively few ratings having greater variance

# we define functions to calculate b_i for a given value of lambda and to predict
# ratings based on Y_u,i = mu + b_i + e_u,i
calculate_b_i <- function(training_data, lambda, m=mu){
  training_data %>%
    group_by(movieId) %>%
    summarise(b_i = sum(rating - m)/(n() + lambda))
}

predict_model_1 <- function(training_data, test_data, lambda, m=mu){
  b_i <- calculate_b_i(training_data, lambda)
  test_data %>%
    left_join(b_i, by = "movieId") %>%
    mutate(pred = m + b_i) %>%
    pull(pred)
}

# first we use cross-validation to choose lambda, the tuning parameter

# record start time so we can monitor how long the calculation takes
time1 <- now()

lambdas <- seq(0, 10, 0.25)
rmses <- sapply(lambdas, function(l){
  predictions <- predict_model_1(train_set, test_set, l)
  return(RMSE(predictions, test_set$rating))
})
lambda <- lambdas[which.min(rmses)]
lambda

# and then run the model using the optimal lambda and calculate the RMSE
predicted_ratings <- predict_model_1(train_set, test_set, lambda)
model_1_rmse <- RMSE(predicted_ratings, test_set$rating)

# record the time after calculation and print the total time taken
time2 <- now()
time2 - time1

# show lambdas and RMSEs graphically - to show how we can cut down on cross-validation time
tibble(lambda = lambdas, RMSE = rmses) %>%
  ggplot(aes(lambda, RMSE)) +
  geom_point()

# given the parabolic nature, we can cut down on run time by implementing an algorithm
# that test lambdas in increments of 2, then 1, then 0.5, then 0.25

# first define function to calculate RMSEs for given list of lambdas using a particular
# prediction function
calculate_RMSEs <- function(prediction_function, lambdas, training_data, test_data){
  sapply(lambdas, function(l){
    predictions <- prediction_function(training_data, test_data, l)
    return(RMSE(predictions, test_data$rating))
  })
}

# then define function to return the best lambda and corresponding RMSE for a given
# list of lambdas
best_lambda <- function(prediction_function, lambdas, training_data, test_data, lambda_rmse){
  best <- lambda_rmse
  rmses <- calculate_RMSEs(prediction_function, lambdas, training_data, test_data)
  if(min(rmses) < lambda_rmse$rmse){
    best$rmse <- min(rmses)
    best$lambda <- lambdas[which.min(rmses)]
  }
  best
}

# then define a function to narrow in on the best lambda by examining small subsets
# of the total list of the sequence seq(0, 10, 0.25) - thus saving time by running
# the prediction model 5 + 2 + 2 + 2 = 11 times instead of 1 + 10 x 4 = 41 times
optimise_lambda <- function(prediction_function, training_data, test_data){
  best <- tibble(lambda = 0, rmse = 9999)
  lambdas <- seq(1, 9, 2)
  best <- best_lambda(prediction_function, lambdas, training_data, test_data, best)
  lambdas <- c(best$lambda - 1, best$lambda + 1)
  best <- best_lambda(prediction_function, lambdas, training_data, test_data, best)
  lambdas <- c(best$lambda - 0.5, best$lambda + 0.5)
  best <- best_lambda(prediction_function, lambdas, training_data, test_data, best)
  lambdas <- c(best$lambda - 0.25, best$lambda + 0.25)
  best <- best_lambda(prediction_function, lambdas, training_data, test_data, best)
  best
}

# the above function also saves us from re-running the model for the correct lambda where
# we only need to know RMSE. It could be done more elegantly as a recursive function but
# as we're looking at the same range of possible lambdas every time here we can leave the
# code as is

# confirm that the above algorithm returns the same value of lambda
time1 <- now()
new_method <- optimise_lambda(predict_model_1, train_set, test_set)
time2 <- now()
print(new_method)
print(new_method$rmse == model_1_rmse)
print(new_method$lambda == lambda)

# check the algorithm saves time
time2 - time1

# from now on we will use the algorithmic approach

# add RMSE to our RMSE results table
rmse_results <- bind_rows(rmse_results,
                          tibble(method="Regularised Movie Effect Model",
                                 RMSE = model_1_rmse ))
rmse_results

# we then calculate user bias to fit the model (also using regularisation)
# Y_u,i = mu + b_i + b_u + e_u,i

# we define functions to calculate b_u for a given value of lambda and to predict
# ratings based on Y_u,i = mu + b_i + b_u + e_u,i
calculate_b_u <- function(training_data, lambda, b_i, m=mu){
  training_data %>%
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - mu - b_i)/(n()+lambda))
}

predict_model_2 <- function(training_data, test_data, lambda, m=mu){
  b_i <- calculate_b_i(training_data, lambda)
  b_u <- calculate_b_u(training_data, lambda, b_i)
  test_data %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)
}

# calculate optimal lambda and RMSE for model 2, and add to results table
time1 <- now()
optimal <- optimise_lambda(predict_model_2, train_set, test_set)
model_2_rmse <- optimal$rmse
rmse_results <- bind_rows(rmse_results,
                          tibble(method="Regularised Movie + User Effects Model",
                                 RMSE = model_2_rmse ))
time2 <- now()
time2 - time1

# print RMSE results
rmse_results

# now we add to the recommendation system. firstly we will look at time
# and genre effects (as suggested in section 6.2 comprehension check)

# movies have the release year in the title but this doesn't tell us WHEN in the year
# the movie was released - so we won't know e.g. which reviews are from immediately after release

# instead we will look at the first review (both for each movie and each user) in the dataset
# and calculate the time elapsed since that review for each subsequent review

# to be certain that we are getting the correct first review we would need to note the timestamp
# of the first review using both the edx and validation data sets - otherwise we won't know the
# time of the first review if it is in the validation set. However, we have been instructed
# to not use the validation set at all for model training. To be cautious, we calculate time
# elapsed based on the first reviews found in the edx dataset - noting that there will be some
# resulting inaccuracies if we use a time-based model on the validation set in the end

# get timestamp of first review for each movie
first_reviews_by_movie <- edx %>%
  group_by(movieId) %>%
  summarise(first_movie_review = min(timestamp, na.rm = TRUE))

# and timestamp of first review by each user
first_reviews_by_user <- edx %>%
  group_by(userId) %>%
  summarise(first_user_review = min(timestamp, na.rm = TRUE))

# check visualisation for effect of time since first movie review on residuals on test set
library(lubridate)
test_set %>%
  left_join(first_reviews_by_movie, by = "movieId") %>%
  mutate(time_since_first_movie_review = round((timestamp - first_movie_review) / (60 * 60 * 24)),
         pred = predicted_ratings,
         res = rating - pred) %>%
  group_by(time_since_first_movie_review) %>%
  summarise(mean_residual = mean(res)) %>%
  ggplot(aes(time_since_first_movie_review, mean_residual)) +
  geom_point() +
  geom_smooth()

# and first user review
test_set %>%
  left_join(first_reviews_by_user, by = "userId") %>%
  mutate(time_since_first_user_review = round((timestamp - first_user_review) / (60 * 60 * 24)),
         pred = predicted_ratings,
         res = rating - pred) %>%
  group_by(time_since_first_user_review) %>%
  summarise(mean_residual = mean(res)) %>%
  ggplot(aes(time_since_first_user_review, mean_residual)) +
  geom_point() +
  geom_smooth()

# there may be a slight time effect based on time since first movie review (time since first user review looked
# to have no effect)

# the geom_smooth function uses the loess method to determine the conditional
# mean. We will also use loess to further refine our prediction according to time since first movie review
loess_b_ti <- function(training_data, b_i, b_u, first_reviews_by_movie, seed = 142){
  set.seed(seed, sample.kind = "Rounding")
  training_data %>%
    left_join(b_i, by="movieId") %>%
    left_join(b_u, by="userId") %>%
    left_join(first_reviews_by_movie, by = "movieId") %>%
    mutate(pred = mu + b_i + b_u,
           error = rating - pred,
           days_since_first_movie_review = round((timestamp - first_movie_review) / (60 * 60 * 24))) %>%
    group_by(days_since_first_movie_review) %>%
    summarise(avg_error = mean(error)) %>%
    loess(avg_error ~ days_since_first_movie_review, data = .)
}

calculate_b_ti <- function(b_ti_loess, range){
  tibble(days_since_first_movie_review = range) %>%
    mutate(b_ti = predict(b_ti_loess, days_since_first_movie_review))
}

predict_model_3 <- function(training_data, test_data, lambda, first_reviews_by_movie, m=mu){
  b_i <- calculate_b_i(training_data, lambda)
  b_u <- calculate_b_u(training_data, lambda, b_i)
  b_ti_loess <- loess_b_ti(training_data, b_i, b_u, first_reviews_by_movie)
  max_days <- bind_rows(training_data, test_data) %>%
    left_join(first_reviews_by_movie, by = "movieId") %>%
    mutate(days_since_first_movie_review = round((timestamp - first_movie_review) / (60 * 60 * 24))) %>%
    summarise(max = max(days_since_first_movie_review)) %>%
    pull(max)
  b_ti <- calculate_b_ti(b_ti_loess, 0:max_days)
  test_data %>%
    left_join(first_reviews_by_movie, by = "movieId") %>%
    left_join(b_i, by="movieId") %>%
    left_join(b_u, by="userId") %>%
    mutate(days_since_first_movie_review = round((timestamp - first_movie_review) / (60 * 60 * 24)),
           # set days since to zero in cases where they are negative using validation set
           days_since_first_movie_review = ifelse(days_since_first_movie_review < 0,
                                                  0,
                                                  days_since_first_movie_review)) %>%
    left_join(b_ti, by = "days_since_first_movie_review") %>%
    mutate(pred = mu + b_i + b_u + b_ti) %>%
    pull(pred)
}

# predict new ratings
predicted_ratings <- predict_model_3(train_set, test_set, lambda, first_reviews_by_movie)

# test RMSE
model_3_rmse <- RMSE(predicted_ratings, test_set$rating)
rmse_results <- bind_rows(rmse_results,
                          tibble(method="Movie + User Effects + Time Effects Model",  
                                 RMSE = model_3_rmse ))
rmse_results

# this seems to have very, very marginally improved our prediction. When accounting also
# for inaccuracies that may present when using the validation set, we discard the time model
# from our analysis

# we now investigate genre effects

# look at residuals graphically by genre to see if there looks to be an effect
# for now we keep the genres as written in the column
test_set %>%
  mutate(pred = predicted_ratings,
         res = rating - pred) %>%
  group_by(genres) %>%
  summarise(n = n(), mean_residual = mean(res, na.rm = TRUE)) %>%
  ungroup() %>%
  mutate(num_reviews = ifelse(n < 20, "less than 20 reviews", "less than 100 reviews"),
         num_reviews = ifelse(n >= 100, "100+ reviews", num_reviews)) %>%
  ggplot(aes(genres, mean_residual, color = num_reviews)) +
  geom_point() +
  theme(axis.text.x = element_blank(),
        axis.ticks.x = element_blank())

# residuals are around zero, except where a particular genre has a small number of reviews 
# (and therefore more variance) - don't investigate further

# Next we will try similar movies/users - as per textbook section 33.11

# we will try to cluster similar movies and similar users into groups. Because we have
# a large data set, we will need to filter it down to make the calculation workable

# first we will cluster movies. To do this we will whittle down the number of users
# - trial/error led to 1,000 users, otherwise run times get very long

# filter by top 1,000 users
user_list <- train_set %>%
  group_by(userId) %>%
  summarise(n = n()) %>%
  top_n(1000, n) %>%
  pull(userId)

# further filter by only choosing movies with 10+ reviews

# check how many movies this leaves us - with 500 users was 6,377 with 20+ reviews, 8,085 with 10+
# HOWEVER k-clustering did not always converge with only 10+
# BUT it does when we try 10+ with top 1000 users - 8,531 movies
num_movies <- train_set$movieId %>%
  unique() %>%
  length()
num_movies_filtered <- train_set %>%
  filter(userId %in% user_list) %>%
  group_by(movieId) %>%
  filter(n() >= 10) %>%
  ungroup() %>%
  .$movieId %>%
  unique() %>%
  length()
print(paste0("Number of movies: ", num_movies))
print(paste0("Number of movies after filtering: ", num_movies_filtered))
  
# create matrix of all movies with 10+ reviews and our top 1000 users
x <- train_set %>%
  filter(userId %in% user_list) %>%
  group_by(movieId) %>%
  filter(n() >= 10) %>%
  ungroup() %>%
  select(movieId, userId, rating) %>%
  pivot_wider(names_from = userId, values_from = rating)

# convert to matrix of users and movie scores
row_names <- x$movieId
x <- x[,-1] %>% as.matrix()
x <- sweep(x, 2, colMeans(x, na.rm = TRUE))
x <- sweep(x, 1, rowMeans(x, na.rm = TRUE))
rownames(x) <- row_names

# convert NAs to zero
x_0 <- x
x_0[is.na(x_0)] <- 0

# Use k-clustering to group movies into 10 groups - takes 35s with top 1000 raters (and 10+ reviews)
library(stats)
set.seed(4334)
time1 <- now()
k <- kmeans(x_0, centers = 10, nstart = 10, iter.max = 100) # gets error message if not enough iterations are set - 'did not converge in 10 iterations'
groups <- k$cluster
movie_groups <- tibble(movieId = as.numeric(names(groups)), movie_group = groups)
time2 <- now()
time2 - time1

# View a selection of the groups
train_set %>%
  left_join(movie_groups, by = "movieId") %>%
  group_by(movieId) %>%
  summarise(title = first(title), movie_group = first(movie_group)) %>%
  filter(movie_group == 1) %>%
  pull(title) %>%
  .[1:10]

train_set %>%
  left_join(movie_groups, by = "movieId") %>%
  group_by(movieId) %>%
  summarise(title = first(title), movie_group = first(movie_group)) %>%
  filter(movie_group == 2) %>%
  pull(title) %>%
  .[1:10]

# group 1 seems to be critically acclaimed drams, group 2 broader comedies/schlock

# the groups appear to have coherent themes, indicating the algorithm has
# chosen pretty well

# group users

# first whittle down to top movies
movie_list <- train_set %>%
  group_by(movieId) %>%
  summarise(n = n()) %>%
  top_n(500, n) %>%
  pull(movieId)

# create matrix of the top movies reviews and users with at least 10 reviews
x <- train_set %>%
  filter(movieId %in% movie_list) %>%
  group_by(userId) %>%
  filter(n() >= 10) %>%
  ungroup() %>%
  select(movieId, userId, rating) %>%
  pivot_wider(names_from = movieId, values_from = rating)

row_names <- x$userId
x <- x[,-1] %>% as.matrix()
x <- sweep(x, 2, colMeans(x, na.rm = TRUE))
x <- sweep(x, 1, rowMeans(x, na.rm = TRUE))
rownames(x) <- row_names

# convert NAs to zero
x_0 <- x
x_0[is.na(x_0)] <- 0

# Use k-clustering to group movies into 10 groups - takes ~8.5 minutes
time1 <- now()
set.seed(4217)
k <- kmeans(x_0, centers = 10, nstart = 10, iter.max = 1000, algorithm = "MacQueen") # gets error message if not enough iterations are set - 'did not converge in 10 iterations'
# also need to use MacQueen algorithm to avoid error 'Quick-TRANSfer stage steps exceeded maximum'
groups <- k$cluster
user_groups <- tibble(userId = as.numeric(names(groups)), user_group = groups)
time2 <- now()
time2 - time1

# now we have movie and user groups, determine group averages

# calculate bias from user u for movies from movie group I (b_u,I) to fit the model
# (using regularisation)
# Y_u,i = mu + b_i + b_u + b_u,I + e_u,i

# create functions to estimate b_uI and predict ratings
calculate_b_uI <- function(training_data, lambda, b_i, b_u, mov_groups, m=mu){
  training_data %>%
    left_join(b_i, by="movieId") %>%
    left_join(b_u, by="userId") %>%
    left_join(mov_groups, by = "movieId") %>%
    mutate(movie_group = ifelse(is.na(movie_group), 99, movie_group)) %>%
    group_by(userId, movie_group) %>%
    summarise(b_uI = sum(rating - mu - b_i - b_u)/(n()+lambda))
}

predict_model_4 <- function(training_data, test_data, lambda,
                            mov_groups = movie_groups, m=mu){
  b_i <- calculate_b_i(training_data, lambda)
  b_u <- calculate_b_u(training_data, lambda, b_i)
  b_uI <- calculate_b_uI(training_data, lambda, b_i, b_u, mov_groups)
  test_data %>%
    left_join(b_i, by="movieId") %>%
    left_join(b_u, by="userId") %>%
    left_join(mov_groups, by = "movieId") %>%
    mutate(movie_group = ifelse(is.na(movie_group), 99, movie_group)) %>%
    left_join(b_uI, by = c("userId", "movie_group")) %>%
    mutate(b_uI = ifelse(is.na(b_uI), 0, b_uI),
           pred = mu + b_i + b_u + b_uI) %>%
    pull(pred)
}

# calculate optimal lambda and RMSE for model 4, and add to results table
time1 <- now()
best <- optimise_lambda(predict_model_4, train_set, test_set)
time2 <- now()
print(time2 - time1)
model_4_rmse <- best$rmse
rmse_results <- bind_rows(rmse_results,
                          tibble(method="Regularised Movie + User + Movie Group Effects Model",
                                 RMSE = model_4_rmse ))
rmse_results

# took 3m20s

# we've improved the model!!!

# calculate bias from user group U for movie i (b_i,U) to fit the model
# (using regularisation)
# Y_u,i = mu + b_i + b_u + b_u,I + b_i,U + e_u,i

# create functions to estimate b_i,U and predict ratings
calculate_b_iU <- function(training_data, lambda, b_i, b_u, b_uI,
                           mov_groups, u_groups, m=mu){
  training_data %>%
    left_join(b_i, by="movieId") %>%
    left_join(b_u, by="userId") %>%
    left_join(mov_groups, by = "movieId") %>%
    left_join(u_groups, by = "userId") %>%
    mutate(movie_group = ifelse(is.na(movie_group), 99, movie_group),
           user_group = ifelse(is.na(user_group), 99, user_group)) %>%
    left_join(b_uI, by = c("userId", "movie_group")) %>%
    group_by(movieId, user_group) %>%
    summarise(b_iU = sum(rating - mu - b_i - b_u - b_uI)/(n()+lambda))
}

predict_model_5 <- function(training_data, test_data, lambda, mov_groups = movie_groups,
                            u_groups = user_groups, m=mu){
  # calculate bias terms from training data
  b_i <- calculate_b_i(training_data, lambda)
  b_u <- calculate_b_u(training_data, lambda, b_i)
  b_uI <- calculate_b_uI(training_data, lambda, b_i, b_u, mov_groups)
  b_iU <- calculate_b_iU(training_data, lambda, b_i, b_u, b_uI, mov_groups, u_groups)
  # use bias terms to predict test data ratings
  test_data %>%
    left_join(b_i, by="movieId") %>%
    left_join(b_u, by="userId") %>%
    left_join(mov_groups, by = "movieId") %>%
    left_join(u_groups, by = "userId") %>%
    mutate(movie_group = ifelse(is.na(movie_group), 99, movie_group),
           user_group = ifelse(is.na(user_group), 99, user_group)) %>%
    left_join(b_uI, by = c("userId", "movie_group")) %>%
    left_join(b_iU, by = c("movieId", "user_group")) %>%
    mutate(b_uI = ifelse(is.na(b_uI), 0, b_uI),
           b_iU = ifelse(is.na(b_iU), 0, b_iU),
           pred = mu + b_i + b_u + b_uI + b_iU) %>%
    pull(pred)
}

# calculate optimal lambda and RMSE for model 5, and add to results table
time1 <- now()
best <- optimise_lambda(predict_model_5, train_set, test_set)
time2 <- now()
print(time2 - time1)
model_5_rmse <- best$rmse
rmse_results <- bind_rows(rmse_results,
                          tibble(method="Regularised Movie + User + Movie Group + User Group Effects Model",
                                 RMSE = model_5_rmse ))
rmse_results

# RMSE has improved again - we will use this as our final model

# update mu for full edx set
mu <- mean(edx$rating)

# train model 5 on the edX set and predict values for validation set
time1 <- now()

# calculate bias terms using optimal value of lambda and the edx dataset
b_i <- calculate_b_i(edx, best$lambda)
b_u <- calculate_b_u(edx, best$lambda, b_i)
b_uI <- calculate_b_uI(edx, best$lambda, b_i, b_u, movie_groups)
b_iU <- calculate_b_iU(edx, best$lambda, b_i, b_u, b_uI, movie_groups, user_groups)

# predict ratings for validation set using the bias terms calculated with the edx set
predicted_values <- validation %>%
  left_join(b_i, by="movieId") %>%
  left_join(b_u, by="userId") %>%
  left_join(movie_groups, by = "movieId") %>%
  left_join(user_groups, by = "userId") %>%
  mutate(movie_group = ifelse(is.na(movie_group), 99, movie_group),
         user_group = ifelse(is.na(user_group), 99, user_group)) %>%
  left_join(b_uI, by = c("userId", "movie_group")) %>%
  left_join(b_iU, by = c("movieId", "user_group")) %>%
  mutate(b_uI = ifelse(is.na(b_uI), 0, b_uI),
         b_iU = ifelse(is.na(b_iU), 0, b_iU),
         pred = mu + b_i + b_u + b_uI + b_iU) %>%
  pull(pred)
time2 <- now()
time2 - time1

# took just under a minute

# check the RMSE
final_model_rmse <- RMSE(predicted_values, validation$rating)
final_model_rmse

# looks good!

# compare with model 2 (textbook approach)
time1 <- now()
optimal <- optimise_lambda(predict_model_2, train_set, test_set)
textbook_predicted_values <- predict_model_2(edx, validation, optimal$lambda)
time2 <- now()
time2 - time1

textbook_rmse <- RMSE(textbook_predicted_values, validation$rating)

# textbook RMSE is higher than our final model RMSE!

# create table of all model RMSEs 

# get optimal lambda values for each model step
optimal1 <- optimise_lambda(predict_model_1, train_set, test_set)
# model 2 RMSE already calculated
# model 3 was not included in our final model
optimal4 <- optimise_lambda(predict_model_4, train_set, test_set)
# model 5 RMSE already calculated

# make predictions using optimal lambdas on validation set
predict1 <- predict_model_1(edx, validation, optimal1$lambda)
predict4 <- predict_model_4(edx, validation, optimal4$lambda)

# calculate RMSEs
rmse0 <- RMSE(rep(mu, length(validation$rating))) # naive Bayesian model
rmse1 <- RMSE(predict1, validation$rating)
rmse2 <- textbook_rmse # already calculated
rmse4 <- RMSE(predict4, validation$rating)
rmse5 <- final_model_rmse

tibble(method=c("Just the average",
                "Regularised Movie Effects Model",
                "Regularised Movie + User Effects Model",
                "Regularised Movie + User + Movie Group Effects Model",
                "Regularised Movie + User + Movie Group + User Group Effects Model"),
                                 RMSE = c(rmse0, rmse1, rmse2, rmse4, rmse5))

