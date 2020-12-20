library(keras)

# Hyperparameter flags --------------------------------------------------------

FLAGS <- flags(
  flag_integer('kernel_size1', 5),
  flag_integer('strides1', 1)
)

# Define Model ----------------------------------------------------------------

model <- keras_model_sequential() %>%
  layer_embedding(input_dim = max_words + 1, output_dim = 16,
                  input_length = max_length) %>%
  layer_conv_1d(filter = 32, 
                kernel_size = FLAGS$kernel_size1,
                strides = FLAGS$strides1,
                activation = "relu") %>%
  layer_global_max_pooling_1d() %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")

model %>% compile(
  optimizer = "adam",
  loss = "binary_crossentropy",
  metrics = c("accuracy")
)

# Training & Evaluation ----------------------------------------------------

history <- model %>% fit(
  x = prepped_training, 
  y = kickstarter_train$state,
  batch_size = 512,
  epochs = 10,
  validation_split = 0.2
)

plot(history)

score <- model %>% evaluate(
  bake(prepped_recipe, kickstarter_test, composition = "matrix"),
  kickstarter_test$state
)

cat('Test accuracy:', score$acc, '\n')
