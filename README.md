This repository contains the implementation of ... based on the paper:...

This repository contains both a library for performing efficient Machine Learning and code to run MICE inside PostgreSQL 12+ (not tested on lower versions). The supported models are:

```
Linear Regression
Stochastic Linear Regression
Linear Discriminant Analysis
Quadratic Discriminant Analysis
Naive Bayes
```

At the moment, imputation is always done with Stochastic Linear Regression (continuous columns) and Linear Discriminant Analysis (categorical columns).

## Installation 

Clone this repository.

Make sure `pg_config --includedir-server` and `pg_config --pkglibdir` return the correct paths (C header files for server programming and location of dynamically loadable modules respectively), otherwise manually set PGSQL\_INCLUDE\_DIRECTORY and  PGSQL\_LIB\_DIRECTORY in CMakeLists.txt.

Compile with

```
cmake .
make
sql/create_UDFs.sh
```

## How to use ML models

You can use the new simplified interface to use models. Train functions return an array of parameters, prediction functions use the parameters to generate a prediction.

### (Stochastic) Linear Regression

A LR model can be trained with the `linregr_train` function

```
linregr_train(ARRAY['num. columns'], ARRAY['cat. columns'], 'table', label_index (from 1), learning_rate, regularization value, max iterations, compute variance)

```
Example: train over iris\_train, with columns 's\_length', 's\_width', 'p\_length', 'p\_width' numerical and class categorical. The label is s_length

```
select linregr_train(ARRAY['s_length', 's_width', 'p_length', 'p_width'], ARRAY['class'], 'iris_train', 1, 0.001, 0, 100000, false);

```

This will return an array of parameters.

You can then use the predict function to generate predictions inside SQL queries:

```
linregr_predict(params_array,cont_columns,cat_columns, add_noise)
```
You can set add_noise > 0 only if you have computed the variance in the train function.
Example:

```
select linregr_predict(params_array,ARRAY[ s_width, p_length, p_width ]::float8[],ARRAY[ target ]::int4[], 0) from iris_test
```

### LDA

### QDA

### Naive Bayes

## Imputation
