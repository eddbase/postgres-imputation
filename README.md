This repository contains the implementation of ... based on the paper:...

This repository contains both a library for performing efficient Machine Learning and code to run MICE inside PostgreSQL 12+ (not tested on lower versions). The supported models are:

```
Linear Regression
Stochastic Linear Regression
Linear Discriminant Analysis
Quadratic Discriminant Analysis
Naive Bayes (Gaussian and categorical)
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
linregr_train(
	num. columns text[],
	cat. columns text[], 
	table name text,
	label_index (from 1) int,
	learning_rate float8,
	regularization float8,
	max_iterations integer,
	compute_variance boolean)

```
Example: train over iris\_train, with columns 's\_length', 's\_width', 'p\_length', 'p\_width' numerical and class categorical. The label is s_length

```
train_params := select linregr_train(
	ARRAY['s_length', 's_width', 'p_length', 'p_width'], 
	ARRAY['class'],
	'iris_train', 1, 0.001, 0, 100000, false);

```

This will return an array of parameters (float8).

You can then use the predict function to generate predictions inside SQL queries:

```
linregr_predict(
         params_array float8[],
         cont_columns float8[],
         cat_columns int[],
         add_noise boolean,
         norm boolean)
```
You can set add_noise to true only if you have computed the variance in the train function.

Example:

```
select linregr_predict(
	train_params, ARRAY[ s_width, p_length, p_width ]::float8[],
	ARRAY[ target ]::int4[],
	0)
from iris_test
```

### LDA

A LDA model can be trained with the `lda_train` function

```
lda_train(
	continuous_columns text[],
	categorical_columns text[],
	table text,
	label_index int,
	shrinkage float8,
	normalize boolean)

```

This will return an array of parameters.

You can then use the predict function to generate predictions inside SQL queries:

```
lda_predict(
         params_array float8[],
         cont_columns float8[],
         cat_columns int[],
         norm boolean)
```

### QDA

A QDA model can be trained with the `qda_train` function

```
qda_train(
	continuous_columns text[],
	categorical_columns text[],
	table text,
	label_index int,
	normalize boolean)

```

This will return an array of parameters.

You can then use the predict function to generate predictions inside SQL queries:

```
qda_predict(
         params_array float8[],
         cont_columns float8[],
         cat_columns int[],
         norm boolean)
```

### Naive Bayes

A Naive Bayes model can be trained with the `nb_train` function

```
nb_train(
	continuous_columns text[],
	categorical_columns text[],
	table text,
	label_index int,
	normalize boolean)

```

This will return an array of parameters.

You can then use the predict function to generate predictions inside SQL queries:

```
qda_predict(
         params_array float8[],
         cont_columns float8[],
         cat_columns int[],
         norm boolean)
```

## Train over multiple tables

You can train a model over multiple tables in two ways:

* Compute the join and then train the model: use a join query instead of specifying a table name in the previous functions. Example:

```
   train_params := select linregr_train(
	ARRAY['con. columns'], ARRAY['cat. columns'],
	'(SELECT con. columns, cat. columns
	FROM A JOIN B ...) as t1',
	1, 0.001, 0, 100000, false);

```
* Train without joining the tables: The previous functions accept an extra query string at the end. You need to write a query which computes a cofactor matrix over the relations. For Linear Regression and LDA, just return a single cofactor matrix. Example of a possible query is:

```
select linregr_train(
	ARRAY[''], ARRAY[''],'', 1, 0.001, 0, 100000,
	false, 
	'SELECT SUM(t1.value*t2.value)
  FROM (SELECT join_key_1, SUM(to_cofactor(
       ARRAY[num_cols],ARRAY[cat_cols])) as value
  		FROM A GROUP BY join_key_1) as t1
  		
  		JOIN (SELECT join_key_2, SUM(to_cofactor(
       ARRAY[num_cols],ARRAY[cat_cols])) as value
  		FROM B GROUP BY join_key_2) as t2
  	ON t1.join_key_1 = t2.join_key_2'
)
```

For QDA / Naive Bayes the query need to generate a cofactor matrix for each label, and both the cofactor matrices and labels need to be inserted in arrays. 

### Quickstart
`quickstart.py` will automatically compare the results of SKLearn and our library over iris for Linear Regression, LDA, QDA and Naive Bayes. Requires SKLearn, Pandas and psycopg2. Defaults PostgreSQL parameters are

```
    "host"      : "localhost",
    "database"  : "postgres",
    "user"      : "postgres",
    "password"  : ""
```

Edit param_dic in the script if you are using different parameters. The output should be:

```
SKLearn R2:  0.8606
Postgres R2:  0.8606
Accuracy SKLearn LDA:  0.98
Accuracy PostgreSQL LDA:  0.98
Accuracy QDA SKLearn  0.98
Accuracy PostgreSQL QDA:  0.98
Accuracy SKLearn Gaussian NB  0.96
Accuracy PostgreSQL NB:  0.96
```

## Imputation
