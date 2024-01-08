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
```

Then, if you don't need the imputation part, install with

```
sql/create_UDFs.sh <database name>
```

Otherwise use

```
sql/create_UDFs.sh <database name> imputation
```

## How to use ML models

You can use the simplified interface to use models. Train functions return an array of parameters (float8), prediction functions use the parameters to generate a prediction.

All train functions accept these parameters:

* num_columns: array of strings with the name of numerical features
* cat_columns: array of strings with the name of numerical features
* table_name: string with the name of the table (or join query, more on this later)
* label\_index: The index (starting from 1) of the label (which feature is the label). The label must be inside num_columns (Linear Regression) or cat\_columns (classifiers)

Certain models might require additional parameters

All predict functions accept the following parameters:

* num_columns: array of strings with the name of numerical features
* cat_columns: array of strings with the name of numerical features
* Table name: string with the name of the table

They will return a float8 (regression) or integer (classifiers).

Make sure the order of parameters is the same inside train and predict functions (of course do not include the label in the predict function). Certain models might require additional parameters.

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

There are 3 variation of MICE implemented: `MICE_baseline`, `MICE_low` and `MICE_high`. They accept the following parameters:

```
MICE_...(
        input_table_name text,
        output_table_name text,
        continuous_columns text[],
        categorical_columns text[],
        continuous_columns_null text[], 
        categorical_columns_null text[],
        fillfactor integer,
        iterations integer,
        add_noise boolean
    )
```

* input\_table\_name: table with missing values
* output\_table\_name: name of the resulting table with imputed values
* continuous\_columns: continuous columns in the model
* categorical\_columns: categorical columns in the model
* continuous\_columns\_null: continuous columns with missing values
* categorical\_columns\_null: categorical columns with missing values
* fillfactor fillfactor of the output table. Between 0 and 100. Usually inverse proportion of the tuples with missing values.
* iterations MICE iterations
* add_noise if true uses stochastic linear regression, otherwise linear regression

### Imputation Quickstart

`quickstart_imputation.py ` will automatically test the three imputation implementations against SKLearn. It will remove 20% of the values in three columns of the iris dataset, copy the table in PostgreSQL and compare SKLearn IterativeImputer agains our three implementations.

Requires numpy, pandas, SKLearn and psycopg2. Expected output:

```
CALL MICE_high('iris_impute', 'iris_impute_res', ARRAY['id','s_length', 's_width', 'p_length', 'p_width', 'target']::text[], ARRAY[]::text[], ARRAY['s_width', 'p_length', 'p_width']::text[], ARRAY[]::text[], 75, 2, false);
MSE between SKLearn and PostgreSQL in col:  s_width  :  5.339167815511109e-10
MSE between SKLearn and PostgreSQL in col:  p_length  :  8.304800608420666e-11
MSE between SKLearn and PostgreSQL in col:  p_width  :  9.189242077255187e-11
CALL MICE_low('iris_impute', 'iris_impute_res', ARRAY['id','s_length', 's_width', 'p_length', 'p_width', 'target']::text[], ARRAY[]::text[], ARRAY['s_width', 'p_length', 'p_width']::text[], ARRAY[]::text[], 75, 2, false);
MSE between SKLearn and PostgreSQL in col:  s_width  :  6.254319451641791e-05
MSE between SKLearn and PostgreSQL in col:  p_length  :  6.25428759426222e-05
MSE between SKLearn and PostgreSQL in col:  p_width  :  6.254286222810589e-05
CALL MICE_baseline('iris_impute', 'iris_impute_res', ARRAY['id','s_length', 's_width', 'p_length', 'p_width', 'target']::text[], ARRAY[]::text[], ARRAY['s_width', 'p_length', 'p_width']::text[], ARRAY[]::text[], 75, 2, false);
MSE between SKLearn and PostgreSQL in col:  s_width  :  6.398214097905294e-10
MSE between SKLearn and PostgreSQL in col:  p_length  :  4.746474435237377e-11
MSE between SKLearn and PostgreSQL in col:  p_width  :  1.071920705582795e-10
```