--SET client_min_messages TO WARNING;

LOAD :FACTML_LIBRARY;

CREATE OR REPLACE FUNCTION ridge_linear_regression(
        c cofactor, 
        label_idx int, 
        step_size float8, 
        lambda float8, 
        max_iterations int,
        return_variance boolean, norm boolean)
    RETURNS float8[]
    AS :FACTML_LIBRARY, 'ridge_linear_regression'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE OR REPLACE FUNCTION linregr_predict(
         train_data float8[],
         feats_numerical float8[],
         feats_categorical int[],
         add_noise boolean,
         norm boolean
    )
    RETURNS float8
    AS :FACTML_LIBRARY, 'linregr_predict'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE OR REPLACE FUNCTION ridge_linear_regression_from_params(
        c float4[],
        label_idx int,
        step_size float8,
        lambda float8,
        max_iterations int
    )
    RETURNS float8[]
    AS :FACTML_LIBRARY, 'ridge_linear_regression_from_params'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

 CREATE OR REPLACE FUNCTION train_lda(
         c cofactor,
         label int,
         shrinkage float8,
         norm boolean
     )
     RETURNS float8[]
     AS :FACTML_LIBRARY, 'train_lda'
     LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;


 CREATE OR REPLACE FUNCTION lda_predict(
         train_data float8[],
         feats_numerical float8[],
         feats_categorical int[],
         norm boolean
     )
     RETURNS int
     AS :FACTML_LIBRARY, 'lda_predict'
     LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;
     
     
 CREATE OR REPLACE FUNCTION naive_bayes_train(
         aggregates nb_aggregates[]
     )
     RETURNS float8[]
     AS :FACTML_LIBRARY, 'naive_bayes_train'
     LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;
     
 CREATE OR REPLACE FUNCTION naive_bayes_predict(
         params float8[],
         cont_feats float8[],
         cat_feats int[]
     )
     RETURNS int
     AS :FACTML_LIBRARY, 'naive_bayes_predict'
     LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;
     
 CREATE OR REPLACE FUNCTION train_qda(
         aggregates cofactor[],
         labels int[],
         norm boolean
     )
     RETURNS float8[]
     AS :FACTML_LIBRARY, 'train_qda'
     LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;
     
 CREATE OR REPLACE FUNCTION qda_predict(
         params float8[],
         cont_feats float8[],
         cat_feats int[],
         norm boolean
     )
     RETURNS int
     AS :FACTML_LIBRARY, 'qda_predict'
     LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;
     