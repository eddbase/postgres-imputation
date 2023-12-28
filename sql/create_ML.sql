--SET client_min_messages TO WARNING;

LOAD :FACTML_LIBRARY;

CREATE OR REPLACE FUNCTION linregr_train_s(continuous_columns text[], categorical_columns text[], tbl text, label_index int, step_size float8, lambda float8, max_iterations int, return_variance int) RETURNS float4[] AS $$
        DECLARE
    		cofactor_g cofactor;
        BEGIN
        	EXECUTE 'SELECT to_cofactor(cont_vals float8[], cat_vals int4[]) FROM ' INTO cofactor_g;
            RETURN ridge_linear_regression(cofactor_g, label_index , step_size, lambda, max_iterations, return_variance);
        END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION lda_train_s(continuous_columns text[], categorical_columns text[], tbl text, label_index int, step_size float8, lambda float8, max_iterations int, return_variance int) RETURNS float4[] AS $$
        DECLARE
    		cofactor_g cofactor;
        BEGIN
        	EXECUTE 'SELECT to_cofactor(cont_vals float8[], cat_vals int4[]) FROM ' INTO cofactor_g;
            RETURN ridge_linear_regression(cofactor_g, label_index , step_size, lambda, max_iterations, return_variance);
        END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION qda_train_s(continuous_columns text[], categorical_columns text[], tbl text, label_index int, step_size float8, lambda float8, max_iterations int, return_variance int) RETURNS float4[] AS $$
        DECLARE
    		cofactor_g cofactor;
        BEGIN
        	EXECUTE 'SELECT to_cofactor(cont_vals float8[], cat_vals int4[]) FROM ' INTO cofactor_g;
            RETURN ridge_linear_regression(cofactor_g, label_index , step_size, lambda, max_iterations, return_variance);
        END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION nb_train_s(continuous_columns text[], categorical_columns text[], tbl text, label_index int, step_size float8, lambda float8, max_iterations int, return_variance int) RETURNS float4[] AS $$
        DECLARE
    		cofactor_g cofactor;
        BEGIN
        	EXECUTE 'SELECT to_cofactor(cont_vals float8[], cat_vals int4[]) FROM ' INTO cofactor_g;
            RETURN ridge_linear_regression(cofactor_g, label_index , step_size, lambda, max_iterations, return_variance);
        END;
$$ LANGUAGE plpgsql;