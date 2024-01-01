LOAD :FACTML_LIBRARY;

CREATE OR REPLACE FUNCTION linregr_train(continuous_columns text[], categorical_columns text[], tbl text, label_index int, step_size float8, lambda float8, max_iterations int, return_variance boolean, norm boolean, query text DEFAULT '') RETURNS float8[] AS $$
        DECLARE
    		cofactor_g cofactor;
        BEGIN
        	if query = '' then
                query := 'SELECT SUM(to_cofactor(' ||
                    'ARRAY[ ' || array_to_string(continuous_columns, ', ') || ' ]::float8[],' ||
                    'ARRAY[ ' || array_to_string(categorical_columns, ', ') || ' ]::int4[]' ||
                ')) FROM '||tbl;
            end if;
            
        	EXECUTE query INTO cofactor_g;
            RETURN ridge_linear_regression(cofactor_g, label_index -1 , step_size, lambda, max_iterations, return_variance, norm);
        END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION lda_train(continuous_columns text[], categorical_columns text[], tbl text, label_index int, shrinkage float8, norm boolean, query text DEFAULT '') RETURNS float8[] AS $$
        DECLARE
    		cofactor_g cofactor;
        BEGIN
        	if query = '' then
                query := 'SELECT SUM(to_cofactor(' ||
                    'ARRAY[ ' || array_to_string(continuous_columns, ', ') || ' ]::float8[],' ||
                    'ARRAY[ ' || array_to_string(categorical_columns, ', ') || ' ]::int4[]' ||
                ')) FROM '||tbl;
            end if;
        	EXECUTE query INTO cofactor_g;
            RETURN train_lda(cofactor_g, label_index -1 , shrinkage, norm);
        END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION qda_train(continuous_columns text[], categorical_columns text[], tbl text, label_index int, norm boolean, query text DEFAULT '') RETURNS float8[] AS $$
        DECLARE
    		aggregates cofactor[];
    		labels int[];
        BEGIN
        	if query = '' then        	
                query := 'SELECT array_agg(label), array_agg(aggregate) from (select '|| categorical_columns[label_index] ||' as label, SUM(to_cofactor(' ||
                    'ARRAY[ ' || array_to_string(continuous_columns, ', ') || ' ]::float8[],' ||
                    'ARRAY[ ' || array_to_string(array_remove(categorical_columns, categorical_columns[label_index]), ', ') || ' ]::int4[]' ||
                ')) as aggregate FROM '||tbl || ' GROUP BY '||categorical_columns[label_index]||') as x';
            end if;
        	EXECUTE query INTO labels, aggregates;
            RETURN train_qda(aggregates, labels, norm);
        END;
$$ LANGUAGE plpgsql;

--select linregr_train(ARRAY['s_length', 's_width', 'p_length', 'p_width'], ARRAY['target'], 'iris_train', 1, 0.001, 0, 100000, false, true)

--select linregr_train(ARRAY['dep_delay', 'taxi_out', 'taxi_in', 'arr_delay', 'air_time', 'dep_time_hour'], ARRAY['diverted', 'extra_day_arr', 'extra_day_dep'], 'flight.flight', 1, 0.001, 0, 100000, false)

--select sum(to_cofactor(ARRAY[dep_delay, taxi_out, taxi_in, arr_delay, air_time, dep_time_hour]::float8[], ARRAY[diverted, extra_day_arr, extra_day_dep]::int4[])) from flight.flight

--select linregr_predict_query(ARRAY[1,0,3,0,1,2,0.6513272625282294,0.30945684880257734,-0.11056794594381988,0.6543010155192934,1.189197041952367,-0.1259800679792555,-0.4118897114458238]::float8[], ARRAY['s_width', 'p_length', 'p_width'], ARRAY['target'], false);


--select linregr_predict(ARRAY[1,0,3,0,1,2,0.6513272625282294,0.30945684880257734,-0.11056794594381988,0.6543010155192934,1.189197041952367,-0.1259800679792555,-0.4118897114458238,0.24726508666622204]::float8[],ARRAY[ s_width, p_length, p_width ]::float8[],ARRAY[ target ]::int4[]) from iris_test limit 1;
--select linregr_predict(ARRAY[1,0,3,0,1,2,0.6513272625282294,0.30945684880257734,-0.11056794594381988,0.6543010155192934,1.189197041952367,-0.1259800679792555,-0.4118897114458238,0.24726508666622204]::float8[],ARRAY[ s_width, p_length, p_width ]::float8[],ARRAY[ target ]::int4[], 1) from iris_train limit 1;

--select linregr_predict(ARRAY[0,0,0]::float8[],ARRAY[l_quantity]::float8[],ARRAY[]::int4[], false) from lineitem;


----select linregr_predict(ARRAY[1,0,3,0,1,2,5.814999999999999,0.4716438332344789,0.8883556584817722,-0.4997642915453853,0.5811198486524791,-0.1426024429639806,-0.38143964774755496,3.011000000000001,3.8160000000000003,1.219,0.31,0.35,0.34]::float8[],ARRAY[ s_width, p_length, p_width ]::float8[],ARRAY[ target ]::int4[], 0, true) from iris_test limit 1;


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