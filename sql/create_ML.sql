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
            RETURN ridge_linear_regression(cofactor_g, label_index -1 , step_size::float8, lambda::float8, max_iterations::int, return_variance::boolean, norm::boolean);
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

CREATE OR REPLACE FUNCTION nb_train(continuous_columns text[], categorical_columns text[], tbl text, label_index int, norm boolean, query text DEFAULT '') RETURNS float8[] AS $$
        DECLARE
    		aggregates nb_aggregates[];
    		labels int[];
        BEGIN
        	if query = '' then        	
                query := 'SELECT array_agg(label), array_agg(aggregate) from (select '|| categorical_columns[label_index] ||' as label, SUM(to_nb_aggregates(' ||
                    'ARRAY[ ' || array_to_string(continuous_columns, ', ') || ' ]::float8[],' ||
                    'ARRAY[ ' || array_to_string(array_remove(categorical_columns, categorical_columns[label_index]), ', ') || ' ]::int4[]' ||
                ')) as aggregate FROM '||tbl || ' GROUP BY '||categorical_columns[label_index]||') as x';
            end if;
        	EXECUTE query INTO labels, aggregates;
            RETURN naive_bayes_train(aggregates, labels);
        END;
$$ LANGUAGE plpgsql;