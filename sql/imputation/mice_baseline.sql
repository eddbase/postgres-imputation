CREATE OR REPLACE PROCEDURE MICE_baseline(
        input_table_name text,
        output_table_name text,
        continuous_columns text[],
        categorical_columns text[],
        continuous_columns_null text[], 
        categorical_columns_null text[],
        fillfactor integer,
        iterations integer,
        add_noise boolean

    ) LANGUAGE plpgsql AS $$
DECLARE 
    start_ts timestamptz;
    end_ts   timestamptz;
    query text;
    query2 text;
    subquery text;
    query_null1 text;
    query_null2 text;
    col_averages float8[];
    tmp_array text[];
    tmp_array2 text[];
    col_mode int4[];
    cofactor_global cofactor;
    cofactor_null cofactor;
    col text;
    params float8[];
    label_index int4;
BEGIN
    --select linregr_train(ARRAY[''], ARRAY[''], '', 1, 0.001, 0, 100000, true, false, 'SELECT SUM(to_cofactor(ARRAY[ id, s_length, s_width, p_length, p_width ]::float8[],ARRAY[ target ]::int4[])) FROM iris_impute_res WHERE NOT s_width_ISNULL;');

    -- COMPUTE COLUMN AVERAGES (over a subset)
    SELECT array_agg('AVG(' || x || ')')
    FROM unnest(continuous_columns_null) AS x
    INTO tmp_array;
    
    SELECT array_agg('MODE() WITHIN GROUP (ORDER BY ' || x || ') ')
    FROM unnest(categorical_columns_null) AS x
    INTO tmp_array2;

    query := ' SELECT ARRAY[ ' || array_to_string(tmp_array, ', ') || ' ]::float8[]' || 
             ' FROM ( SELECT ' || array_to_string(continuous_columns_null, ', ') || 
                    ' FROM ' || input_table_name || ' LIMIT 500000 ) AS t';
    query2 := ' SELECT ARRAY[ ' || array_to_string(tmp_array2, ', ') || ' ]::int[]' || 
             ' FROM ( SELECT ' || array_to_string(categorical_columns_null, ', ') || 
                    ' FROM ' || input_table_name || ' LIMIT 500000 ) AS t';

    RAISE DEBUG '%', query;

    start_ts := clock_timestamp();
    IF array_length(continuous_columns_null, 1) > 0 THEN
    EXECUTE query INTO col_averages;
    END IF;
    IF array_length(categorical_columns_null, 1) > 0 THEN
    	EXECUTE query2 INTO col_mode;
    END IF;
    end_ts := clock_timestamp();
    

    RAISE DEBUG 'AVERAGES: %', col_averages;
    RAISE INFO 'COMPUTE COLUMN AVERAGES: ms = %', 1000 * (extract(epoch FROM end_ts - start_ts));

    -- CREATE TABLE WITH MISSING VALUES
    query := 'DROP TABLE IF EXISTS ' || output_table_name;
    RAISE DEBUG '%', query;
    EXECUTE QUERY;
    
    
    SELECT (
        SELECT array_agg(x || ' float8')
        FROM unnest(continuous_columns) AS x
    ) ||
    (
        SELECT array_agg(x || ' int')
        FROM unnest(categorical_columns) AS x
    ) ||
    (
        SELECT array_agg(x || '_ISNULL bool')
        FROM unnest(continuous_columns_null) AS x
    )||
    (
        SELECT array_agg(x || '_ISNULL bool')
        FROM unnest(categorical_columns_null) AS x
    )
    INTO tmp_array;
        
    query := 'CREATE UNLOGGED TABLE ' || output_table_name || '( ' ||
                array_to_string(tmp_array, ', ') || ', ROW_ID serial) WITH (fillfactor='||fillfactor||')';
    RAISE DEBUG '%', query;
    EXECUTE QUERY;
    
    COMMIT;
          
        -- INSERT INTO TABLE WITH MISSING VALUES
    start_ts := clock_timestamp();
    SELECT (
        SELECT array_agg(
            CASE 
                WHEN array_position(continuous_columns_null, x) IS NULL THEN 
                    x
                ELSE 
                    'COALESCE(' || x || ', ' || col_averages[array_position(continuous_columns_null, x)] || ')'
            END
        )
        FROM unnest(continuous_columns) AS x
    ) || (
        SELECT array_agg(
            CASE 
                WHEN array_position(categorical_columns_null, x) IS NULL THEN 
                    x
                ELSE 
                    'COALESCE(' || x || ', ' || col_mode[array_position(categorical_columns_null, x)] || ')'
            END
        )
        FROM unnest(categorical_columns) AS x
    ) || (
        SELECT array_agg(x || ' IS NULL')
        FROM unnest(continuous_columns_null) AS x
    ) || (
        SELECT array_agg(x || ' IS NULL')
        FROM unnest(categorical_columns_null) AS x
    ) INTO tmp_array;
    
    
    
    query := 'INSERT INTO ' || output_table_name || 
             ' SELECT ' || array_to_string(tmp_array, ', ') ||
             ' FROM ' || input_table_name;
    RAISE DEBUG '%', query;

    start_ts := clock_timestamp();
    EXECUTE query;
    end_ts := clock_timestamp();
    RAISE INFO 'INSERT INTO TABLE WITH MISSING VALUES: ms = %', 1000 * (extract(epoch FROM end_ts - start_ts));
    COMMIT;

    FOR i in 1..iterations LOOP
        RAISE INFO 'Iteration %', i;

        FOREACH col in ARRAY categorical_columns_null LOOP
            RAISE INFO '  |- Column %', col;
            
            -- COMPUTE COFACTOR WHERE $col IS NOT NULL
            query := 'SELECT SUM(to_cofactor(' ||
                    'ARRAY[ ' || array_to_string(continuous_columns, ', ') || ' ]::float8[],' ||
                    'ARRAY[ ' || array_to_string(categorical_columns, ', ') || ' ]::int4[]' ||
                ')) '
            'FROM ' || output_table_name || ' '
            'WHERE NOT ' || col || '_ISNULL;';
            RAISE DEBUG '%', query;
            
                                    
            label_index := array_position(categorical_columns, col);
            RAISE DEBUG 'LABEL INDEX %', query_null1;
            start_ts := clock_timestamp();
            params := lda_train(ARRAY[''], ARRAY[''],'',label_index,0,false, query);
            end_ts := clock_timestamp();
            RAISE DEBUG '%', params;
            
            RAISE INFO 'TRAIN ms = %', 1000 * (extract(epoch FROM end_ts - start_ts));
            
            -- IMPUTE
                         
            query := 'UPDATE ' || output_table_name || 
                ' SET ' || col || ' = lda_predict(ARRAY[ ' || array_to_string(params, ', ') || ']::float8[], ' || 'ARRAY[ ' || array_to_string(continuous_columns, ', ') || ' ]::float8[], '
                || 'ARRAY[ ' || array_to_string(array_remove(categorical_columns, col), ', ') || ' ]::int[], false) WHERE ' ||col||'_ISNULL';
            RAISE DEBUG 'UPDATE QUERY: %', query;

            start_ts := clock_timestamp();
            EXECUTE query;
            end_ts := clock_timestamp();
            RAISE INFO 'IMPUTE DATA: ms = %', 1000 * (extract(epoch FROM end_ts - start_ts));
            
            COMMIT;

        END LOOP;
        
        FOREACH col in ARRAY continuous_columns_null LOOP
            RAISE INFO '  |- Column %', col;
            
            -- COMPUTE COFACTOR WHERE $col IS NOT NULL
            query := 'SELECT SUM(to_cofactor(' ||
                    'ARRAY[ ' || array_to_string(continuous_columns, ', ') || ' ]::float8[],' ||
                    'ARRAY[ ' || array_to_string(categorical_columns, ', ') || ' ]::int4[]' ||
                ')) '
            'FROM ' || output_table_name || ' '
            'WHERE NOT ' || col || '_ISNULL;';
            RAISE DEBUG '%', query;
            
            start_ts := clock_timestamp();                        
            label_index := array_position(continuous_columns, col);
            --params := linregr_train(ARRAY['aaa'], ARRAY['aaa'], '', label_index, 0.001::float, 0, 100000, true::boolean, false::boolean, query);
            
            EXECUTE query INTO STRICT cofactor_global;
            params := ridge_linear_regression(cofactor_global, label_index - 1, 0.001, 0, 10000, true, false);
            
            RAISE DEBUG '---------Cofactor: = %', cofactor_global;
            RAISE INFO 'Params: = %', params;

        	
        	end_ts := clock_timestamp();
            RAISE INFO 'Train: ms = %', 1000 * (extract(epoch FROM end_ts - start_ts));
            
            -- IMPUTE
            
            query := 'UPDATE ' || output_table_name || 
                ' SET ' || col || ' = linregr_predict(ARRAY[' || array_to_string(params, ', ') || ']::float8[], ARRAY['||array_to_string(array_remove(continuous_columns, col), ', ')||']::float8[], ARRAY['||
         			array_to_string(categorical_columns, ', ')||']::int[], '|| add_noise ||' , false) WHERE ' || col || '_ISNULL';            

            start_ts := clock_timestamp();
            EXECUTE query;
            end_ts := clock_timestamp();
            RAISE INFO 'IMPUTE DATA: ms = %', 1000 * (extract(epoch FROM end_ts - start_ts));
            
            COMMIT;

        END LOOP;
        
    END LOOP;
    
END$$;

--CALL MICE_baseline('iris_impute', 'iris_impute_res', ARRAY['id','s_length', 's_width', 'p_length', 'p_width', 'target']::text[], ARRAY[]::text[], ARRAY['s_width', 'p_length', 'p_width']::text[], ARRAY[]::text[], 75, 2, false);

