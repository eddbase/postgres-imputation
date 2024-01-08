--"reverse" partitioning and "baseline" algorithm


CREATE OR REPLACE PROCEDURE create_missing_partition_high(
			params_models float8[],
			params_size int[],
			continuous_columns text[],
        	categorical_columns text[],
        	continuous_columns_null text[], 
        	categorical_columns_null text[],
        	tmp_columns_names text[],
        	old_table_name text,
        	new_table_name text,
        	add_noise boolean
			) LANGUAGE plpgsql AS $$
DECLARE
	n_models int;
	query text := '';
	tmp_array3 text[];
	tmp_array2 text[];
	tmp_array text[];
	params float8[];
	n_model int;
	curr_param_index int;
	label_index int;
	columns_lower text[];
	subquery text;
	col text;
BEGIN
	n_models := array_length(params_size, 1);
	n_model := 1;
	curr_param_index := 1;
	query := old_table_name;
	
	SELECT array_agg(LOWER(x)) FROM unnest(categorical_columns) as x INTO columns_lower;
	
	FOREACH col in ARRAY categorical_columns_null LOOP--categorical imputation
		params := params_models[curr_param_index : curr_param_index + params_size[n_model]-1];
		curr_param_index := curr_param_index + params_size[n_model];
				
		label_index := array_position(categorical_columns, col);
		
		--todo fix array parameters
		
	    SELECT array_agg(
            CASE WHEN array_position(columns_lower, LOWER(x)) = label_index THEN            
            'lda_predict(ARRAY[ ' || array_to_string(params, ', ') || ']::float8[], ' || 'ARRAY[ ' || array_to_string(continuous_columns, ', ') || ' ]::float8[], '
            || 'ARRAY[ ' || array_to_string(array_remove(categorical_columns, col), ', ') || ' ]::int[], false) ' || ' AS ' || x
            ELSE
            	x || ' AS ' || x
            END
            )
        FROM unnest(tmp_columns_names) AS x
        INTO tmp_array3;
        
        IF n_model = 1 THEN
        	query := ' SELECT ' || array_to_string(tmp_array3, ' , ') ||' FROM ' || query || ' AS ' || col;
        ELSE
        	query := ' SELECT ' || array_to_string(tmp_array3, ' , ') ||' FROM (' || query || ') AS ' || col;
        END IF;
        n_model := n_model + 1;
        
    END LOOP;
    
    
    SELECT array_agg(LOWER(x)) FROM unnest(continuous_columns) as x INTO columns_lower;
    
    FOREACH col in ARRAY continuous_columns_null LOOP--numerical imputation
    
    	params := params_models[curr_param_index : curr_param_index + params_size[n_model]-1];
		curr_param_index := curr_param_index + params_size[n_model];
		    
    	label_index := array_position(continuous_columns, col);
    	    	
        
        SELECT array_agg(
        CASE WHEN array_position(columns_lower, LOWER(x)) = label_index THEN
                'linregr_predict(ARRAY[' || array_to_string(params, ', ') || ']::float8[], ARRAY['||array_to_string(array_remove(continuous_columns, col), ', ')||']::float8[], ARRAY['||
         		array_to_string(categorical_columns, ', ')||']::int[], '|| add_noise ||', false)' || ' AS ' || x        
        ELSE
            x || ' AS ' || x
        END
        )
        FROM unnest(tmp_columns_names) AS x
        INTO tmp_array3;
                
        IF n_model = 1 THEN
        	query := ' SELECT ' || array_to_string(tmp_array3, ' , ') ||' FROM ' || query || ' AS ' || col;
        ELSE
        	query := ' SELECT ' || array_to_string(tmp_array3, ' , ') ||' FROM (' || query || ') AS ' || col;
        END IF;
        n_model := n_model + 1;
        
    
    END LOOP;
    RAISE INFO 'Query = %', query;
    EXECUTE 'CREATE UNLOGGED TABLE '||new_table_name||' AS '||query;
    
END$$;


CREATE OR REPLACE PROCEDURE MICE_high(
        input_table_name text,
        output_table_name text,
        continuous_columns text[],
        categorical_columns text[],
        continuous_columns_null text[], 
        categorical_columns_null text[],
        fillfactor float,
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
    col_mode int4[];
    tmp_array text[];
    tmp_array2 text[];
    tmp_array3 text[];
    columns_lower text[];
    cofactor_global cofactor;
    cofactor_fixed cofactor;
    cofactor_tmp cofactor;
    col text;
    col2 text;
    params float8[];
    model_params float8[];
    model_params_size int[];
    tmp_columns_names text[];
    label_index int4;
    max_range int4;    
    low_bound_categorical  int[];
    max_nulls int;

BEGIN
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
        SELECT array_agg(x || '_ISNOTNULL bool')
        FROM unnest(continuous_columns_null) AS x
    )||
    (
        SELECT array_agg(x || '_ISNOTNULL bool')
        FROM unnest(categorical_columns_null) AS x
    )
     || (
        ARRAY [ 'NOT_NULL_CNT int', 'NOT_NULL_COL_ID int', 'ROW_ID serial' ]
    )
    INTO tmp_array;
    
    query := 'CREATE UNLOGGED TABLE ' || output_table_name || '( ' ||
                array_to_string(tmp_array, ', ') || ') PARTITION BY RANGE (NOT_NULL_CNT)';
    RAISE DEBUG '%', query;
    EXECUTE QUERY;

    query := 'CREATE UNLOGGED TABLE ' || output_table_name || '_notnullcnt0' || 
             ' PARTITION OF ' || output_table_name || 
             ' FOR VALUES FROM (0) TO (1) WITH (fillfactor=100)';
    RAISE DEBUG '%', query;--all null (0 not null)
    EXECUTE QUERY;

    query := 'CREATE UNLOGGED TABLE ' || output_table_name || '_notnullcnt1' || 
             ' PARTITION OF ' || output_table_name || 
             ' FOR VALUES FROM (1) TO (2) PARTITION BY RANGE (NOT_NULL_COL_ID)';
    RAISE DEBUG '%', query;
    EXECUTE QUERY;--1 not null
    
    if array_length(categorical_columns_null, 1) > 0 then
        max_range := array_length(continuous_columns_null, 1) + array_length(categorical_columns_null, 1);
    else
        max_range := array_length(continuous_columns_null, 1);
    end if;
    
    
    FOR col_id in 1..max_range LOOP
        query := 'CREATE UNLOGGED TABLE ' || output_table_name || '_notnullcnt1_col' || col_id || 
                 ' PARTITION OF ' || output_table_name || '_notnullcnt1' || 
                 ' FOR VALUES FROM (' || col_id || ') TO (' || col_id + 1 || ') WITH (fillfactor=100)';
        RAISE DEBUG '%', query;
        EXECUTE QUERY;
    END LOOP;--1 not null
    
    IF max_range >= 2 THEN
    	    if array_length(categorical_columns_null, 1) > 0 then
    	    	max_nulls := array_length(continuous_columns_null, 1) + array_length(categorical_columns_null, 1);    			
            else
             	max_nulls := array_length(continuous_columns_null, 1);             	
    		end if;
    		
    		query := 'CREATE UNLOGGED TABLE ' || output_table_name || '_notnullcnt2' || 
             	' PARTITION OF ' || output_table_name ||
             	' FOR VALUES FROM (2) TO (' || max_nulls || ') WITH (fillfactor='||fillfactor||')';
             	
             RAISE DEBUG '%', query;
    		EXECUTE query;
    			
    		query := 'CREATE UNLOGGED TABLE ' || output_table_name || '_notnullcnt3' || 
        	' PARTITION OF ' || output_table_name ||
            ' FOR VALUES FROM (' || max_nulls || ') TO (' || max_nulls + 1 ||') WITH (fillfactor=100)';
            RAISE DEBUG '%', query;
    		EXECUTE query;

    		
    END IF;
    
    
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
        SELECT array_agg(x || ' IS NOT NULL')
        FROM unnest(continuous_columns_null) AS x
    ) || (
        SELECT array_agg(x || ' IS NOT NULL')
        FROM unnest(categorical_columns_null) AS x
    ) || ( 
        SELECT ARRAY [ array_to_string(array_agg('(CASE WHEN ' || x || ' IS NOT NULL THEN 1 ELSE 0 END)'), ' + ') ]
        FROM unnest(continuous_columns_null || categorical_columns_null) AS x
    ) INTO tmp_array;
    
    SELECT (
        SELECT array_agg(
        		CASE 
        			WHEN array_position(continuous_columns_null, x) IS NULL THEN
        				' WHEN ' || x || ' IS NOT NULL THEN ' || array_position (categorical_columns_null, x) + array_length(continuous_columns_null, 1)
                    ELSE 
                    	' WHEN ' || x || ' IS NOT NULL THEN ' || array_position (continuous_columns_null, x)
                END
        )         
        FROM unnest(continuous_columns_null || categorical_columns_null) AS x
    )
    INTO tmp_array2;    
    
    query := 'INSERT INTO ' || output_table_name || 
             ' SELECT ' || array_to_string(tmp_array, ', ') || ', CASE ' || array_to_string(tmp_array2, '  ') || ' ELSE 0 END ' ||
             ' FROM ' || input_table_name;
    RAISE DEBUG '%', query;

    EXECUTE query;
    end_ts := clock_timestamp();
    RAISE INFO 'INSERT INTO TABLE WITH MISSING VALUES: ms = %', 1000 * (extract(epoch FROM end_ts - start_ts));   
    
    -- COMPUTE MAIN COFACTOR
    query := 'SELECT SUM(to_cofactor(' ||
                    'ARRAY[ ' || array_to_string(continuous_columns, ', ') || ' ]::float8[],' ||
                    'ARRAY[ ' || array_to_string(categorical_columns, ', ') || ' ]::int4[]' ||
                ')) '
            'FROM ' || output_table_name || ' WHERE NOT_NULL_CNT = '|| max_nulls;
    RAISE DEBUG '%', query;
    
    start_ts := clock_timestamp();
    EXECUTE query INTO STRICT cofactor_fixed;
    end_ts := clock_timestamp();
    RAISE INFO 'shared cofactor ms = %', 1000 * (extract(epoch FROM end_ts - start_ts));


    COMMIT;
    
    FOR i in 1..iterations LOOP
    
    	model_params := ARRAY[]::float8[];
    	model_params_size := ARRAY[]::int[];
    
        RAISE INFO 'Iteration %', i;
        
        FOREACH col in ARRAY categorical_columns_null LOOP --categorical imputation
            RAISE INFO '  |- Column %', col;
            
            cofactor_global := cofactor_fixed;
            
            -- COMPUTE COFACTOR WHERE $col IS NULL
            query_null1 := 'SELECT SUM(to_cofactor(' ||
                    'ARRAY[ ' || array_to_string(continuous_columns, ', ') || ' ]::float8[],' ||
                    'ARRAY[ ' || array_to_string(categorical_columns, ', ') || ' ]::int4[]' ||
                ')) '
            'FROM ' || output_table_name || ' '
            'WHERE (NOT_NULL_CNT = 1 AND NOT_NULL_COL_ID = ' || array_position(categorical_columns_null, col) + array_length(continuous_columns_null, 1) || ');';
            RAISE DEBUG '%', query_null1;
            
            start_ts := clock_timestamp();
            EXECUTE query_null1 INTO STRICT cofactor_tmp;
            
            IF cofactor_tmp IS NOT NULL THEN
            	cofactor_global := cofactor_tmp + cofactor_global;
            END IF;
            
            end_ts := clock_timestamp();
            RAISE INFO 'COFACTOR NULL (1): ms = %', 1000 * (extract(epoch FROM end_ts - start_ts));
            -- RAISE DEBUG 'COFACTOR NULL = %', cofactor_null;
            
            --RAISE DEBUG 'COFACTOR GLOBAL = %', cofactor_global;
            
            query_null2 := 'SELECT SUM(to_cofactor(' ||
                    'ARRAY[ ' || array_to_string(continuous_columns, ', ') || ' ]::float8[],' ||
                    'ARRAY[ ' || array_to_string(categorical_columns, ', ') || ' ]::int4[]' ||
                ')) '
            'FROM ' || output_table_name || ' '
            ' WHERE NOT_NULL_CNT >= 2 AND NOT_NULL_CNT <'|| max_nulls || ' AND ' || col || '_ISNOTNULL';
            RAISE DEBUG '%', query_null2;
            
            start_ts := clock_timestamp();
            EXECUTE query_null2 INTO STRICT cofactor_tmp;
            end_ts := clock_timestamp();
            RAISE INFO 'COFACTOR NULL (2): ms = %', 1000 * (extract(epoch FROM end_ts - start_ts));
            -- RAISE DEBUG 'COFACTOR NULL = %', cofactor_null;

            IF cofactor_tmp IS NOT NULL THEN
                cofactor_global := cofactor_global + cofactor_tmp;
            END IF;
            
            RAISE DEBUG 'COFACTOR GLOBAL = %', cofactor_global;

            -- TRAIN
            
            label_index := array_position(categorical_columns, col);
            RAISE DEBUG 'LABEL INDEX %', query_null1;
            start_ts := clock_timestamp();
            params := train_lda(cofactor_global, label_index -1 , 0, false);
            
            model_params := model_params || params;
            model_params_size := array_append(model_params_size, array_length(params, 1));
            
            end_ts := clock_timestamp();
            RAISE INFO 'TRAIN ms = %', 1000 * (extract(epoch FROM end_ts - start_ts));
            
            -- IMPUTE
            start_ts := clock_timestamp();
            
                                    
            query := 'SELECT array_agg(quote_ident(column_name)) FROM information_schema.columns WHERE table_name = ''' || output_table_name || ''';';
            EXECUTE query INTO tmp_columns_names;
            RAISE DEBUG '%', tmp_columns_names;
            
            SELECT array_agg(LOWER(x)) FROM unnest(categorical_columns) as x INTO columns_lower;
            
            --need to recreate all the =1 subpartitions
            start_ts := clock_timestamp();
            FOREACH col2 in ARRAY continuous_columns_null || categorical_columns_null LOOP
            	continue when col2 = col;
            	
            	query := 'ALTER TABLE '||output_table_name || '_notnullcnt1 DETACH PARTITION '|| output_table_name ||'_notnullcnt1_col'||array_position(continuous_columns_null || categorical_columns_null, col2);
            	RAISE DEBUG '%', query;
            	EXECUTE query;

            	query := 'ALTER TABLE '||output_table_name ||'_notnullcnt1_col'||array_position(continuous_columns_null || categorical_columns_null, col2)  ||' RENAME TO tmp_table';

            	RAISE DEBUG '%', query;
            	EXECUTE query;
                        
            
            	SELECT array_agg(
            	CASE WHEN array_position(columns_lower, LOWER(x)) = label_index THEN
            	    'lda_predict(ARRAY[ ' || array_to_string(params, ', ') || ']::float8[], ' || 'ARRAY[ ' || array_to_string(continuous_columns, ', ') || ' ]::float8[], '
                || 'ARRAY[ ' || array_to_string(array_remove(categorical_columns, col), ', ') || ' ]::int[], false) ' || ' AS ' || x
            	ELSE
            	    x || ' AS ' || x
            	END
            	)
            	FROM unnest(tmp_columns_names) AS x
            	INTO tmp_array3;
            
            	RAISE DEBUG '%', categorical_columns;
            	RAISE DEBUG '%', label_index;                   
                      
            
            	query := 'CREATE UNLOGGED TABLE ' || output_table_name || '_notnullcnt1_col'||array_position(continuous_columns_null || categorical_columns_null, col2)  || ' AS SELECT ';
            	query := query || array_to_string(tmp_array3, ' , ') ||' FROM tmp_table';
                    RAISE DEBUG '%', query;
            	EXECUTE query;
        		query := 'ALTER TABLE '|| output_table_name || '_notnullcnt1_col'||array_position(continuous_columns_null || categorical_columns_null, col2)  ||' ALTER COLUMN row_id SET NOT NULL';
        	    RAISE DEBUG '%', query;
                EXECUTE query;
            	query := 'ALTER TABLE '|| output_table_name || '_notnullcnt1 ATTACH PARTITION '||output_table_name || '_notnullcnt1_col'||array_position(continuous_columns_null || categorical_columns_null, col2)  ||' FOR VALUES FROM (' || array_position(continuous_columns_null || categorical_columns_null, col2)  || ') TO (' || array_position(continuous_columns_null || categorical_columns_null, col2)  + 1 || ')';
                RAISE DEBUG '%', query;
                EXECUTE query;
                EXECUTE 'DROP TABLE tmp_table';
            ----            	
            END LOOP;
            end_ts := clock_timestamp();
            RAISE INFO 'IMPUTE DATA (1): ms = %', 1000 * (extract(epoch FROM end_ts - start_ts));

            ------------------------------------
                        
            start_ts := clock_timestamp();
            
            ---remove this or the other one
            
            query := 'UPDATE ' || output_table_name || 
                ' SET ' || col || ' = lda_predict(ARRAY[ ' || array_to_string(params, ', ') || ']::float8[], ' || 'ARRAY[ ' || array_to_string(continuous_columns, ', ') || ' ]::float8[], ARRAY['
                || array_to_string(array_remove(categorical_columns, col), ', ')  || ']::int[], false) ' ||
                ' WHERE NOT_NULL_CNT >= 2 AND NOT_NULL_CNT <'|| max_nulls || ' AND ' || col || '_ISNOTNULL IS NOT TRUE;';
            
            EXECUTE query;

            end_ts := clock_timestamp();

            
            RAISE INFO 'IMPUTE DATA (2): ms = %', 1000 * (extract(epoch FROM end_ts - start_ts));
            
            ---------------------------------
                        
            COMMIT;
            
        END LOOP;
        
        FOREACH col in ARRAY continuous_columns_null LOOP
            RAISE INFO '  |- Column %', col;
            
            cofactor_global := cofactor_fixed;
            
            start_ts := clock_timestamp();
            
            -- COMPUTE COFACTOR WHERE $col IS NULL
            query_null1 := 'SELECT SUM(to_cofactor(' ||
                    'ARRAY[ ' || array_to_string(continuous_columns, ', ') || ' ]::float8[],' ||
                    'ARRAY[ ' || array_to_string(categorical_columns, ', ') || ' ]::int4[]' ||
                ')) '
            'FROM ' || output_table_name || ' '
            'WHERE (NOT_NULL_CNT = 1 AND NOT_NULL_COL_ID = ' || array_position(continuous_columns_null, col) || ');';
            RAISE DEBUG '%', query_null1;
            
            EXECUTE query_null1 INTO STRICT cofactor_tmp;
            
            IF cofactor_tmp IS NOT NULL THEN
            	cofactor_global := cofactor_tmp + cofactor_global;
            END IF;
                        
            end_ts := clock_timestamp();
            RAISE INFO 'COFACTOR NULL (1): ms = %', 1000 * (extract(epoch FROM end_ts - start_ts));
            
            query_null2 := 'SELECT SUM(to_cofactor(' ||
                    'ARRAY[ ' || array_to_string(continuous_columns, ', ') || ' ]::float8[],' ||
                    'ARRAY[ ' || array_to_string(categorical_columns, ', ') || ' ]::int4[]' ||
                ')) '
            'FROM ' || output_table_name || ' '
            ' WHERE NOT_NULL_CNT >= 2 AND NOT_NULL_CNT <'|| max_nulls || ' AND ' || col || '_ISNOTNULL;';
            RAISE DEBUG '%', query_null2;
            
            start_ts := clock_timestamp();
            
            EXECUTE query_null2 INTO STRICT cofactor_tmp;
            
            IF cofactor_tmp IS NOT NULL THEN
                cofactor_global := cofactor_global + cofactor_tmp;
            END IF;
            
            end_ts := clock_timestamp();
            RAISE INFO 'COFACTOR NULL (2): ms = %', 1000 * (extract(epoch FROM end_ts - start_ts));
            
            -- TRAIN
            label_index := array_position(continuous_columns, col);
            start_ts := clock_timestamp();
            
            RAISE DEBUG '-----------------Cofactor: = %', cofactor_global;
            params := ridge_linear_regression(cofactor_global, label_index - 1, 0.001, 0, 10000, true, false);
            RAISE DEBUG 'PARAMS%', params;

            model_params := model_params || params;
            model_params_size := array_append(model_params_size, array_length(params, 1));
            end_ts := clock_timestamp();
            RAISE INFO 'TRAIN: ms = %', 1000 * (extract(epoch FROM end_ts - start_ts));

            -- IMPUTE
            
            start_ts := clock_timestamp();
            
            query := 'SELECT array_agg(quote_ident(column_name)) FROM information_schema.columns WHERE table_name = ''' || output_table_name || ''';';
            EXECUTE query INTO tmp_columns_names;
            RAISE DEBUG '%', tmp_columns_names;
            
            
            FOREACH col2 in ARRAY continuous_columns_null || categorical_columns_null LOOP
            	continue when col2 = col;
            	
            	query := 'ALTER TABLE '||output_table_name || '_notnullcnt1 DETACH PARTITION '|| output_table_name ||'_notnullcnt1_col'||array_position(continuous_columns_null || categorical_columns_null, col2);
            	RAISE DEBUG '%', query;
            	EXECUTE query;

            	query := 'ALTER TABLE '||output_table_name ||'_notnullcnt1_col'||array_position(continuous_columns_null || categorical_columns_null, col2)  ||' RENAME TO tmp_table';

            	RAISE DEBUG '%', query;
            	EXECUTE query;
                        
                        
            	--- 'normal_rand(1, 0, ' || sqrt(params[array_length(params, 1)])::text || ')'
            	SELECT array_agg(LOWER(x)) FROM unnest(continuous_columns) as x INTO columns_lower;

            
            	SELECT array_agg(
                CASE WHEN array_position(columns_lower, LOWER(x)) = label_index THEN
                    'linregr_predict(ARRAY[' || array_to_string(params, ', ') || ']::float8[], ARRAY['||array_to_string(array_remove(continuous_columns, col), ', ')||']::float8[], ARRAY['||
         			array_to_string(categorical_columns, ', ')||']::int[], '|| add_noise ||', false)' || ' AS ' || x
                ELSE
                    x || ' AS ' || x
                END
                )
            	FROM unnest(tmp_columns_names) AS x
            	INTO tmp_array3;
            
            	RAISE DEBUG '%', categorical_columns;
            	RAISE DEBUG '%', label_index;                        
                      
            
            	query := 'CREATE UNLOGGED TABLE ' || output_table_name || '_notnullcnt1_col'||array_position(continuous_columns_null || categorical_columns_null, col2)  || ' AS SELECT ';
            	query := query || array_to_string(tmp_array3, ' , ') ||' FROM tmp_table';
                    RAISE DEBUG '%', query;
            	EXECUTE query;
        		query := 'ALTER TABLE '|| output_table_name || '_notnullcnt1_col'||array_position(continuous_columns_null || categorical_columns_null, col2)  ||' ALTER COLUMN row_id SET NOT NULL';
        	    RAISE DEBUG '%', query;
                EXECUTE query;
            	query := 'ALTER TABLE '|| output_table_name || '_notnullcnt1 ATTACH PARTITION '||output_table_name || '_notnullcnt1_col'||array_position(continuous_columns_null || categorical_columns_null, col2)  ||' FOR VALUES FROM (' || array_position(continuous_columns_null || categorical_columns_null, col2)  || ') TO (' || array_position(continuous_columns_null || categorical_columns_null, col2)  + 1 || ')';
                RAISE DEBUG '%', query;
                EXECUTE query;
                EXECUTE 'DROP TABLE tmp_table';
            ----
            	
            END LOOP;
            
            end_ts := clock_timestamp();
            RAISE INFO 'IMPUTE DATA (1): ms = %', 1000 * (extract(epoch FROM end_ts - start_ts)); 
            ---------
                        
            start_ts := clock_timestamp();
            
                        
            query := 'UPDATE ' || output_table_name || 
                ' SET ' || col || ' = ' || 'linregr_predict(ARRAY[' || array_to_string(params, ', ') || ']::float8[], ARRAY['||array_to_string(array_remove(continuous_columns, col), ', ')||']::float8[], ARRAY['||
         			array_to_string(categorical_columns, ', ')||']::int[], '|| add_noise ||', false) ' ||
                ' WHERE NOT_NULL_CNT >= 2 AND NOT_NULL_CNT <'|| max_nulls || ' AND ' || col || '_ISNOTNULL IS NOT TRUE;';
            
            ---------
            
            EXECUTE query;
            end_ts := clock_timestamp();
            RAISE INFO 'IMPUTE DATA (2): ms = %', 1000 * (extract(epoch FROM end_ts - start_ts));
                        
            COMMIT;
            
        END LOOP;
        
        ---perform recomputation of partition all nulls
        
        start_ts := clock_timestamp();
        
        query := 'ALTER TABLE '||output_table_name || ' DETACH PARTITION '|| output_table_name ||'_notnullcnt0';
        RAISE DEBUG '%', query;
        EXECUTE query;

        query := 'ALTER TABLE '||output_table_name ||'_notnullcnt0 RENAME TO tmp_table';
        RAISE DEBUG '%', query;
        EXECUTE query;
        
        --consider moving tmp_column_names outside loop
        
        CALL create_missing_partition_high(
			model_params,
			model_params_size,
			continuous_columns,
        	categorical_columns,
        	continuous_columns_null, 
        	categorical_columns_null,
        	tmp_columns_names,
        	'tmp_table',
        	output_table_name || '_notnullcnt0',
        	add_noise
			);
			
        
        query := 'ALTER TABLE '|| output_table_name || '_notnullcnt0 ALTER COLUMN row_id SET NOT NULL';
        RAISE DEBUG '%', query;
        EXECUTE query;
            
        query := 'ALTER TABLE '|| output_table_name || ' ATTACH PARTITION '||output_table_name || '_notnullcnt0 FOR VALUES FROM (0) TO (1)';
        RAISE DEBUG '%', query;
        EXECUTE query;
        EXECUTE 'DROP TABLE tmp_table';
        
        end_ts := clock_timestamp();
        RAISE INFO 'materialize full nulls (ms) = %', 1000 * (extract(epoch FROM end_ts - start_ts));
        
        
    END LOOP;
END$$;

--CALL MICE_high('iris_impute', 'iris_impute_res', ARRAY['id','s_length', 's_width', 'p_length', 'p_width', 'target']::text[], ARRAY[]::text[], ARRAY['s_width', 'p_length', 'p_width']::text[], ARRAY[]::text[], 75, 2, false);


