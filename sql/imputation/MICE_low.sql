CREATE OR REPLACE PROCEDURE create_missing_partition_low(
			params_models float8[],
			params_size int[],
			continuous_columns text[],
        	categorical_columns text[],
        	continuous_columns_null text[], 
        	categorical_columns_null text[],
        	tmp_columns_names text[],
        	old_table_name text,
        	new_table_name text
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
         	array_to_string(categorical_columns, ', ')||']::int[], true, false)' || ' AS ' || x                
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


CREATE OR REPLACE PROCEDURE MICE_low(
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
    cofactor_null cofactor;
    col text;
    params float8[];
    model_params float8[];
    model_params_size int[];
    tmp_columns_names text[];
    label_index int4;
    max_range int4;    
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
        SELECT array_agg(x || '_ISNULL bool')
        FROM unnest(continuous_columns_null) AS x
    )||
    (
        SELECT array_agg(x || '_ISNULL bool')
        FROM unnest(categorical_columns_null) AS x
    )
     || (
        ARRAY [ 'NULL_CNT int', 'NULL_COL_ID int', 'ROW_ID serial' ]
    )
    INTO tmp_array;
    
    query := 'CREATE UNLOGGED TABLE ' || output_table_name || '( ' ||
                array_to_string(tmp_array, ', ') || ') PARTITION BY RANGE (NULL_CNT)';
    RAISE DEBUG '%', query;
    EXECUTE QUERY;

    query := 'CREATE UNLOGGED TABLE ' || output_table_name || '_nullcnt0' || 
             ' PARTITION OF ' || output_table_name || 
             ' FOR VALUES FROM (0) TO (1) WITH (fillfactor=100)';
    RAISE DEBUG '%', query;
    EXECUTE QUERY;

    query := 'CREATE UNLOGGED TABLE ' || output_table_name || '_nullcnt1' || 
             ' PARTITION OF ' || output_table_name || 
             ' FOR VALUES FROM (1) TO (2) PARTITION BY RANGE (NULL_COL_ID)';
    RAISE DEBUG '%', query;
    EXECUTE QUERY;
    
    if array_length(categorical_columns_null, 1) > 0 then
        max_range := array_length(continuous_columns_null, 1) + array_length(categorical_columns_null, 1);
    else
        max_range := array_length(continuous_columns_null, 1);
    end if;
    
    
    FOR col_id in 1..max_range LOOP
        query := 'CREATE UNLOGGED TABLE ' || output_table_name || '_nullcnt1_col' || col_id || 
                 ' PARTITION OF ' || output_table_name || '_nullcnt1' || 
                 ' FOR VALUES FROM (' || col_id || ') TO (' || col_id + 1 || ') WITH (fillfactor=100)';
        RAISE DEBUG '%', query;
        EXECUTE QUERY;
    END LOOP;
    
    IF max_range >= 2 THEN
    	    if array_length(categorical_columns_null, 1) > 0 then
    	    	max_nulls := array_length(continuous_columns_null, 1) + array_length(categorical_columns_null, 1);
    			query := 'CREATE UNLOGGED TABLE ' || output_table_name || '_nullcnt2' || 
             	' PARTITION OF ' || output_table_name ||
             	' FOR VALUES FROM (2) TO (' || max_nulls || ') WITH (fillfactor='||fillfactor||')';
             	
             	RAISE DEBUG '%', query;
    			EXECUTE query;
    			
    			query := 'CREATE UNLOGGED TABLE ' || output_table_name || '_nullcnt3' || 
        		' PARTITION OF ' || output_table_name ||
            	' FOR VALUES FROM (' || max_nulls || ') TO (' || max_nulls + 1 ||') WITH (fillfactor=100)';
            	RAISE DEBUG '%', query;
    			EXECUTE query;
    			
             	
             else
             	max_nulls := array_length(continuous_columns_null, 1);
            	query := 'CREATE UNLOGGED TABLE ' || output_table_name || '_nullcnt2' || 
             	' PARTITION OF ' || output_table_name ||
             	' FOR VALUES FROM (2) TO (' || max_nulls || ') WITH (fillfactor='||fillfactor||')';
             	
             	RAISE DEBUG '%', query;
    			EXECUTE query;
    			
    			query := 'CREATE UNLOGGED TABLE ' || output_table_name || '_nullcnt3' || 
        		' PARTITION OF ' || output_table_name ||
            	' FOR VALUES FROM (' || max_nulls || ') TO (' || max_nulls + 1 ||') WITH (fillfactor=100)';
            	RAISE DEBUG '%', query;
    			EXECUTE query;
             	
    		end if;
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
        SELECT array_agg(x || ' IS NULL')
        FROM unnest(continuous_columns_null) AS x
    ) || (
        SELECT array_agg(x || ' IS NULL')
        FROM unnest(categorical_columns_null) AS x
    ) || ( 
        SELECT ARRAY [ array_to_string(array_agg('(CASE WHEN ' || x || ' IS NULL THEN 1 ELSE 0 END)'), ' + ') ]
        FROM unnest(continuous_columns_null || categorical_columns_null) AS x
    ) INTO tmp_array;
    
    SELECT (
        SELECT array_agg(
        		CASE 
        			WHEN array_position(continuous_columns_null, x) IS NULL THEN
        				' WHEN ' || x || ' IS NULL THEN ' || array_position (categorical_columns_null, x) + array_length(continuous_columns_null, 1)
                    ELSE 
                    	' WHEN ' || x || ' IS NULL THEN ' || array_position (continuous_columns_null, x)
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
            'FROM ' || output_table_name || ' WHERE NULL_CNT <'|| max_nulls;
    RAISE DEBUG '%', query;

    start_ts := clock_timestamp();
    EXECUTE query INTO STRICT cofactor_global;
    end_ts := clock_timestamp();
    RAISE INFO 'COFACTOR global: ms = %', 1000 * (extract(epoch FROM end_ts - start_ts));
    
    COMMIT;
    
    FOR i in 1..iterations LOOP
    
    	model_params := ARRAY[]::float8[];
    	model_params_size := ARRAY[]::int[];
    
        RAISE INFO 'Iteration %', i;
        
        FOREACH col in ARRAY categorical_columns_null LOOP --categorical imputation
            RAISE INFO '  |- Column %', col;
            
            -- COMPUTE COFACTOR WHERE $col IS NULL
            query_null1 := 'SELECT SUM(to_cofactor(' ||
                    'ARRAY[ ' || array_to_string(continuous_columns, ', ') || ' ]::float8[],' ||
                    'ARRAY[ ' || array_to_string(categorical_columns, ', ') || ' ]::int4[]' ||
                ')) '
            'FROM ' || output_table_name || ' '
            'WHERE (NULL_CNT = 1 AND NULL_COL_ID = ' || array_position(categorical_columns_null, col) + array_length(continuous_columns_null, 1) || ');';
            RAISE DEBUG '%', query_null1;
            
            start_ts := clock_timestamp();
            EXECUTE query_null1 INTO STRICT cofactor_null;
            end_ts := clock_timestamp();
            RAISE INFO 'COFACTOR NULL (1): ms = %', 1000 * (extract(epoch FROM end_ts - start_ts));
            -- RAISE DEBUG 'COFACTOR NULL = %', cofactor_null;

            IF cofactor_null IS NOT NULL THEN
                cofactor_global := cofactor_global - cofactor_null;
            END IF;
            
            --RAISE DEBUG 'COFACTOR GLOBAL = %', cofactor_global;
            
            query_null2 := 'SELECT SUM(to_cofactor(' ||
                    'ARRAY[ ' || array_to_string(continuous_columns, ', ') || ' ]::float8[],' ||
                    'ARRAY[ ' || array_to_string(categorical_columns, ', ') || ' ]::int4[]' ||
                ')) '
            'FROM ' || output_table_name || ' '
            ' WHERE NULL_CNT >= 2 AND NULL_CNT <'|| max_nulls || ' AND ' || col || '_ISNULL';
            RAISE DEBUG '%', query_null2;
            
            start_ts := clock_timestamp();
            EXECUTE query_null2 INTO STRICT cofactor_null;
            end_ts := clock_timestamp();
            RAISE INFO 'COFACTOR NULL (2): ms = %', 1000 * (extract(epoch FROM end_ts - start_ts));
            -- RAISE DEBUG 'COFACTOR NULL = %', cofactor_null;

            IF cofactor_null IS NOT NULL THEN
                cofactor_global := cofactor_global - cofactor_null;
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
            
            --skip current label in categorical features
            --cat columns: letters
            
            --offset wrong when label skipped
            RAISE DEBUG ' cat columns 3: %',categorical_columns ;
            RAISE DEBUG ' cat columns null 3: %', categorical_columns_null;
            
            RAISE DEBUG ' SUBQUERY: %', subquery;
                        
            query := 'SELECT array_agg(quote_ident(column_name)) FROM information_schema.columns WHERE table_name = ''' || output_table_name || ''';';
            EXECUTE query INTO tmp_columns_names;
            RAISE DEBUG '%', tmp_columns_names;
            
            query := 'ALTER TABLE '||output_table_name || '_nullcnt1 DETACH PARTITION '|| output_table_name ||'_nullcnt1_col'||array_position(categorical_columns_null, col)+ array_length(continuous_columns_null, 1);
            RAISE DEBUG '%', query;
            EXECUTE query;

            query := 'ALTER TABLE '||output_table_name ||'_nullcnt1_col'||array_position(categorical_columns_null, col)+ array_length(continuous_columns_null, 1)  ||' RENAME TO tmp_table';

            RAISE DEBUG '%', query;
            EXECUTE query;
            
            
            SELECT array_agg(LOWER(x)) FROM unnest(categorical_columns) as x INTO columns_lower;
            
            
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
                      
            
            query := 'CREATE UNLOGGED TABLE ' || output_table_name || '_nullcnt1_col'||array_position(categorical_columns_null, col)+ array_length(continuous_columns_null, 1)  || ' AS SELECT ';
            query := query || array_to_string(tmp_array3, ' , ') ||' FROM tmp_table';
                    RAISE DEBUG '%', query;
            EXECUTE query;
        	query := 'ALTER TABLE '|| output_table_name || '_nullcnt1_col'||array_position(categorical_columns_null, col) + array_length(continuous_columns_null, 1)  ||' ALTER COLUMN row_id SET NOT NULL';
        	        RAISE DEBUG '%', query;
                    EXECUTE query;
            query := 'ALTER TABLE '|| output_table_name || '_nullcnt1 ATTACH PARTITION '||output_table_name || '_nullcnt1_col'||array_position(categorical_columns_null, col)+ array_length(continuous_columns_null, 1)  ||' FOR VALUES FROM (' || array_position(categorical_columns_null, col)+ array_length(continuous_columns_null, 1)  || ') TO (' || array_position(categorical_columns_null, col)+ array_length(continuous_columns_null, 1)  + 1 || ')';
                    RAISE DEBUG '%', query;
                    EXECUTE query;
                    EXECUTE 'DROP TABLE tmp_table';
            ----
            

            end_ts := clock_timestamp();
            RAISE INFO 'IMPUTE DATA (1): ms = %', 1000 * (extract(epoch FROM end_ts - start_ts));
                        
            
            start_ts := clock_timestamp();
            
            ---remove this or the other one
            
            query := 'UPDATE ' || output_table_name || 
                ' SET ' || col || ' = lda_predict(ARRAY[ ' || array_to_string(params, ', ') || ']::float8[], ' || 'ARRAY[ ' || array_to_string(continuous_columns, ', ') || ' ]::float8[], ARRAY['
                || array_to_string(array_remove(categorical_columns, col), ', ')  || ']::int[], false)' ||
                ' WHERE NULL_CNT >= 2 AND NULL_CNT <'|| max_nulls || ' AND ' || col || '_ISNULL;';
            
            EXECUTE query;

            end_ts := clock_timestamp();

            
            RAISE INFO 'IMPUTE DATA (2): ms = %', 1000 * (extract(epoch FROM end_ts - start_ts));
            
            ---------------------------------
            
            
            start_ts := clock_timestamp();
            EXECUTE query_null1 INTO STRICT cofactor_null;
            end_ts := clock_timestamp();
            RAISE INFO 'COFACTOR NULL (1): ms = %', 1000 * (extract(epoch FROM end_ts - start_ts));

            IF cofactor_null IS NOT NULL THEN
                cofactor_global := cofactor_global + cofactor_null;
            END IF;

            start_ts := clock_timestamp();
            EXECUTE query_null2 INTO STRICT cofactor_null;
            end_ts := clock_timestamp();
            RAISE INFO 'COFACTOR NULL (2): ms = %', 1000 * (extract(epoch FROM end_ts - start_ts));
            
            IF cofactor_null IS NOT NULL THEN
                cofactor_global := cofactor_global + cofactor_null;
            END IF;
            
            COMMIT;
            
        END LOOP;
        
        FOREACH col in ARRAY continuous_columns_null LOOP
            RAISE INFO '  |- Column %', col;
            
            -- COMPUTE COFACTOR WHERE $col IS NULL
            query_null1 := 'SELECT SUM(to_cofactor(' ||
                    'ARRAY[ ' || array_to_string(continuous_columns, ', ') || ' ]::float8[],' ||
                    'ARRAY[ ' || array_to_string(categorical_columns, ', ') || ' ]::int4[]' ||
                ')) '
            'FROM ' || output_table_name || ' '
            'WHERE (NULL_CNT = 1 AND NULL_COL_ID = ' || array_position(continuous_columns_null, col) || ');';
            RAISE DEBUG '%', query_null1;
            
            start_ts := clock_timestamp();
            EXECUTE query_null1 INTO STRICT cofactor_null;
            end_ts := clock_timestamp();
            RAISE INFO 'COFACTOR NULL (1): ms = %', 1000 * (extract(epoch FROM end_ts - start_ts));

            IF cofactor_null IS NOT NULL THEN
                cofactor_global := cofactor_global - cofactor_null;
            END IF;

            query_null2 := 'SELECT SUM(to_cofactor(' ||
                    'ARRAY[ ' || array_to_string(continuous_columns, ', ') || ' ]::float8[],' ||
                    'ARRAY[ ' || array_to_string(categorical_columns, ', ') || ' ]::int4[]' ||
                ')) '
            'FROM ' || output_table_name || ' '
            ' WHERE NULL_CNT >= 2 AND NULL_CNT <'|| max_nulls || ' AND ' || col || '_ISNULL;';
            RAISE INFO '%', query_null2;
            
            start_ts := clock_timestamp();
            EXECUTE query_null2 INTO STRICT cofactor_null;
            end_ts := clock_timestamp();
            RAISE INFO 'COFACTOR NULL (2): ms = %', 1000 * (extract(epoch FROM end_ts - start_ts));

            IF cofactor_null IS NOT NULL THEN
                cofactor_global := cofactor_global - cofactor_null;
            END IF;

            -- TRAIN
            label_index := array_position(continuous_columns, col);
            start_ts := clock_timestamp();
            params := ridge_linear_regression(cofactor_global, label_index - 1, 0.001, 0, 10000, true, false);            
            model_params := model_params || params;
            model_params_size := array_append(model_params_size, array_length(params, 1));
            end_ts := clock_timestamp();
            RAISE INFO 'TRAIN: ms = %', 1000 * (extract(epoch FROM end_ts - start_ts));
            RAISE DEBUG '%', params;

            -- IMPUTE
            
            start_ts := clock_timestamp();
            
            query := 'SELECT array_agg(quote_ident(column_name)) FROM information_schema.columns WHERE table_name = ''' || output_table_name || ''';';
            EXECUTE query INTO tmp_columns_names;
            RAISE DEBUG '%', tmp_columns_names;
            
            query := 'ALTER TABLE '||output_table_name || '_nullcnt1 DETACH PARTITION '|| output_table_name ||'_nullcnt1_col'||array_position(continuous_columns_null, col);
            RAISE DEBUG '%', query;
            EXECUTE query;

            query := 'ALTER TABLE '||output_table_name ||'_nullcnt1_col'||array_position(continuous_columns_null, col) ||' RENAME TO tmp_table';

            RAISE DEBUG '%', query;
            EXECUTE query;
            
                                                
            --- 'normal_rand(1, 0, ' || sqrt(params[array_length(params, 1)])::text || ')'
            SELECT array_agg(LOWER(x)) FROM unnest(continuous_columns) as x INTO columns_lower;

            
            SELECT array_agg(
                CASE WHEN array_position(columns_lower, LOWER(x)) = label_index THEN
                    'linregr_predict(ARRAY[' || array_to_string(params, ', ') || ']::float8[], ARRAY['||array_to_string(array_remove(continuous_columns, col), ', ')||']::float8[], ARRAY['||
         			array_to_string(categorical_columns, ', ')||']::int[], '||add_noise||', false)' || ' AS ' || x                
                ELSE
                    x || ' AS ' || x
                END
                )
            FROM unnest(tmp_columns_names) AS x
            INTO tmp_array3;
            
            --EXECUTE query;
                        
            
            query := 'CREATE UNLOGGED TABLE ' || output_table_name || '_nullcnt1_col'||array_position(continuous_columns_null, col) || ' AS SELECT ';
            query := query || array_to_string(tmp_array3, ' , ') ||' FROM tmp_table';
                    RAISE DEBUG '%', query;
            EXECUTE query;
        	query := 'ALTER TABLE '|| output_table_name || '_nullcnt1_col'||array_position(continuous_columns_null, col) ||' ALTER COLUMN row_id SET NOT NULL';
        	RAISE DEBUG '%', query;
            EXECUTE query;
            query := 'ALTER TABLE '|| output_table_name || '_nullcnt1 ATTACH PARTITION '||output_table_name || '_nullcnt1_col'||array_position(continuous_columns_null, col) ||' FOR VALUES FROM (' || array_position(continuous_columns_null, col) || ') TO (' || array_position(continuous_columns_null, col) + 1 || ')';
            RAISE DEBUG '%', query;
            EXECUTE query;
            EXECUTE 'DROP TABLE tmp_table';
            
                        
            end_ts := clock_timestamp();
            RAISE INFO 'IMPUTE DATA (1): ms = %', 1000 * (extract(epoch FROM end_ts - start_ts));
            
            
            start_ts := clock_timestamp();
            
            
            ---remove this or the other one
            
            query := 'UPDATE ' || output_table_name || ' SET ' || col || ' = ' || 'linregr_predict(ARRAY[' || array_to_string(params, ', ') || ']::float8[], ARRAY['||array_to_string(array_remove(continuous_columns, col), ', ')||']::float8[], ARRAY['||
         			array_to_string(categorical_columns, ', ')||']::int[], '||add_noise||', false) ' ||            
                ' WHERE NULL_CNT >= 2 AND NULL_CNT <'|| max_nulls || ' AND ' || col || '_ISNULL;';
            
            ---------
            

            EXECUTE query;
            end_ts := clock_timestamp();
            RAISE INFO 'IMPUTE DATA (2): ms = %', 1000 * (extract(epoch FROM end_ts - start_ts));
            
            start_ts := clock_timestamp();
            EXECUTE query_null1 INTO STRICT cofactor_null;
            end_ts := clock_timestamp();
            RAISE INFO 'COFACTOR NULL (1): ms = %', 1000 * (extract(epoch FROM end_ts - start_ts));

            IF cofactor_null IS NOT NULL THEN
                cofactor_global := cofactor_global + cofactor_null;
            END IF;

            start_ts := clock_timestamp();
            EXECUTE query_null2 INTO STRICT cofactor_null;
            end_ts := clock_timestamp();
            RAISE INFO 'COFACTOR NULL (2): ms = %', 1000 * (extract(epoch FROM end_ts - start_ts));
            
            IF cofactor_null IS NOT NULL THEN
                cofactor_global := cofactor_global + cofactor_null;
            END IF;
            
            COMMIT;
            
        END LOOP;
        
        ---perform recomputation of partition all nulls
        
        start_ts := clock_timestamp();
        
        query := 'ALTER TABLE '||output_table_name || ' DETACH PARTITION '|| output_table_name ||'_nullcnt3';
        RAISE DEBUG '%', query;
        EXECUTE query;

        query := 'ALTER TABLE '||output_table_name ||'_nullcnt3 RENAME TO tmp_table';
        RAISE DEBUG '%', query;
        EXECUTE query;
        
        --consider moving tmp_column_names outside loop
        
        CALL create_missing_partition_low(
			model_params,
			model_params_size,
			continuous_columns,
        	categorical_columns,
        	continuous_columns_null, 
        	categorical_columns_null,
        	tmp_columns_names,
        	'tmp_table',
        	output_table_name || '_nullcnt3'
			);
			
        
        query := 'ALTER TABLE '|| output_table_name || '_nullcnt3 ALTER COLUMN row_id SET NOT NULL';
        RAISE DEBUG '%', query;
        EXECUTE query;
            
        query := 'ALTER TABLE '|| output_table_name || ' ATTACH PARTITION '||output_table_name || '_nullcnt3 FOR VALUES FROM (' || max_nulls || ') TO (' || max_nulls + 1 || ')';
        RAISE DEBUG '%', query;
        EXECUTE query;
        EXECUTE 'DROP TABLE tmp_table';
        
        end_ts := clock_timestamp();
        RAISE INFO 'materialize full nulls (ms) = %', 1000 * (extract(epoch FROM end_ts - start_ts));
        
        
    END LOOP;
END$$;

--CALL MICE_high('iris_impute', 'iris_impute_res', ARRAY['id','s_length', 's_width', 'p_length', 'p_width', 'target']::text[], ARRAY[]::text[], ARRAY['s_width', 'p_length', 'p_width']::text[], ARRAY[]::text[], 75, 2, false);


