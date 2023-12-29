#include <datatype.h>
#include <relation.h>

/**
 * Compute size of scalar values
 * @param num_cont number of columns in the table
 * @param is_cofactor if true, contains LDA/QDA/LR aggregates, otherwise Naive Bayes aggregates
 * @return
 */
static inline size_t size_scalar_array(size_t num_cont, uint16_t is_cofactor)
{
    if (is_cofactor){
        return (num_cont *(num_cont+ 3)) >> 1;
    }
    return num_cont * 2;
}

/**
 * Returns pointer to categorical values
 * @param c input aggregate
 * @return pointer to categorical values
 */

static inline char *relation_array(cofactor_t *c)
{
    return (char *)(scalar_array(c) + size_scalar_array(c->num_continuous_vars, c->aggregate_type));
}

/**
 * Returns pointer to categorical values
 * @param c input aggregate
 * @return pointer to categorical values
 */
static inline const char *crelation_array(const cofactor_t *c)
{
    return (const char *)(cscalar_array(c) + size_scalar_array(c->num_continuous_vars, c->aggregate_type));
}

/**
 * Computes the n. of columns of a cofactor matrix 1-hot encoded from a sequence of aggregates
 * @param cofactors input aggregates ()
 * @param n_aggregates n. of group by aggregates
 * @param cat_idxs OUTPUT: Indices of cat_unique_array. For each column stores begin:end of cat. values inside cat_unique_array
 * @param cat_unique_array OUTPUT: for every categorical column, stores sorted categorical values
 * @param drop_first If true, remove first entry for each categorical attribute (avoids multicollinearity in case of QDA)
 * @return number of values in a 1-hot encoded vector given the aggregates
 */
size_t n_cols_1hot_expansion(const cofactor_t **cofactors, size_t n_aggregates, uint32_t **cat_idxs, uint64_t **cat_unique_array, int drop_first)
{
    size_t num_categories = 0;
    for(size_t k=0; k<n_aggregates; k++) {
        num_categories += get_num_categories(crelation_array(cofactors[k]), cofactors[k]->num_categorical_vars, -1);//potentially overestimate
    }


    uint32_t *cat_vars_idxs = (uint32_t *)palloc0(sizeof(uint32_t) * (cofactors[0]->num_categorical_vars + 1)); // track start each cat. variable
    uint64_t *cat_array = (uint64_t *)palloc0(sizeof(uint64_t) * num_categories);//max. size

    (*cat_idxs) = cat_vars_idxs;
    (*cat_unique_array) = cat_array;

    cat_vars_idxs[0] = 0;
    size_t search_start = 0;        // within one category class
    size_t search_end = search_start;

    const char **relation_scan = (const char **)palloc0(sizeof(char *) * n_aggregates); // track start each cat. variable
    for(size_t k=0; k<n_aggregates; k++) {
        relation_scan[k] = crelation_array(cofactors[k]);
    }

    for (size_t i = 0; i < cofactors[0]->num_categorical_vars; i++) {
        for(size_t k=0; k<n_aggregates; k++) {
            relation_t *r = (relation_t *) relation_scan[k];
            //create sorted array
            for (size_t j = 0; j < r->num_tuples; j++) {
                size_t key_index = find_in_array(r->tuples[j].key, cat_array, search_start, search_end);
                elog(WARNING, "search val = %d from %d to %d", r->tuples[j].key, (int)search_start, (int)search_end);

                if (key_index == search_end) {
                    uint64_t value_to_insert = r->tuples[j].key;
                    uint64_t tmp;
                    for (size_t k = search_start; k < search_end; k++) {
                        if (value_to_insert < cat_array[k]) {
                            tmp = cat_array[k];
                            cat_array[k] = value_to_insert;
                            value_to_insert = tmp;
                        }
                    }
                    cat_array[search_end] = value_to_insert;
                    search_end++;
                }
            }
            relation_scan[k] += r->sz_struct;
        }
        cat_vars_idxs[i + 1] = cat_vars_idxs[i] + (search_end - search_start);
        elog(WARNING, "cat_vars_idxs = %d", cat_vars_idxs[i + 1]);
        search_start = search_end;
    }

    if (drop_first){//remove first entry for each categorical attribute (avoids multicollinearity when inverting matrix in QDA)
        for (size_t i = 0; i < cofactors[0]->num_categorical_vars; i++){
            cat_vars_idxs[i+1]-= (i+1);
            for(size_t j=cat_vars_idxs[i]; j<cat_vars_idxs[i+1]; j++){
                cat_array[j] = cat_array[j+i+1];
            }
        }
    }

    // count + numerical + 1-hot_categories
    return 1 + cofactors[0]->num_continuous_vars + cat_vars_idxs[cofactors[0]->num_categorical_vars];
}


/**
 * Computes the n. of columns of a cofactor matrix 1-hot encoded given a relation. Assumption: relations contain distinct tuples
 * @param relation_data_o input relation
 * @param num_categorical_vars number of categorical columns
 * @param label_categorical_sigma if >= 0, do not count categorical values in 'label_categorical_sigma' column
 * @return n. of columns of a cofactor matrix 1-hot encoded
 */
size_t get_num_categories(const char *relation_data_o, size_t num_categorical_vars, int label_categorical_sigma)
{
    size_t num_categories = 0;
    const char *relation_data = relation_data_o;
    for (size_t i = 0; i < num_categorical_vars; i++)
    {
        relation_t *r = (relation_t *) relation_data;

        if (label_categorical_sigma >= 0 && ((size_t)label_categorical_sigma) == i)
        {
            //skip this variable
            relation_data += r->sz_struct;
            continue;
        }
        num_categories += r->num_tuples;
        relation_data += r->sz_struct;
    }
    return num_categories;
}

/**
 * Find element in array
 * @param a element
 * @param array array
 * @param start start searching from this index
 * @param end end searching at this index
 * @return index of element or end if element is not in the array
 */
size_t find_in_array(uint64_t a, const uint64_t *array, size_t start, size_t end)
{
    size_t index = start;
    while (index < end)
    {
        if (array[index] == a)
            break;
        index++;
    }
    return index;
}