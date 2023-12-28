
#ifndef POSTGRES_NB_AGGREGATES_H
#define POSTGRES_NB_AGGREGATES_H


#include <postgres.h>
#include <assert.h>
#include <datatype.h>


/*
 * Aggregates for Naive Bayes: it usually requires less aggregates than LDA/QDA
 * Naive Bayes requires sum(xi) for categorical columns, and sum(xi) and sum(xi*xi) for numerical columns (group by target)
 */

//stores just the sum for each categorical column
inline size_t size_relation_array(size_t num_cat)
{
    return num_cat;
}

//stores xi and xi^2 for each continuous column
inline size_t size_scalar_array(size_t num_cont)
{
    return num_cont * 2;
}

//returns the relation array
inline char *relation_array(cofactor_t *c)
{
    return (char *)(scalar_array(c) + size_scalar_array(c->num_continuous_vars));
}

inline const char *crelation_array(const cofactor_t *c)
{
    return (const char *)(cscalar_array(c) + size_scalar_array(c->num_continuous_vars));
}


inline size_t sizeof_cofactor_t(const cofactor_t *c)
{
    return sizeof(cofactor_t) +
           size_scalar_array(c->num_continuous_vars) * sizeof(float8) +
           c->sz_relation_data;
}

#endif //POSTGRES_NB_AGGREGATES_H
