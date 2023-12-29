#include "cofactor.h"
#include "relation.h"
#include <math.h>

//#include <clapack.h>
#include <float.h>

#include <fmgr.h>
#include <catalog/pg_type.h>
#include <utils/array.h>
#include <utils/lsyscache.h>

#define  TYPALIGN_INT			'i'
#define  TYPALIGN_DOUBLE		'd'

/*
 * LAPACK routines
 * */

extern void dgesdd_(char *JOBZ, int *m, int *n, double *A, int *lda, double *s, double *u, int *LDU, double *vt, int *l,
                    double *work, int *lwork, int *iwork, int *info);
extern void dgelsd( int* m, int* n, int* nrhs, double* a, int* lda,
                    double* b, int* ldb, double* s, double* rcond, int* rank,
                    double* work, int* lwork, int* iwork, int* info );

extern void dgemm (char *TRANSA, char* TRANSB, int *M, int* N, int *K, double *ALPHA,
                   double *A, int *LDA, double *B, int *LDB, double *BETA, double *C, int *LDC);

extern void dgemv (char *TRANSA, int *M, int* N, double *ALPHA,
                   double *A, int *LDA, double *X, int *INCX, double *BETA, double *Y, int *INCY);

int compare( const void* a, const void* b)
{
    int int_a = * ( (int*) a );
    int int_b = * ( (int*) b );

    if ( int_a == int_b ) return 0;
    else if ( int_a < int_b ) return -1;
    else return 1;
}

//trains a LDA model given the required aggregates
//input: triple. Output: coefficients
PG_FUNCTION_INFO_V1(lda_train);

Datum lda_train(PG_FUNCTION_ARGS)
{
    const cofactor_t *cofactor = (const cofactor_t *)PG_GETARG_VARLENA_P(0);
    int label = PG_GETARG_INT64(1);
    double shrinkage = PG_GETARG_FLOAT8(2);

    size_t num_params = sizeof_sigma_matrix(cofactor, label);
    float8 *sigma_matrix = (float8 *)palloc0(sizeof(float8) * num_params * num_params);
    //count distinct classes in var
    size_t num_categories = 0;
    const char *relation_data = crelation_array(cofactor);
    for (size_t i = 0; i < cofactor->num_categorical_vars; i++)
    {
        relation_t *r = (relation_t *) relation_data;
        if (i == label) {
            //count unique keys
            num_categories = r->num_tuples;//stores categories
            break;
        }
        relation_data += r->sz_struct;
    }
    uint64_t *cat_array = NULL;//keeps all the categorical values inside the columns
    uint32_t *cat_vars_idxs = NULL;//keeps index of begin and end for each column in cat_array

    size_t tot_columns = n_cols_1hot_expansion(&cofactor, 1, &cat_vars_idxs, &cat_array, 0);//I need this function because I need cat_vars_idxs and cat_array
    //given aggregates, LDA requires a sigma (cofactor) matrix and a sum for each attribute grouped by label
    double *sum_vector = (double *)palloc0(sizeof(double) * num_params * num_categories);
    double *mean_vector = (double *)palloc0(sizeof(double) * (num_params-1) * num_categories);
    double *coef = (double *)palloc0(sizeof(double) * (num_params-1) * num_categories);//from mean to coeff

    build_sigma_matrix(cofactor, num_params, label, cat_array, cat_vars_idxs, 0, sigma_matrix);
    build_sum_matrix(cofactor, num_params, label, cat_array, cat_vars_idxs, 0, sum_vector);

    //Removed constant terms: we just need cofactor (covariance)
    //shift cofactor, coef and mean
    for (size_t j = 1; j < num_params; j++) {
        for (size_t k = 1; k < num_params; k++) {
            sigma_matrix[((j-1) * (num_params-1)) + (k-1)] = sigma_matrix[(j * num_params) + k];
        }
    }
    num_params--;

    //build covariance matrix and mean vectors from cofactor matrix and sum vector
    //we can reuse sigma_matrix to generate covariance, so don't allocate a new matrix
    //means is in mean_vector
    for (size_t i = 0; i < num_categories; i++) {
        for (size_t j = 0; j < num_params; j++) {
            for (size_t k = 0; k < num_params; k++) {
                sigma_matrix[(j*num_params)+k] -= ((float8)(sum_vector[(i*(num_params+1))+(j+1)] * sum_vector[(i*(num_params+1))+(k+1)]) / (float8) sum_vector[i*(num_params+1)]);//cofactor->count
            }
            coef[(i*num_params)+j] = sum_vector[(i*(num_params+1))+(j+1)] / sum_vector[(i*(num_params+1))];
            mean_vector[(j*num_categories)+i] = coef[(i*num_params)+(j)];
        }
    }

    //introduce covariance matrix introducing shrinkage (regulatization) if set
    double mu = 0;
    for (size_t j = 0; j < num_params; j++) {
        mu += sigma_matrix[(j*num_params)+j];
    }
    mu /= (float) num_params;

    for (size_t j = 0; j < num_params; j++) {
        for (size_t k = 0; k < num_params; k++) {
            sigma_matrix[(j*num_params)+k] *= (1-shrinkage);//apply shrinkage
        }
    }

    for (size_t j = 0; j < num_params; j++) {
        sigma_matrix[(j*num_params)+j] += shrinkage * mu;
    }

    //normalize with / count
    for (size_t j = 0; j < num_params; j++) {
        for (size_t k = 0; k < num_params; k++) {
            sigma_matrix[(j*num_params)+k] /= (float8)(cofactor->count);//or / cofactor->count - num_categories
        }
    }


    //Solve the optimization problem with LAPACK
    int err, lwork, rank;
    double rcond = -1.0;
    double wkopt;
    double* work;
    int *iwork = (int *) palloc(((int)((3 * num_params * log2(num_params/2)) + (11*num_params))) * sizeof(int));
    double *s = (double *) palloc(num_params * sizeof(double));
    int num_params_int = (int) num_params;
    int num_categories_int = (int) num_categories;

    lwork = -1;
    dgelsd( &num_params_int, &num_params_int, &num_categories_int, sigma_matrix, &num_params_int, coef, &num_params_int, s, &rcond, &rank, &wkopt, &lwork, iwork, &err);
    lwork = (int)wkopt;
    work = (double*)malloc( lwork*sizeof(double) );
    dgelsd( &num_params_int, &num_params_int, &num_categories_int, sigma_matrix, &num_params_int, coef, &num_params_int, s, &rcond, &rank, work, &lwork, iwork, &err);

    //check errors:
    if (err > 0)
        elog(ERROR, "LDA failed to converge. Error: %d", err);
    else if (err < 0)
        elog(ERROR, "the i-th argument had an illegal value. Error: %d", err);


    //compute intercept with lapack

    double alpha = 1;
    double beta = 0;
    double *res = (double *)palloc(num_categories*num_categories*sizeof(double));

    char task = 'N';
    dgemm(&task, &task, &num_categories_int, &num_categories_int, &num_params_int, &alpha, mean_vector, &num_categories_int, coef, &num_params_int, &beta, res, &num_categories_int);
    double *intercept = (double *)palloc(num_categories*sizeof(double));
    for (size_t j = 0; j < num_categories; j++) {
        intercept[j] = (res[(j*num_categories)+j] * (-0.5)) + log(sum_vector[(j) * (num_params+1)] / cofactor->count);
    }

    // export in pgpsql. Return values
    Datum *d = (Datum *)palloc(sizeof(Datum) * (num_categories + (num_params * num_categories) + 2 + cat_vars_idxs[cofactor->num_categorical_vars] + cofactor->num_categorical_vars));

    d[0] = Float8GetDatum((float)num_categories);//n. classes
    d[1] = Float8GetDatum((float)cofactor->num_categorical_vars);//size categorical columns -1 (label) (size idxs)

    size_t idx_output = 2;
    if (num_params - cofactor->num_continuous_vars > 0) {//there are categorical variables
        size_t remove = 0;
        //store categorical value indices of cat. columns (without label)
        for (size_t i = 0; i < cofactor->num_categorical_vars + 1; i++) {
            if (i == label) {
                remove = cat_vars_idxs[i+1] - cat_vars_idxs[i];
                continue;
            }
            d[idx_output] = Float8GetDatum((float) cat_vars_idxs[i] - remove);
            idx_output++;
        }
        //now store categorical values without label
        for (size_t i = 0; i < cat_vars_idxs[label]; i++) {
            d[idx_output] = Float8GetDatum((float) cat_array[i]);
            idx_output++;
        }
        for (size_t i = cat_vars_idxs[label + 1]; i < cat_vars_idxs[cofactor->num_categorical_vars]; i++) {
            d[idx_output] = Float8GetDatum((float) cat_array[i]);
            idx_output++;
        }
    }

    //add categorical labels
    for (size_t i = cat_vars_idxs[label]; i < cat_vars_idxs[label + 1]; i++) {
        d[idx_output] = Float8GetDatum((float) cat_array[i]);
        idx_output++;
    }

    //store coefficients
    for (int i = 0; i < num_params * num_categories; i++) {
        d[i + idx_output] = Float8GetDatum((float) coef[i]);
    }
    idx_output += num_params * num_categories;

    //store intercept
    for (int i = 0; i < num_categories; i++) {
        d[i + idx_output] = Float8GetDatum((float) intercept[i]);
    }

    ArrayType *a = construct_array(d, (num_categories + (num_params * num_categories) + 2 + cat_vars_idxs[cofactor->num_categorical_vars] + cofactor->num_categorical_vars), FLOAT8OID, sizeof(float8), true, TYPALIGN_INT);
    PG_RETURN_ARRAYTYPE_P(a);
}

//impute: given train parameters and features, returns class
PG_FUNCTION_INFO_V1(lda_impute);
Datum lda_impute(PG_FUNCTION_ARGS)
{
    ArrayType *means_covariance = PG_GETARG_ARRAYTYPE_P(0);//result of train
    ArrayType *feats_numerical = PG_GETARG_ARRAYTYPE_P(1);
    ArrayType *feats_categorical = PG_GETARG_ARRAYTYPE_P(2);

    Oid arrayElementType1 = ARR_ELEMTYPE(means_covariance);
    Oid arrayElementType2 = ARR_ELEMTYPE(feats_numerical);
    Oid arrayElementType3 = ARR_ELEMTYPE(feats_categorical);

    //extract arrays into C format

    int16 arrayElementTypeWidth1, arrayElementTypeWidth2, arrayElementTypeWidth3, arrayElementTypeWidth4;
    bool arrayElementTypeByValue1, arrayElementTypeByValue2, arrayElementTypeByValue3, arrayElementTypeByValue4;
    Datum *arrayContent1, *arrayContent2, *arrayContent3, *arrayContent4;
    bool *arrayNullFlags1, *arrayNullFlags2, *arrayNullFlags3, *arrayNullFlags4;
    int arrayLength1, arrayLength2, arrayLength3, arrayLength4;
    char arrayElementTypeAlignmentCode1, arrayElementTypeAlignmentCode2, arrayElementTypeAlignmentCode3, arrayElementTypeAlignmentCode4;

    get_typlenbyvalalign(arrayElementType1, &arrayElementTypeWidth1, &arrayElementTypeByValue1, &arrayElementTypeAlignmentCode1);
    deconstruct_array(means_covariance, arrayElementType1, arrayElementTypeWidth1, arrayElementTypeByValue1, arrayElementTypeAlignmentCode1,
                      &arrayContent1, &arrayNullFlags1, &arrayLength1);

    get_typlenbyvalalign(arrayElementType2, &arrayElementTypeWidth2, &arrayElementTypeByValue2, &arrayElementTypeAlignmentCode2);
    deconstruct_array(feats_numerical, arrayElementType2, arrayElementTypeWidth2, arrayElementTypeByValue2, arrayElementTypeAlignmentCode2,
                      &arrayContent2, &arrayNullFlags2, &arrayLength2);

    get_typlenbyvalalign(arrayElementType3, &arrayElementTypeWidth3, &arrayElementTypeByValue3, &arrayElementTypeAlignmentCode3);
    deconstruct_array(feats_categorical, arrayElementType3, arrayElementTypeWidth3, arrayElementTypeByValue3, arrayElementTypeAlignmentCode3,
                      &arrayContent3, &arrayNullFlags3, &arrayLength3);

    //re-create train values
    int size_cat_vars_idxs = DatumGetFloat8(arrayContent1[1]);
    int num_categories = (int) DatumGetFloat8(arrayContent1[0]);
    size_t curr_param_offset = 2;

    int num_params = arrayLength2;
    uint64_t *cat_vars_idxs;
    uint64_t *cat_vars;
    if (size_cat_vars_idxs > 0) {
        cat_vars_idxs = (uint64_t *) palloc0(sizeof(uint64_t) * (size_cat_vars_idxs));
        for(size_t i=0; i<size_cat_vars_idxs; i++){//build cat_var_idxs
            cat_vars_idxs[i] = DatumGetFloat8(arrayContent1[i+curr_param_offset]);//re-build index vector (begin:end of each cat. column)
        }
        curr_param_offset += size_cat_vars_idxs;
        num_params = arrayLength2 + cat_vars_idxs[size_cat_vars_idxs-1];
        cat_vars = (uint64_t *) palloc0(sizeof(uint64_t) * (cat_vars_idxs[size_cat_vars_idxs-1]));//build cat_var
        for(size_t i=0; i<cat_vars_idxs[size_cat_vars_idxs-1]; i++){
            cat_vars[i] = DatumGetFloat8(arrayContent1[i+curr_param_offset]);
        }
        curr_param_offset += cat_vars_idxs[size_cat_vars_idxs-1];
    }

    int *target_labels = (int *) palloc0(sizeof(int) * num_categories);
    for(size_t i=0; i<num_categories; i++) {
        target_labels[i] = DatumGetFloat8(arrayContent1[i + curr_param_offset]);//build label classes
    }

    curr_param_offset += num_categories;
    //build coefficients and intercept
    double *coefficients = (double *)palloc0(sizeof(double) * num_params * num_categories);
    float *intercept = (float *)palloc0(sizeof(float) * num_categories);

    for(int i=0;i< num_categories;i++)
        for(int j=0;j< num_params;j++)
            coefficients[(j*num_categories)+i] = (double) DatumGetFloat8(arrayContent1[(i*num_params)+j+curr_param_offset]);

    curr_param_offset += (num_params * num_categories);

    //build intercept
    for(int i=0;i<num_categories;i++)
        intercept[i] = (double) DatumGetFloat8(arrayContent1[i+curr_param_offset]);

    //allocate numerical features

    double *feats_c = (double *)palloc0(sizeof(double) * (num_params));
    for(int i=0;i<arrayLength2;i++) {
        feats_c[i] = (double) DatumGetFloat8(arrayContent2[i]);
    }

    //allocate categorical features

    for(int i=0;i<arrayLength3;i++) {
        int class = DatumGetInt64(arrayContent3[i]);
        size_t index = find_in_array(class, cat_vars, cat_vars_idxs[i], cat_vars_idxs[i+1]);//use cat.vars to find index of cat. value
        assert (index < cat_vars_idxs[i+1]);//1-hot used here, without removing values from 1-hot
        feats_c[arrayLength2 + index] = 1;//build 1-hot feat vector
    }

    //end unpacking
    //use lapack to generate result with matrix vector multiplication

    char task = 'N';
    double alpha = 1;
    int increment = 1;
    double beta = 0;

    double *res = (double *)palloc0(sizeof(double) * num_categories);
    dgemv(&task, &num_categories, &num_params, &alpha, coefficients, &num_categories, feats_c, &increment, &beta, res, &increment);

    double max = -DBL_MAX;
    int class = 0;

    for(int i=0;i<num_categories;i++) {
        double val = res[i] + intercept[i];
        elog(WARNING, " val: %lf", val);
        if (val > max){
            max = val;
            class = target_labels[i];
        }
    }
    PG_RETURN_INT64(class);
}
