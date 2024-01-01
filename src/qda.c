#include "cofactor.h"
#include "relation.h"
#include <postgres.h>
#include <fmgr.h>
#include <catalog/pg_type.h>
#include <utils/array.h>
#include <math.h>
#include <assert.h>
#include <float.h>

#include <utils/array.h>
#include <utils/lsyscache.h>

#define  TYPALIGN_INT			'i'
#define  TYPALIGN_DOUBLE		'd'

extern void dgemm (char *TRANSA, char* TRANSB, int *M, int* N, int *K, double *ALPHA,
                   double *A, int *LDA, double *B, int *LDB, double *BETA, double *C, int *LDC);

extern void dgemv (char *TRANSA, int *M, int* N, double *ALPHA,
                   double *A, int *LDA, double *X, int *INCX, double *BETA, double *Y, int *INCY);

extern void dgesvd( char* jobu, char* jobvt, int* m, int* n, double* a,
                    int* lda, double* s, double* u, int* ldu, double* vt, int* ldvt,
                    double* work, int* lwork, int* info );

extern void dscal( int* N, double* DA, double *DX, int* INCX );

/*
static void print_matrix(size_t sz, const double *m)
{
    for (size_t i = 0; i < sz; i++)
    {
        for(size_t j = 0; j < sz; j++)
        {
            elog(WARNING, "%zu, %zu -> %f", i, j, m[(i * sz) + j]);
        }
    }
}*/


PG_FUNCTION_INFO_V1(train_qda);

/**
 * Train QDA
 * @param cofactors : sequence of cofactors (generated grouping by label)
 * @param shrinkage : regularization
 * @param avoid_collinearity : if not 0, for each categorical column the first value is represented as all 0s rather than introducing a column
 * @return QDA parameters
 */
Datum train_qda(PG_FUNCTION_ARGS)
{
    ArrayType *cofactors = PG_GETARG_ARRAYTYPE_P(0);
    Oid arrayElementType1 = ARR_ELEMTYPE(cofactors);
    ArrayType *labels = PG_GETARG_ARRAYTYPE_P(1);
    Oid arrayElementType2 = ARR_ELEMTYPE(labels);
    bool normalize = PG_GETARG_BOOL(2);
    //double shrinkage = PG_GETARG_FLOAT8(1);
    int drop_first = 1;//PG_GETARG_INT64(2);


    int16 arrayElementTypeWidth1, arrayElementTypeWidth2;
    bool arrayElementTypeByValue1, arrayElementTypeByValue2;
    Datum *arrayContent1, *arrayContent2;
    bool *arrayNullFlags1, *arrayNullFlags2;
    int n_aggregates, n_aggregates_2;
    char arrayElementTypeAlignmentCode1, arrayElementTypeAlignmentCode2;

    get_typlenbyvalalign(arrayElementType1, &arrayElementTypeWidth1, &arrayElementTypeByValue1, &arrayElementTypeAlignmentCode1);
    deconstruct_array(cofactors, arrayElementType1, arrayElementTypeWidth1, arrayElementTypeByValue1, arrayElementTypeAlignmentCode1,
                      &arrayContent1, &arrayNullFlags1, &n_aggregates);

    get_typlenbyvalalign(arrayElementType2, &arrayElementTypeWidth2, &arrayElementTypeByValue2, &arrayElementTypeAlignmentCode2);
    deconstruct_array(labels, arrayElementType2, arrayElementTypeWidth2, arrayElementTypeByValue2, arrayElementTypeAlignmentCode2,
                      &arrayContent2, &arrayNullFlags2, &n_aggregates_2);

    const cofactor_t **aggregates = (const cofactor_t **)palloc0(sizeof(cofactor_t *) * n_aggregates);
    for(size_t i=0; i<n_aggregates; i++) {
        aggregates[i] = (cofactor_t *) DatumGetPointer(arrayContent1[i]);
    }

    uint32_t *cat_idxs = NULL;
    uint64_t *cat_array = NULL;
    size_t num_params = n_cols_1hot_expansion(aggregates, n_aggregates, &cat_idxs, &cat_array, drop_first);//enable drop first
    //size_t num_categories = cat_idxs[aggregates[0]->num_categorical_vars];

    int m = num_params-1;
    double *sing_values = (double*)palloc(m*sizeof(double));
    double *u = (double*)palloc(m*m*sizeof(double));
    double *vt = (double*) palloc(m*m*sizeof(double));
    double* inva = (double*)palloc(m*m*sizeof(double));
    double* mean_vector = (double*)palloc0(sizeof(double) * (m));//remove constant term
    double *lin_result = (double *)palloc0(sizeof(double) * m);

    size_t res_size = (1+1+ (((m*m)+m+1)*n_aggregates)) + n_aggregates_2;
    if (aggregates[0]->num_categorical_vars > 0){
        res_size += aggregates[0]->num_categorical_vars+1 + cat_idxs[aggregates[0]->num_categorical_vars];
    }
    if(normalize){
        res_size += m;
    }

    Datum *result = (Datum *)palloc(sizeof(Datum) * res_size);
    result[0] = Float8GetDatum(n_aggregates);

    size_t param_out_index = 2;

    //save categorical (unique) values and begin:end indices in output
    if (aggregates[0]->num_categorical_vars > 0) {
        result[1] = Float8GetDatum(aggregates[0]->num_categorical_vars + 1);
        for(size_t i=0; i<aggregates[0]->num_categorical_vars+1; i++)
            result[i+2] = Float8GetDatum(cat_idxs[i]);

        param_out_index = aggregates[0]->num_categorical_vars+3;
        for(size_t i=0; i<cat_idxs[aggregates[0]->num_categorical_vars]; i++)
            result[param_out_index+i] = Float8GetDatum(cat_array[i]);
        param_out_index += cat_idxs[aggregates[0]->num_categorical_vars];
    }
    else
        result[1] = Float8GetDatum(0);

    //copy labels
    for(size_t i=0; i<n_aggregates_2; i++) {
        result[param_out_index + i] =  Float8GetDatum(DatumGetInt32(arrayContent2[i]));
    }

    param_out_index += n_aggregates_2;

    double tot_tuples = 0;
    for(size_t i=0; i<n_aggregates; i++){
        tot_tuples += aggregates[i]->count;
    }

    float8 **sigma_matrices = (float8 **)palloc0(sizeof(float8 **) * n_aggregates);
    //double **sum_vectors = (double **)palloc0(sizeof(double **) * n_aggregates);

    for(size_t i=0; i<n_aggregates; i++) {
        sigma_matrices[i] = (float8 *)palloc0(sizeof(float8 *) * num_params * num_params);
        //sum_vectors[i] = (double *)palloc0(sizeof(double *) * num_params * n_aggregates);

        build_sigma_matrix(aggregates[i], num_params, -1, cat_array, cat_idxs, drop_first, sigma_matrices[i]);
        //build_sum_vector(aggregates[i], num_params, cat_array, cat_idxs, drop_first, sum_vectors[i]);
    }

    //handle normalization
    double *means = NULL;
    double *std = NULL;
    if (normalize){
        means = (double *)palloc0(sizeof(double) * num_params);
        std = (double *)palloc0(sizeof(double) * num_params);
        for(size_t i=0; i<n_aggregates; i++) {
            for (size_t j = 0; j < num_params; j++) {
                means[j] += sigma_matrices[i][j];
                std[j] += sigma_matrices[i][(j*num_params)+j];
            }
        }
        for (size_t j = 0; j < num_params; j++) {
            means[j] /= tot_tuples;
            std[j] = sqrt((std[j]/tot_tuples) - pow(means[j], 2));//sqrt((sigma[(i*num_params)+i]/sigma[0]) - pow(sigma[i]/sigma[0], 2));
        }
        //std[0] = 1;

        //normalize
        for(size_t mm=0; mm<n_aggregates; mm++) {
            float8 *sigma = sigma_matrices[mm];
            //double *sum_vector = sum_vectors[mm];

            //elog(WARNING, "Printing sigma 0... mm= %lu", mm);
            //print_matrix(num_params, sigma);

            for (size_t i = 1; i < num_params; i++) {
                for (size_t j = 1; j < num_params; j++) {
                    sigma[(i * num_params) + j] =
                            (sigma[(i * num_params) + j] - (means[i] * sigma[j]) - (means[j] * sigma[i]) +
                             (sigma[0] * means[j] * means[i])) / (std[i] * std[j]);
                }
            }

            //fix first row and col
            for (size_t i = 1; i < num_params; i++) {
                sigma[i] = (sigma[i] - (means[i] * sigma[0])) / std[i];
                sigma[(i * num_params)] = (sigma[(i * num_params)]- (means[i] * sigma[0])) / std[i];
            }

            //elog(WARNING, "Printing sigma 1...");
            //print_matrix(num_params, sigma);

            //standarize sums.
            /*
            for (size_t i=0; i<n_aggregates;i++){
                for(size_t j=1; j<num_params; j++){
                    sum_vector[(i*num_params)+j] = (sum_vector[(i*num_params)+j] - (means[j]*sum_vector[i*num_params])) / std[j];
                    elog(WARNING, "sum vector %lf...", sum_vector[j]);
                }
            }*/
        } //end for each aggregate
    }

    for(size_t ii=0; ii<n_aggregates; ii++) {

        float8 *sigma_matrix = sigma_matrices[ii];
        //double *sum_vector = sum_vectors[ii];

        //print_matrix(num_params, sigma_matrix);

        //generate covariance matrix (for each class)
        double count_tuples = sigma_matrix[0];
        for(size_t i=0; i<m; i++)
            mean_vector[i] = sigma_matrix[i+1];//this is sum vector at the moment

        for (size_t j = 1; j < num_params; j++) {
            for (size_t k = 1; k < num_params; k++) {
                sigma_matrix[((j-1) * (num_params-1)) + (k-1)] = (sigma_matrix[(j * num_params) + k] - ((float8)(mean_vector[j-1] * mean_vector[k-1]) / (float8) count_tuples));
            }
        }

        for(size_t i=0; i<m; i++)
            mean_vector[i] /= count_tuples;

        //normalize with / count
        //regularization if enabled should be added before
        for (size_t j = 0; j < m; j++) {
            for (size_t k = 0; k < m; k++) {
                sigma_matrix[(j*m)+k] /= count_tuples;//or / cofactor->count - num_categories
            }
        }

        //print_matrix(num_params-1, sigma_matrix);

        //invert the matrix with SVD. We can also use LU decomposition, might be faster but less stable
        int lwork = -1;
        double wkopt;
        int info;
        dgesvd( "S", "S", &m, &m, sigma_matrix, &m, sing_values, u, &m, vt, &m, &wkopt, &lwork, &info );
        double *work = (double*) palloc(((int) wkopt)*sizeof(double));
        lwork = wkopt;
        /* Compute SVD */
        dgesvd( "S", "S", &m, &m, sigma_matrix, &m, sing_values, u, &m, vt, &m, work, &lwork, &info );
        /* Check for convergence */
        if( info > 0 ) {
            elog(ERROR, "The algorithm computing SVD failed to converge." );
        }
        //we computed SVD, now calculate u=Σ-1U, where Σ is diagonal, so compute by a loop of calling BLAS ?scal function for computing product of a vector by a scalar
        //u is stored column-wise, and vt is row-wise here, thus the formula should be like u=UΣ-1
        double rcond = (1.0e-15);//Cutoff for small singular values. Singular values less than or equal to rcond * largest_singular_value are set to zero
        //consider using syevd for SVD. For positive semi-definite matrices (like covariance) SVD and eigenvalue decomp. are the same

        int incx = 1;
        for(int i=0; i<m; i++)
        {
            double ss;
            if(sing_values[i] > 1.0e-9)
                ss=1.0/sing_values[i];//1/sing. value is the inverse of diagonal matrix
            else
                ss=sing_values[i];
            dscal(&m, &ss, &u[i*m], &incx);
        }
        double determinant = 1;
        for(int i=0; i<m; i++)
            determinant *= sing_values[i];

        //elog(WARNING, "determinant %lf", determinant);
        //calculate A+=(V*)TuT, use MKL ?GEMM function to calculate matrix multiplication
        double alpha=1.0, beta=0.0;
        dgemm( "T", "T", &m, &m, &m, &alpha, vt, &m, u, &m, &beta, inva, &m);
        //elog(WARNING, "g");

        if(normalize){
            std[0] = 1;
            //make sure the inverse is / by square of std
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < m; j++) {
                    result[param_out_index + (i * m) + j] = Float8GetDatum(((double) -1 * inva[(i * m) + j] / 2) / (std[i+1]*std[j+1]));
                }
            }
        }
        else {
            //I have the inverse, save it in output
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < m; j++) {
                    result[param_out_index + (i * m) + j] = Float8GetDatum((double) -1 * inva[(i * m) + j] / 2);
                }
            }
        }
        //elog(WARNING, "h");
        param_out_index += (m*m);
        //compute product with mean
        char task = 'N';
        int increment = 1;

        dgemv(&task, &m, &m, &alpha, inva, &m, mean_vector, &increment, &beta, lin_result, &increment);
        if (normalize){//need to scale down params
            for (int j = 0; j < m; j++) {
                result[param_out_index + j] = Float8GetDatum((double) lin_result[j]/std[j+1]);
            }
        }
        else {
            for (int j = 0; j < m; j++) {
                //elog(WARNING, "mean %lf", mean_vector[j]);
                result[param_out_index + j] = Float8GetDatum((double) lin_result[j]);
            }
        }
        //elog(WARNING, "i");

        param_out_index += m;
        int row=1;
        double intercept = 0;
        dgemv(&task, &row, &m, &alpha, mean_vector, &row, lin_result, &increment, &beta, &intercept, &increment);
        /*elog(WARNING, "intercept 1 %lf", (intercept));
        for(int i=0; i<m; i++)
            elog(WARNING, " %lf * %lf", mean_vector[i], lin_result[i]);*/
        intercept = ((-1)*intercept/(double)2) - (log(determinant)/(double)2) + log(count_tuples/(double)tot_tuples);
        //elog(WARNING, "intercept 1 %lf", (log(determinant)/(double)2));
        //elog(WARNING, "intercept 1 %lf", log(sum_vector[0]/(double)tot_tuples));
        result[param_out_index] = Float8GetDatum((double ) intercept);
        param_out_index++;
        //elog(WARNING, "l");
    }

    if(normalize){
        //save means
        for (int j = 0; j < m; j++) {
            result[param_out_index+j] = Float8GetDatum(means[j+1]);
        }
        param_out_index += m;
    }

    assert(res_size == param_out_index);

    ArrayType *a = construct_array(result, res_size, FLOAT8OID, sizeof(float8), true, TYPALIGN_INT);
    PG_RETURN_ARRAYTYPE_P(a);
}

/**
 * Generates prediction using QDA
 * @param fcinfo Parameters
 * @param fcinfo numerical features
 * @param fcinfo categorical features
 * @return
 */
PG_FUNCTION_INFO_V1(qda_predict);
Datum qda_predict(PG_FUNCTION_ARGS)
{
    //elog(WARNING, "Predicting...");
    //make sure encoding (order) of features is the same of cofactor and input data
    ArrayType *means_covariance = PG_GETARG_ARRAYTYPE_P(0);
    ArrayType *feats_numerical = PG_GETARG_ARRAYTYPE_P(1);
    ArrayType *feats_categorical = PG_GETARG_ARRAYTYPE_P(2);
    bool normalize = PG_GETARG_BOOL(3);
    //ArrayType *max_feats_categorical = PG_GETARG_ARRAYTYPE_P(3);
    //elog(WARNING, "a");
    Oid arrayElementType1 = ARR_ELEMTYPE(means_covariance);
    Oid arrayElementType2 = ARR_ELEMTYPE(feats_numerical);
    Oid arrayElementType3 = ARR_ELEMTYPE(feats_categorical);
    //Oid arrayElementType4 = ARR_ELEMTYPE(max_feats_categorical);

    int16 arrayElementTypeWidth1, arrayElementTypeWidth2, arrayElementTypeWidth3;
    bool arrayElementTypeByValue1, arrayElementTypeByValue2, arrayElementTypeByValue3;
    Datum *arrayContent1, *arrayContent2, *arrayContent3;
    bool *arrayNullFlags1, *arrayNullFlags2, *arrayNullFlags3;
    int arrayLength1, arrayLength2, arrayLength3;
    char arrayElementTypeAlignmentCode1, arrayElementTypeAlignmentCode2, arrayElementTypeAlignmentCode3;
    //elog(WARNING, "b");

    get_typlenbyvalalign(arrayElementType1, &arrayElementTypeWidth1, &arrayElementTypeByValue1, &arrayElementTypeAlignmentCode1);
    deconstruct_array(means_covariance, arrayElementType1, arrayElementTypeWidth1, arrayElementTypeByValue1, arrayElementTypeAlignmentCode1,
                      &arrayContent1, &arrayNullFlags1, &arrayLength1);

    get_typlenbyvalalign(arrayElementType2, &arrayElementTypeWidth2, &arrayElementTypeByValue2, &arrayElementTypeAlignmentCode2);
    deconstruct_array(feats_numerical, arrayElementType2, arrayElementTypeWidth2, arrayElementTypeByValue2, arrayElementTypeAlignmentCode2,
                      &arrayContent2, &arrayNullFlags2, &arrayLength2);

    get_typlenbyvalalign(arrayElementType3, &arrayElementTypeWidth3, &arrayElementTypeByValue3, &arrayElementTypeAlignmentCode3);
    deconstruct_array(feats_categorical, arrayElementType3, arrayElementTypeWidth3, arrayElementTypeByValue3, arrayElementTypeAlignmentCode3,
                      &arrayContent3, &arrayNullFlags3, &arrayLength3);

    int n_classes = DatumGetFloat8(arrayContent1[0]);
    int size_idxs = DatumGetFloat8(arrayContent1[1]);
    int one_hot_size = 0;
    uint64_t *cat_vars_idxs;
    uint64_t *cat_vars;
    //elog(WARNING, "c");
    size_t start_params_idx=2;

    //extract categorical values and indices
    if (size_idxs > 0) {
        cat_vars_idxs = (uint64_t *) palloc0(sizeof(uint64_t) * (size_idxs));//max. size
        for (size_t i = 0; i < size_idxs; i++)
            cat_vars_idxs[i] = DatumGetFloat8(arrayContent1[i + 2]);

        one_hot_size = cat_vars_idxs[size_idxs - 1];
        cat_vars = (uint64_t *) palloc(sizeof(uint64_t) * one_hot_size);//max. size
        for (size_t i = 0; i < cat_vars_idxs[size_idxs - 1]; i++)
            cat_vars[i] = DatumGetFloat8(arrayContent1[i + 2 + size_idxs]);

        start_params_idx=2+cat_vars_idxs[size_idxs-1]+size_idxs;
    }

    //skip classes output
    size_t start_out_classes = start_params_idx;
    start_params_idx += n_classes;

    size_t n_params = one_hot_size + arrayLength2;
    double *features = (double *)palloc0(sizeof(double) * (n_params));
    double *quad_matrix = (double *)palloc0(sizeof(double) * (n_params*n_params));
    double *lin_matrix = (double *)palloc0(sizeof(double) * (n_params*n_params));
    double *res_matmul = (double *)palloc0(sizeof(double) * (n_params));

    int m = n_params;

    //copy features

    for (int i = 0; i < arrayLength2; i++) {
        features[i] = DatumGetFloat8(arrayContent2[i]);
        //elog(WARNING, "Feature %d : %lf", i, features[i]);
    }

    for(int i=0; i<arrayLength3; i++){//categorical feats (builds 1-hot encoded vector)
        int class = DatumGetInt64(arrayContent3[i]);
        size_t index = find_in_array(class, cat_vars, cat_vars_idxs[i], cat_vars_idxs[i+1]);
        //elog(WARNING, "1-hot class %d -> index %d (from %d to %d)", class, index, cat_vars_idxs[i], cat_vars_idxs[i+1]);
        if (index < cat_vars_idxs[i+1])
            features[index+arrayLength2] = 1;
    }

    if(normalize) {
        size_t offset_params = start_params_idx + (((n_params*n_params) + n_params + 1)*n_classes);
        for (int i = 0; i < arrayLength2; i++) {
            features[i] -= DatumGetFloat8(arrayContent1[offset_params + i]);
            //elog(WARNING, "normalize removing : %lf", DatumGetFloat8(arrayContent1[offset_params + i]));
        }

        if(size_idxs > 0) {
            for (int i = 0; i < cat_vars_idxs[size_idxs - 1]; i++) {//categorical feats (builds 1-hot encoded vector)
                features[i + arrayLength2] -= DatumGetFloat8(arrayContent1[offset_params + arrayLength2 + i]);
            }
        }
    }


    size_t best_class = 0;
    double max_prob = -DBL_MAX;
    for(size_t i=0; i<n_classes; i++){
        //copy qda params
        for(size_t j=0; j<n_params*n_params; j++){
            quad_matrix[j] = DatumGetFloat8(arrayContent1[j + start_params_idx]);
            //elog(WARNING, "QUAD MATRIX: %lf", quad_matrix[j]);
        }
        start_params_idx += (n_params*n_params);
        for(size_t j=0; j<n_params; j++){
            lin_matrix[j] = DatumGetFloat8(arrayContent1[j + start_params_idx]);
            //elog(WARNING, "LIN MATRIX: %lf", lin_matrix[j]);
        }
        start_params_idx += (n_params);
        double intercept = DatumGetFloat8(arrayContent1[start_params_idx]);
        //elog(WARNING, "INTERCEPT: %lf", intercept);

        start_params_idx++;

        //compute probability of given class with matrix multiplication

        char task = 'N';
        int increment = 1;
        double alpha=1.0, beta=0.0;

        //quad_matrix * features
        dgemv(&task, &m, &m, &alpha, quad_matrix, &m, features, &increment, &beta, res_matmul, &increment);
        int row=1;
        double res_prob_1 = 0;
        //(quad_matrix * features)*features
        dgemv(&task, &row, &m, &alpha, res_matmul, &row, features, &increment, &beta, &res_prob_1, &increment);

        //lin_matrix * features
        double res_prob_2 = 0;
        dgemv(&task, &row, &m, &alpha, lin_matrix, &row, features, &increment, &beta, &res_prob_2, &increment);

        double total_prob = intercept + res_prob_1 + res_prob_2;
        //elog(WARNING, "prob: %lf intercept %lf prob 1 %lf prob 2 %lf", total_prob, intercept, res_prob_1, res_prob_2);

        if (total_prob > max_prob){
            max_prob = total_prob;
            best_class = i;
        }
    }

    int actual_class = (int) DatumGetFloat8(arrayContent1[start_out_classes + best_class]);

    PG_RETURN_INT64(actual_class);
}
