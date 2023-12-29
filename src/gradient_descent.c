#include "cofactor.h"
#include "relation.h"
#include <postgres.h>
#include <fmgr.h>
#include <catalog/pg_type.h>
#include <utils/array.h>
#include <math.h>
#include <assert.h>

#include <utils/array.h>
#include <utils/lsyscache.h>
#include <sys/time.h>
#include <limits.h>

static bool random_seed_set = false;
static int random_seed = 0;

#define PI         3.14159265   // The value of pi


extern void dgemv (char *TRANSA, int *M, int* N, double *ALPHA,
                   double *A, int *LDA, double *X, int *INCX, double *BETA, double *Y, int *INCY);

void print_matrix(size_t sz, const double *m)
{
    for (size_t i = 0; i < sz; i++)
    {
        for(size_t j = 0; j < sz; j++)
        {
            elog(DEBUG3, "%zu, %zu -> %f", i, j, m[(i * sz) + j]);
        }
    }
}

void compute_gradient(size_t num_params, size_t label_idx,
                      const double *sigma, const double *params,
        /* out */ double *grad)
{
    if (sigma[0] == 0.0) return;

    /* Compute Sigma * Theta */
    for (size_t i = 0; i < num_params; i++)
    {
        grad[i] = 0.0;
        for (size_t j = 0; j < num_params; j++)
        {
            grad[i] += sigma[(i * num_params) + j] * params[j];
        }
        grad[i] /= sigma[0]; // count
    }
    grad[label_idx] = 0.0;
}

double compute_error(size_t num_params, const double *sigma,
                     const double *params, const double lambda)
{
    if (sigma[0] == 0.0) return 0.0;

    double error = 0.0;

    /* Compute 1/N * Theta^T * Sigma * Theta */
    for (size_t i = 0; i < num_params; i++)
    {
        double tmp = 0.0;
        for (size_t j = 0; j < num_params; j++)
        {
            tmp += sigma[(i * num_params) + j] * params[j];
        }
        error += params[i] * tmp;
    }
    error /= sigma[0]; // count

    /* Add the regulariser to the error */
    double param_norm = 0.0;
    for (size_t i = 1; i < num_params; i++)
    {
        param_norm += params[i] * params[i];
    }
    param_norm -= 1; // param_norm -= params[LABEL_IDX] * params[LABEL_IDX];
    error += lambda * param_norm;

    return error / 2;
}

inline double compute_step_size(double step_size, int num_params,
                                const double *params, const double *prev_params,
                                const double *grad, const double *prev_grad)
{
    double DSS = 0.0, GSS = 0.0, DGS = 0.0;

    for (int i = 0; i < num_params; i++)
    {
        double paramDiff = params[i] - prev_params[i];
        double gradDiff = grad[i] - prev_grad[i];

        DSS += paramDiff * paramDiff;
        GSS += gradDiff * gradDiff;
        DGS += paramDiff * gradDiff;
    }

    if (DGS == 0.0 || GSS == 0.0)
        return step_size;

    double Ts = DSS / DGS;
    double Tm = DGS / GSS;

    if (Tm < 0.0 || Ts < 0.0)
        return step_size;

    return (Tm / Ts > 0.5) ? Tm : Ts - 0.5 * Tm;
}

/**
 * Implements linear regression in PostgreSQL
 * @param cofactor input aggregates
 * @param label label column position
 * @param step_size learning rate
 * @param lambda regularization value
 * @param max_num_iterations max. num. of learning iterations
 * @param compute_variance if 0, does not compute variance, otherwise last parameter returned is variance
 * @return learned parameters: n. cat. columns, cat columns idxs, unique cat columns values, parameters, means (if standardize), variance
 */
PG_FUNCTION_INFO_V1(ridge_linear_regression);
Datum ridge_linear_regression(PG_FUNCTION_ARGS)
{
    const cofactor_t *cofactor = (const cofactor_t *)PG_GETARG_VARLENA_P(0);
    size_t label = PG_GETARG_INT64(1);
    double step_size = PG_GETARG_FLOAT8(2);
    double lambda = PG_GETARG_FLOAT8(3);
    int max_num_iterations = PG_GETARG_INT64(4);
    bool compute_variance = PG_GETARG_BOOL(5);
    bool normalize = PG_GETARG_BOOL(6);

    if (cofactor->num_continuous_vars <= label) {
        elog(ERROR, "label ID >= number of continuous attributes");
        PG_RETURN_NULL();
    }

    //size_t num_params = sizeof_sigma_matrix(cofactor, -1);
    uint64_t *cat_array = NULL;
    uint32_t *cat_vars_idxs = NULL;
    size_t num_params = n_cols_1hot_expansion(&cofactor, 1, &cat_vars_idxs, &cat_array, 0);//tot columns include label as well

    elog(DEBUG3, "num_params = %zu", num_params);

    double *grad = (double *)palloc0(sizeof(double) * num_params);
    double *prev_grad = (double *)palloc0(sizeof(double) * num_params);
    double *learned_coeff = (double *)palloc(sizeof(double) * num_params);
    double *prev_learned_coeff = (double *)palloc0(sizeof(double) * num_params);
    double *sigma = (double *)palloc0(sizeof(double) * num_params * num_params);
    double *update = (double *)palloc0(sizeof(double) * num_params);

    build_sigma_matrix(cofactor, num_params, -1, cat_array, cat_vars_idxs, 0, sigma);
    double *means = NULL;
    double *std = NULL;
    if (normalize){
        means = (double *)palloc0(sizeof(double) * num_params);
        std = (double *)palloc0(sizeof(double) * num_params);
        standardize(sigma, num_params, means, std);
        for(size_t i=0; i<num_params; i++)
            elog(DEBUG3, "std: %lf", std[i]);//0 variance for first col
    }

    print_matrix(num_params, sigma);

    for (size_t i = 0; i < num_params; i++)
    {
        learned_coeff[i] = 0; // ((double) (rand() % 800 + 1) - 400) / 100;
    }

    label += 1;     // index 0 corresponds to intercept
    prev_learned_coeff[label] = -1;
    learned_coeff[label] = -1;

    compute_gradient(num_params, label, sigma, learned_coeff, grad);

    double gradient_norm = grad[0] * grad[0]; // bias
    for (size_t i = 1; i < num_params; i++)
    {
        double upd = grad[i] + lambda * learned_coeff[i];
        gradient_norm += upd * upd;
    }
    gradient_norm -= lambda * lambda; // label correction
    double first_gradient_norm = sqrt(gradient_norm);

    double prev_error = compute_error(num_params, sigma, learned_coeff, lambda);

    size_t num_iterations = 1;
    do
    {
        // Update parameters and compute gradient norm
        update[0] = grad[0];
        gradient_norm = update[0] * update[0];
        prev_learned_coeff[0] = learned_coeff[0];
        prev_grad[0] = grad[0];
        learned_coeff[0] = learned_coeff[0] - step_size * update[0];
        double dparam_norm = update[0] * update[0];

        for (size_t i = 1; i < num_params; i++)
        {
            update[i] = grad[i] + lambda * learned_coeff[i];
            gradient_norm += update[i] * update[i];
            prev_learned_coeff[i] = learned_coeff[i];
            prev_grad[i] = grad[i];
            learned_coeff[i] = learned_coeff[i] - step_size * update[i];
            dparam_norm += update[i] * update[i];
        }
        learned_coeff[label] = -1;
        gradient_norm -= lambda * lambda; // label correction
        dparam_norm = step_size * sqrt(dparam_norm);

        double error = compute_error(num_params, sigma, learned_coeff, lambda);

        /* Backtracking Line Search: Decrease step_size until condition is satisfied */
        size_t backtracking_steps = 0;
        while (error > prev_error - (step_size / 2) * gradient_norm && backtracking_steps < 500)
        {
            step_size /= 2; // Update parameters based on the new step_size.

            dparam_norm = 0.0;
            for (size_t i = 0; i < num_params; i++)
            {
                double newp = prev_learned_coeff[i] - step_size * update[i];
                double dp = learned_coeff[i] - newp;
                learned_coeff[i] = newp;
                dparam_norm += dp * dp;
            }
            dparam_norm = sqrt(dparam_norm);
            learned_coeff[label] = -1;
            error = compute_error(num_params, sigma, learned_coeff, lambda);
            backtracking_steps++;
        }

        /* Normalized residual stopping condition */
        gradient_norm = sqrt(gradient_norm);
        if (dparam_norm < 1e-20 ||
            gradient_norm / (first_gradient_norm + 0.001) < 1e-8)
        {
            break;
        }
        compute_gradient(num_params, label, sigma, learned_coeff, grad);

        step_size = compute_step_size(step_size, num_params, learned_coeff, prev_learned_coeff, grad, prev_grad);

        elog(DEBUG3, "error = %f", error);
        prev_error = error;
        num_iterations++;
    } while (num_iterations < max_num_iterations);

    elog(DEBUG1, "num_iterations = %zu", num_iterations);
    elog(DEBUG1, "error = %lf", prev_error);
    double variance = 0;
    if(compute_variance){
        //compute variance for stochastic linear regression

        char task = 'N';
        double alpha = 1;
        int increment = 1;
        double beta = 0;
        int int_num_params = num_params;
        double *res = (double *)palloc0(sizeof(double) * num_params);
        //sigma * (X^T * X)
        learned_coeff[label] = -1;
        dgemv(&task, &int_num_params, &int_num_params, &alpha, sigma, &int_num_params, learned_coeff, &increment, &beta, res, &increment);
        int size_row = 1;
        //(sigma * (X^T * X))*sigma^T
        dgemv(&task, &size_row, &int_num_params, &alpha, res, &size_row, learned_coeff, &increment, &beta, &variance, &increment);

        //if standardize compensate
        if (normalize){
            variance *= pow(std[label], 2);
        }

        variance /= (double) cofactor->count;
    }
    //make sure variance is computed before editing params

    if (normalize){//rescale coeff. because of standardized dataset
        for (size_t i=1; i<num_params; i++)
            learned_coeff[i] = (learned_coeff[i] / std[i]) * std[label];

        learned_coeff[0] = (learned_coeff[0]* std[label]) + means[label];
    }

    size_t learned_params_size = num_params;
    if (cofactor->num_categorical_vars > 0)//there are categorical variables, store unique vars and idxs
        num_params += cat_vars_idxs[cofactor->num_categorical_vars] + cofactor->num_categorical_vars;

    if(normalize){
        num_params += (learned_params_size - 2);//need to store mean for each column (except label and constant term)
    }

    num_params --;//no label element

    elog(DEBUG3, "num_params = %lu size 1 %d size 2 %d", num_params, cat_vars_idxs[cofactor->num_categorical_vars], cofactor->num_categorical_vars);

    // export params to pgpsql
    Datum *d;
    if(!compute_variance)
        d = (Datum *)palloc(sizeof(Datum) * num_params);
    else
        d = (Datum *)palloc(sizeof(Datum) * (num_params + 1));


    d[0] = Float8GetDatum((float)cofactor->num_categorical_vars);//size categorical columns

    int idx_output = 1;
    if (cofactor->num_categorical_vars > 0) {//there are categorical variables
        //store categorical value indices of cat. columns (without label)
        for (size_t i = 0; i < cofactor->num_categorical_vars + 1; i++) {
            d[idx_output] = Float8GetDatum((float) cat_vars_idxs[i]);
            idx_output++;
        }
        for (size_t i = 0; i < cat_vars_idxs[cofactor->num_categorical_vars]; i++) {
            d[idx_output] = Float8GetDatum((float) cat_array[i]);
            idx_output++;
        }
    }

    elog(DEBUG3, "idx_output %d", idx_output);

    for (size_t i = 0; i < label; i++){//the first element is the constant term, so label element is label +1
        d[idx_output] = Float8GetDatum(learned_coeff[i]);
        idx_output++;
    }
    for (size_t i = label+1; i < learned_params_size; i++){
        d[idx_output] = Float8GetDatum(learned_coeff[i]);
        idx_output++;
    }

    if (normalize) {
        for (size_t i = 1; i < label; i++) {
            d[idx_output] = Float8GetDatum(means[i]);
            idx_output++;
        }
        for (size_t i = label+1; i < learned_params_size; i++) {
            d[idx_output] = Float8GetDatum(means[i]);
            idx_output++;
        }
    }

    if(compute_variance){
        d[idx_output] = Float8GetDatum(sqrt(variance));//returns std instead of variance
        idx_output++;
    }

    elog(DEBUG3, "idx_output %d", idx_output);

    pfree(grad);
    pfree(prev_grad);
    pfree(learned_coeff);
    pfree(prev_learned_coeff);
    pfree(sigma);
    pfree(update);

    ArrayType *a = construct_array(d, idx_output, FLOAT8OID, sizeof(float8), true, 'd');
    PG_RETURN_ARRAYTYPE_P(a);
}


/**
 * Given parameters, train a linear regression
 * @param fcinfo
 * @return
 */
PG_FUNCTION_INFO_V1(ridge_linear_regression_from_params);
Datum ridge_linear_regression_from_params(PG_FUNCTION_ARGS)
{
    ArrayType *cofactor_vals = PG_GETARG_ARRAYTYPE_P(0);
    Oid arrayElementType1 = ARR_ELEMTYPE(cofactor_vals);
    int16 arrayElementTypeWidth1;
    bool arrayElementTypeByValue1;
    Datum *arrayContent1;
    bool *arrayNullFlags1;
    int arrayLength1;
    char arrayElementTypeAlignmentCode1;

    get_typlenbyvalalign(arrayElementType1, &arrayElementTypeWidth1, &arrayElementTypeByValue1, &arrayElementTypeAlignmentCode1);
    deconstruct_array(cofactor_vals, arrayElementType1, arrayElementTypeWidth1, arrayElementTypeByValue1, arrayElementTypeAlignmentCode1,
                      &arrayContent1, &arrayNullFlags1, &arrayLength1);

    size_t label = PG_GETARG_INT64(1);
    double step_size = PG_GETARG_FLOAT8(2);
    double lambda = PG_GETARG_FLOAT8(3);
    int max_num_iterations = PG_GETARG_INT64(4);

    int num_params = (-1 + sqrt(1+(4*arrayLength1)))/2;

    //elog(DEBUG5, "num_params = %zu", num_params);

    double *grad = (double *)palloc0(sizeof(double) * num_params);
    double *prev_grad = (double *)palloc0(sizeof(double) * num_params);
    double *learned_coeff = (double *)palloc(sizeof(double) * num_params);
    double *prev_learned_coeff = (double *)palloc0(sizeof(double) * num_params);
    double *sigma = (double *)palloc0(sizeof(double) * num_params * num_params);
    double *update = (double *)palloc0(sizeof(double) * num_params);

    size_t row = 0;
    size_t col = 0;
    //build sigma matrix
    for (size_t i = 0; i < arrayLength1; i++){
        sigma[(num_params*row)+col] = (double) DatumGetFloat8(arrayContent1[i]);
        sigma[(num_params*col)+row] = (double) DatumGetFloat8(arrayContent1[i]);
        //elog(WARNING, "val: = %lf, row %d, col %d", (double) DatumGetFloat4(arrayContent1[i]), row, col);
        col++;
        if (col%num_params == 0){
            row++;
            col = row;
        }
    }
    print_matrix(num_params, sigma);

    for (size_t i = 0; i < num_params; i++)
    {
        learned_coeff[i] = 0; // ((double) (rand() % 800 + 1) - 400) / 100;
    }

    label += 1;     // index 0 corresponds to intercept
    prev_learned_coeff[label] = -1;
    learned_coeff[label] = -1;

    compute_gradient(num_params, label, sigma, learned_coeff, grad);

    double gradient_norm = grad[0] * grad[0]; // bias
    for (size_t i = 1; i < num_params; i++)
    {
        double upd = grad[i] + lambda * learned_coeff[i];
        gradient_norm += upd * upd;
    }
    gradient_norm -= lambda * lambda; // label correction
    double first_gradient_norm = sqrt(gradient_norm);

    double prev_error = compute_error(num_params, sigma, learned_coeff, lambda);

    size_t num_iterations = 1;
    do
    {
        // Update parameters and compute gradient norm
        update[0] = grad[0];
        gradient_norm = update[0] * update[0];
        prev_learned_coeff[0] = learned_coeff[0];
        prev_grad[0] = grad[0];
        learned_coeff[0] = learned_coeff[0] - step_size * update[0];
        double dparam_norm = update[0] * update[0];

        for (size_t i = 1; i < num_params; i++)
        {
            update[i] = grad[i] + lambda * learned_coeff[i];
            gradient_norm += update[i] * update[i];
            prev_learned_coeff[i] = learned_coeff[i];
            prev_grad[i] = grad[i];
            learned_coeff[i] = learned_coeff[i] - step_size * update[i];
            dparam_norm += update[i] * update[i];
        }
        learned_coeff[label] = -1;
        gradient_norm -= lambda * lambda; // label correction
        dparam_norm = step_size * sqrt(dparam_norm);

        double error = compute_error(num_params, sigma, learned_coeff, lambda);

        /* Backtracking Line Search: Decrease step_size until condition is satisfied */
        size_t backtracking_steps = 0;
        while (error > prev_error - (step_size / 2) * gradient_norm && backtracking_steps < 500)
        {
            step_size /= 2; // Update parameters based on the new step_size.

            dparam_norm = 0.0;
            for (size_t i = 0; i < num_params; i++)
            {
                double newp = prev_learned_coeff[i] - step_size * update[i];
                double dp = learned_coeff[i] - newp;
                learned_coeff[i] = newp;
                dparam_norm += dp * dp;
            }
            dparam_norm = sqrt(dparam_norm);
            learned_coeff[label] = -1;
            error = compute_error(num_params, sigma, learned_coeff, lambda);
            backtracking_steps++;
        }

        /* Normalized residual stopping condition */
        gradient_norm = sqrt(gradient_norm);
        if (dparam_norm < 1e-20 ||
            gradient_norm / (first_gradient_norm + 0.001) < 1e-8)
        {
            break;
        }
        compute_gradient(num_params, label, sigma, learned_coeff, grad);

        step_size = compute_step_size(step_size, num_params, learned_coeff, prev_learned_coeff, grad, prev_grad);

        elog(DEBUG5, "error = %f", error);
        prev_error = error;
        num_iterations++;
    } while (num_iterations < 1000 || num_iterations < max_num_iterations);

    elog(DEBUG1, "num_iterations = %zu", num_iterations);
    elog(DEBUG1, "error = %lf", prev_error);

    // export params to pgpsql
    Datum *d = (Datum *)palloc(sizeof(Datum) * num_params);
    for (int i = 0; i < num_params; i++)
    {
        d[i] = Float8GetDatum(learned_coeff[i]);
        elog(DEBUG2, "learned_coeff[%d] = %f", i, learned_coeff[i]);
    }
    pfree(grad);
    pfree(prev_grad);
    pfree(learned_coeff);
    pfree(prev_learned_coeff);
    pfree(sigma);
    //pfree(update);
    ArrayType *a = construct_array(d, num_params, FLOAT8OID, sizeof(float8), true, 'd');
    PG_RETURN_ARRAYTYPE_P(a);
}

PG_FUNCTION_INFO_V1(linregr_predict);
Datum linregr_predict(PG_FUNCTION_ARGS) {
    ArrayType *params = PG_GETARG_ARRAYTYPE_P(0);//result of train
    ArrayType *cont = PG_GETARG_ARRAYTYPE_P(1);//num columns
    ArrayType *cat = PG_GETARG_ARRAYTYPE_P(2);//cat columns
    bool noise = PG_GETARG_BOOL(3);
    bool normalize = PG_GETARG_BOOL(4);


    Oid arrayElementType1 = ARR_ELEMTYPE(params);
    Oid arrayElementType2 = ARR_ELEMTYPE(cont);
    Oid arrayElementType3 = ARR_ELEMTYPE(cat);


    //extract arrays into C format

    int16 arrayElementTypeWidth1, arrayElementTypeWidth2, arrayElementTypeWidth3;
    bool arrayElementTypeByValue1, arrayElementTypeByValue2, arrayElementTypeByValue3;
    Datum *arrayContent1, *arrayContent2, *arrayContent3;
    bool *arrayNullFlags1, *arrayNullFlags2, *arrayNullFlags3;
    int arrayLength1, arrayLength2, arrayLength3;
    char arrayElementTypeAlignmentCode1, arrayElementTypeAlignmentCode2, arrayElementTypeAlignmentCode3;

    get_typlenbyvalalign(arrayElementType1, &arrayElementTypeWidth1, &arrayElementTypeByValue1, &arrayElementTypeAlignmentCode1);
    deconstruct_array(params, arrayElementType1, arrayElementTypeWidth1, arrayElementTypeByValue1, arrayElementTypeAlignmentCode1,
                      &arrayContent1, &arrayNullFlags1, &arrayLength1);

    get_typlenbyvalalign(arrayElementType2, &arrayElementTypeWidth2, &arrayElementTypeByValue2, &arrayElementTypeAlignmentCode2);
    deconstruct_array(cont, arrayElementType2, arrayElementTypeWidth2, arrayElementTypeByValue2, arrayElementTypeAlignmentCode2,
                      &arrayContent2, &arrayNullFlags2, &arrayLength2);

    get_typlenbyvalalign(arrayElementType3, &arrayElementTypeWidth3, &arrayElementTypeByValue3, &arrayElementTypeAlignmentCode3);
    deconstruct_array(cat, arrayElementType3, arrayElementTypeWidth3, arrayElementTypeByValue3, arrayElementTypeAlignmentCode3,
                      &arrayContent3, &arrayNullFlags3, &arrayLength3);

    int n_cat_columns = (int) DatumGetFloat8(arrayContent1[0]);
    int max_cat_vars_idx = 0;
    size_t start_params = 1 + n_cat_columns;//skip n. cat. columns and idx_cat_vals is n.cols

    if (n_cat_columns > 0) {
        max_cat_vars_idx = (int) DatumGetFloat8(arrayContent1[start_params]);
        start_params += max_cat_vars_idx +1;
    }

    double result = DatumGetFloat8(arrayContent1[start_params]);//init with intercept
    elog(DEBUG3, "intercept %lf", result);

    if (normalize){
        for (size_t i = 0; i < arrayLength2; i++) {//build num. pred.
            result += ((double) DatumGetFloat8(arrayContent1[i + start_params + 1]) * ((double) DatumGetFloat8(
                    arrayContent2[i]) - DatumGetFloat8(arrayContent1[1 + arrayLength2 + max_cat_vars_idx + start_params + i])));
            elog(DEBUG3, "normalize: %lf - %lf", DatumGetFloat8(arrayContent2[i]), DatumGetFloat8(arrayContent1[1 + arrayLength2 + max_cat_vars_idx + start_params + i]));
        }
    }
    else {
        for (size_t i = 0; i < arrayLength2; i++) {//build num. pred.
            result += ((double) DatumGetFloat8(arrayContent1[i + start_params + 1]) * (double) DatumGetFloat8(
                    arrayContent2[i]));//re-build index vector (begin:end of each cat. column)
        }
    }

    for(size_t i=0; i<arrayLength3; i++){//build cat. pred.
        int class = DatumGetInt64(arrayContent3[i]);
        //search for class in cat array
        int start = (int) DatumGetFloat8(arrayContent1[1+i]);;
        int end = (int) DatumGetFloat8(arrayContent1[2+i]);

        size_t index = start;
        while (index < end)
        {
            if ((int) DatumGetFloat8(arrayContent1[index+2+n_cat_columns]) == class)
                break;
            index++;
        }
        elog(DEBUG3, "search val %d start %d end %d found %lu position %lu", class, start, end, index, index + start_params + arrayLength2);
        if (normalize){
            for(size_t j=start; j<index; j++){
                result += (double) (DatumGetFloat8(arrayContent1[j + start_params + arrayLength2 + 1]) * (0-DatumGetFloat8(arrayContent1[1 + (2 * arrayLength2) + max_cat_vars_idx + start_params + j])));
                elog(DEBUG3, "normalize cat: %lf ", DatumGetFloat8(arrayContent1[1 + (2 * arrayLength2) + max_cat_vars_idx + start_params + j]));
            }
            result += (double) (DatumGetFloat8(arrayContent1[index + start_params + arrayLength2 +
                                                            1]) * (1-DatumGetFloat8(arrayContent1[1 + (2 * arrayLength2) + max_cat_vars_idx + start_params + index])));
            for(size_t j=index+1; j<end; j++){
                result += (double) (DatumGetFloat8(arrayContent1[j + start_params + arrayLength2 + 1]) * (0-DatumGetFloat8(arrayContent1[1 + (2 * arrayLength2) + max_cat_vars_idx + start_params + j])));
                elog(DEBUG3, "normalize cat: %lf",  DatumGetFloat8(arrayContent1[1 + (2 * arrayLength2) + max_cat_vars_idx + start_params + j]));
            }

        }
        else {
            result += (double) DatumGetFloat8(arrayContent1[index + start_params + arrayLength2 +
                                                            1]);//skip continous vars, class idx and unique class
        }
    }

    if (noise){

        if (!random_seed_set){
            elog(DEBUG3, "set seed");
            FILE *istream = fopen("/dev/urandom", "rb");
            assert(istream);
            unsigned long seed = 0;
            for (unsigned i = 0; i < sizeof seed; i++) {
                seed *= (UCHAR_MAX + 1);
                int ch = fgetc(istream);
                assert(ch != EOF);
                seed += (unsigned) ch;
            }
            fclose(istream);
            srandom(seed);
            random_seed_set=true;
        }

        double u1, u2;
        do{
            u1 = random()/(double)(RAND_MAX+1.0);
        }
        while (u1 == 0);
        u2 = random()/(double)(RAND_MAX+1.0);
        //compute z0 and z1
        double mag = DatumGetFloat8(arrayContent1[arrayLength1-1]) * sqrt(-2.0 * log(u1)) * cos(2* PI * u2);
        result += mag;
    }
    PG_RETURN_FLOAT8(result);
}

