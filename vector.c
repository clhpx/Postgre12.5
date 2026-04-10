/*
 * src/tutorial/vector.c
 */

#include "postgres.h"
#include "fmgr.h"
#include "utils/builtins.h"
#include "common/shortest_dec.h"
#include <math.h>
#include <ctype.h>
#include <float.h>
#include <errno.h>
#include <string.h>

PG_MODULE_MAGIC;

#define MAX_DIM 1024

typedef struct
{
    int32   vl_len_;
    int32   ndim;
    float4  data[FLEXIBLE_ARRAY_MEMBER];
} Vector;

#define VECTOR_HDRSIZE offsetof(Vector, data)

static void
parse_vector_string(const char *str, float4 **values, int *ndim)
{
    const char *p = str;
    char token_buf[64];
    int token_len;
    char *endptr;
    float val;
    int dim = 0;
    int capacity = MAX_DIM;
    float4 *vals = (float4 *) palloc(capacity * sizeof(float4));

    /* No leading whitespace allowed before '{' */
    if (*p != '{')
        ereport(ERROR,
                (errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
                 errmsg("vector must start with '{'")));
    p++; /* skip '{' */

    if (*p == '}')
        ereport(ERROR,
                (errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
                 errmsg("empty vector not allowed")));

    while (*p && *p != '}')
    {
        /* Skip whitespace before a number (allowed) */
        while (*p && isspace((unsigned char)*p))
            p++;
        if (*p == ',' || *p == '}')
        {
            if (*p == ',')
                ereport(ERROR,
                        (errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
                         errmsg("empty element in vector")));
            break;
        }

        /* Extract token: stop at comma, '}', or whitespace */
        token_len = 0;
        while (*p && *p != ',' && *p != '}')
        {
            if (isspace((unsigned char)*p))
                ereport(ERROR,
                        (errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
                         errmsg("extra whitespace after element value")));
            if (token_len < (int)sizeof(token_buf)-1)
                token_buf[token_len++] = *p;
            p++;
        }
        token_buf[token_len] = '\0';

        /* Convert token to float */
        errno = 0;
        val = strtof(token_buf, &endptr);
        if (errno == ERANGE || val > FLT_MAX || val < -FLT_MAX)
            ereport(ERROR,
                    (errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
                     errmsg("value out of range: \"%s\"", token_buf)));
        if (endptr == token_buf || *endptr != '\0')
            ereport(ERROR,
                    (errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
                     errmsg("invalid float value: \"%s\"", token_buf)));

        if (dim >= capacity)
            ereport(ERROR,
                    (errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
                     errmsg("vector dimension exceeds maximum allowed (%d)", MAX_DIM)));

        vals[dim++] = (float4) val;

        /* Now p points to either ',' or '}' (whitespace already rejected) */
        if (*p == ',')
            p++;
        else if (*p != '}')
            ereport(ERROR,
                    (errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
                     errmsg("expected ',' or '}' after element")));
    }

    /* Ensure we are at the closing '}' */
    if (*p != '}')
        ereport(ERROR,
                (errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
                 errmsg("vector must end with '}'")));
    p++; /* skip '}' */

    /* No characters allowed after '}' */
    if (*p != '\0')
        ereport(ERROR,
                (errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
                 errmsg("extra characters after closing '}'")));

    if (dim == 0)
        ereport(ERROR,
                (errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
                 errmsg("vector must have at least one dimension")));

    *values = vals;
    *ndim = dim;
}

PG_FUNCTION_INFO_V1(vector_in);
Datum
vector_in(PG_FUNCTION_ARGS)
{
    char *str = PG_GETARG_CSTRING(0);
    Vector *result;
    float4 *values;
    int ndim;

    parse_vector_string(str, &values, &ndim);

    result = (Vector *) palloc(VECTOR_HDRSIZE + ndim * sizeof(float4));
    SET_VARSIZE(result, VECTOR_HDRSIZE + ndim * sizeof(float4));
    result->ndim = ndim;
    memcpy(result->data, values, ndim * sizeof(float4));

    pfree(values);
    PG_RETURN_POINTER(result);
}

PG_FUNCTION_INFO_V1(vector_out);
Datum
vector_out(PG_FUNCTION_ARGS)
{
    Vector *vec = (Vector *) PG_GETARG_POINTER(0);
    char *result;
    char *p;
    int i, total_len, len;
    char buf[FLOAT_SHORTEST_DECIMAL_LEN];

    total_len = 2; /* braces */
    for (i = 0; i < vec->ndim; i++)
    {
        float_to_shortest_decimal_bufn(vec->data[i], buf);
        total_len += strlen(buf);
        if (i < vec->ndim - 1)
            total_len += 1;
    }
    result = (char *) palloc(total_len + 1);
    p = result;
    *p++ = '{';
    for (i = 0; i < vec->ndim; i++)
    {
        if (i > 0)
            *p++ = ',';
        float_to_shortest_decimal_bufn(vec->data[i], p);
        p += strlen(p);
    }
    *p++ = '}';
    *p = '\0';
    PG_RETURN_CSTRING(result);
}

PG_FUNCTION_INFO_V1(vector_dim);
Datum
vector_dim(PG_FUNCTION_ARGS)
{
    Vector *vec = (Vector *) PG_GETARG_POINTER(0);
    PG_RETURN_INT32(vec->ndim);
}

PG_FUNCTION_INFO_V1(vector_l2_distance);
Datum
vector_l2_distance(PG_FUNCTION_ARGS)
{
    Vector *a = (Vector *) PG_GETARG_POINTER(0);
    Vector *b = (Vector *) PG_GETARG_POINTER(1);
    double sum = 0.0;
    double diff;
    int i;

    if (a->ndim != b->ndim)
        ereport(ERROR,
                (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
                 errmsg("vectors of different dimensions: %d vs %d", a->ndim, b->ndim)));

    for (i = 0; i < a->ndim; i++)
    {
        diff = (double)a->data[i] - (double)b->data[i];
        sum += diff * diff;
    }
    PG_RETURN_FLOAT4((float)sqrt(sum));
}

PG_FUNCTION_INFO_V1(vector_add);
Datum
vector_add(PG_FUNCTION_ARGS)
{
    Vector *a = (Vector *) PG_GETARG_POINTER(0);
    Vector *b = (Vector *) PG_GETARG_POINTER(1);
    Vector *result;
    int i;

    if (a->ndim != b->ndim)
        ereport(ERROR,
                (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
                 errmsg("vectors of different dimensions: %d vs %d", a->ndim, b->ndim)));

    result = (Vector *) palloc(VECTOR_HDRSIZE + a->ndim * sizeof(float4));
    SET_VARSIZE(result, VECTOR_HDRSIZE + a->ndim * sizeof(float4));
    result->ndim = a->ndim;
    for (i = 0; i < a->ndim; i++)
        result->data[i] = a->data[i] + b->data[i];
    PG_RETURN_POINTER(result);
}

PG_FUNCTION_INFO_V1(vector_sub);
Datum
vector_sub(PG_FUNCTION_ARGS)
{
    Vector *a = (Vector *) PG_GETARG_POINTER(0);
    Vector *b = (Vector *) PG_GETARG_POINTER(1);
    Vector *result;
    int i;

    if (a->ndim != b->ndim)
        ereport(ERROR,
                (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
                 errmsg("vectors of different dimensions: %d vs %d", a->ndim, b->ndim)));

    result = (Vector *) palloc(VECTOR_HDRSIZE + a->ndim * sizeof(float4));
    SET_VARSIZE(result, VECTOR_HDRSIZE + a->ndim * sizeof(float4));
    result->ndim = a->ndim;
    for (i = 0; i < a->ndim; i++)
        result->data[i] = a->data[i] - b->data[i];
    PG_RETURN_POINTER(result);
}