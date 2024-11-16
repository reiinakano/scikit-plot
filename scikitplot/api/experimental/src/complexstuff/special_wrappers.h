/* This file is a collection of wrappers around the
 *  Special Function  Fortran library of functions
 *  to be compiled with the other special functions in cephes
 *
 * Functions written by Shanjie Zhang and Jianming Jin.
 * Interface by
 *  Travis E. Oliphant
 */

#pragma once

// Used for system headers or headers provided by libraries.
#include <math.h>
#include <numpy/npy_math.h>

// Used for project-specific or local headers.
#include "Python.h"
#include "npy_2_complexcompat.h"

// #include "sf_error.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */


float special_expitf(float x);
double special_expit(double x);
npy_longdouble special_expitl(npy_longdouble x);

npy_double special_exprel(npy_double x);

float special_log_expitf(float x);
double special_log_expit(double x);
npy_longdouble special_log_expitl(npy_longdouble x);

float special_logitf(float x);
double special_logit(double x);
npy_longdouble special_logitl(npy_longdouble x);



#ifdef __cplusplus
} /* extern "C" */
#endif /* __cplusplus */