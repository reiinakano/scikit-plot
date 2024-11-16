#include "special_wrappers.h"
#include "log_exp.h"

// #include "special/bessel.h"

// using namespace std;

// namespace {

// complex<double> to_complex(npy_cdouble z) {
//     union {
//         npy_cdouble cvalue;
//         complex<double> value;
//     } z_union{z};
//     return z_union.value;
// }

// npy_cdouble to_ccomplex(complex<double> z) {
//     union {
//         complex<double> value;
//         npy_cdouble cvalue;
//     } z_union{z};
//     return z_union.cvalue;
// }

// } // namespace



float special_expitf(float x) { return special::expit(x); };
double special_expit(double x) { return special::expit(x); };
npy_longdouble special_expitl(npy_longdouble x) { return special::expit(x); };

float special_log_expitf(float x) { return special::log_expit(x); };
double special_log_expit(double x) { return special::log_expit(x); };
npy_longdouble special_log_expitl(npy_longdouble x) { return special::log_expit(x); };

float special_logitf(float x) { return special::logit(x); };
double special_logit(double x) { return special::logit(x); };
npy_longdouble special_logitl(npy_longdouble x) { return special::logit(x); };