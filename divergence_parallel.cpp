// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::depends(BH)]]

#include <Rcpp.h>
#include <RcppEigen.h>

// Omit all the prefixes Rcpp::function_call
using namespace Rcpp; 

#include <iostream>
#include <cmath>
#include <algorithm>

// Float point high precision packages (needed for the likelihood calculations - Values too small!)
#include <boost/multiprecision/float128.hpp>
#include <boost/multiprecision/mpfr.hpp>
namespace mp = boost::multiprecision;


// ------------------------------------ Global Constants ------------------------------------
typedef Eigen::Matrix<mp::float128, Eigen::Dynamic, Eigen::Dynamic> MatrixX128;
typedef Eigen::Matrix<mp::float128, Eigen::Dynamic, 1> VectorX128; // A vector for eigen is simply a one column matrix
typedef std::vector<MatrixX128> TensorX128; // More like a list of matrix, but we call it Tensor for convenience
// unsuported in Rcpp
// typedef Eigen::Tensor<mp::float128, 3> TensorX128; // 3-dimensional tensor

typedef Eigen::Matrix<bool, Eigen::Dynamic, 1> VectorXbool; // A 0-1 vector
// ------------------------------------------------------------------------------------------



const double PI = 3.14159265358979323846;

// Event though it would be great, functions that use a template for types of variables cannot be called directly in R. If really needed we can always make an auxiliary function with fixed variable types, such as NumericVector that calls the more general ones with templates.

// [[Rcpp::export]]
void test(NumericVector v, int a = 0){
    // Rcout << RangeVector(1,4) <<"\n";
}

// Gets a vector and an interval and returns the values on that interval
template <typename vector_type>
std::vector<vector_type> splice_vector(std::vector<vector_type>& v, std::size_t start, std::size_t end){
    if (start > end || start >= v.size() || end > v.size())
        throw std::out_of_range("Invalid index.");
    
    std::vector<vector_type> spliced;
    std:copy(v.begin() + start, v.begin() + end, std::back_inserter(spliced));
    
    return spliced;
}

// Gets a vector of values and a vector of indexes and returns the values on those respective indexes
template <typename vector_type>
std::vector<vector_type> value_at(std::vector<vector_type>& v, std::size_t start, std::size_t end){
    std::vector<vector_type> spliced;
    std:copy(v.begin() + start, v.begin() + end, std::back_inserter(spliced));
    
    return spliced;
}

template <typename vector_type>
void print_vector(std::vector<vector_type>& v){
    int n = v.size();
    for(int i = 0; i < n; i++){
        if(i < n-1)
            Rcout << v[i] << " ";
        else
            Rcout << v[i] << "\n";
    }
}

template <typename vector_type>
std::vector<vector_type> copy_vector(const std::vector<vector_type>& v) {
    std::vector<vector_type> copy;
    copy = v;  // Assignment operator
    return copy;
}

// Performs element-wise exponential of a generic vector by a fixed value exponent
template <typename vector_type>
std::vector<vector_type> vector_pow(std::vector<vector_type>& v, double exponent){
    int n = v.size();
    std::vector<vector_type> new_v(n);
    for(int i = 0; i < n; i++){
        new_v[i] = pow(v[i], exponent);
    }
    return new_v;
}

// Obtain the inner product between two vectors
template <typename vector_type1, typename vector_type2>
double inner_product(std::vector<vector_type1>& v1, std::vector<vector_type2>& v2){
    // If vectors are from different sizes, just refuse to calculate anything
    if(v1.size() != v2.size())
        return 0;
    int n = v1.size();
    double S = 0;
    for(int i = 0; i < n; i++){
        S += v1[i] * v2[i];
    }
    return S;
}

// Function to take the product of all elements in a vector
template <typename vector_type>
double prod(std::vector<vector_type> v){
    double p = 1;
    for(int i = 0; i < v.size(); i++){
        p *= v[i];
    }
    return p;
}

// Performs element-wise division of a generic vector by a fixed value c
template <typename vector_type>
std::vector<vector_type> vector_divide(std::vector<vector_type>& v, double c){
    int n = v.size();
    std::vector<vector_type> new_v(n);
    for(int i = 0; i < n; i++){
        new_v[i] = v[i] / c;
    }
    return new_v;
}

// Performs element-wise division of a generic vector by a fixed value c
template <typename vector_type>
double mean(std::vector<vector_type> v){
    int n = v.size();
    double S = 0;
    for(int i = 0; i < n; i++){
        S += v[i];
    }
    return S / n;
}

// Instead of creating a Range object, create an actual list with indexes
std::vector<int> RangeVector(int i, int j){
    int v_size = j-i+1;
    std::vector<int> index(v_size);
    for(int k = 0; k < v_size; k++){
        index[k] = k+i;
    }
    return index;
}

// Obtain the new variance array (preallocated new_h), given that the kth observation has been removed from the time series
std::vector<double> h_remove_k_cpp_call(std::vector<double>& theta, std::vector<double>& y, std::vector<double> h, int k, int p, int q){
    int T = y.size();
    
    // Initialize the values of new_h by copying the elements from h
    std::vector<double> new_h = h;
    
    // If k is the last observation, the variances remain the same (it's a deep copy)
    if(k == T){
        return h;
    }
    
    // Extract the parameters
    double alpha0 = theta[ 0 ];
    std::vector<double> alpha = splice_vector(theta, 1, p+1);
    std::vector<double> beta = splice_vector(theta, p+1, theta.size());
    
    // Get the previous variance values to start the calculations
    std::vector<double> previous_ht_k = splice_vector(h, k-q, k);
    std::reverse(previous_ht_k.begin(), previous_ht_k.end());
    
    for(int t = k; t < T; t++){
        int out_index = t-k;
    
        std::vector<double> past_y = splice_vector(y, t-p, t);
        std::reverse(past_y.begin(), past_y.end());
        
        // If the removed index is in the interval o p, change y_(t-k) by h_k
        if(out_index >= 0 && out_index <= (p-1)){
            past_y[out_index] = sqrt( h[k-1] );
        }
        
        // Compute both parts of the new variance at time t
        std::vector<double> past_y2 = vector_pow(past_y, 2);
        double arch = alpha0 + inner_product( alpha, past_y2 );
        double garch = inner_product( beta, previous_ht_k );
        
        new_h[t] = arch + garch;
        
        // Removes the last element of the vector
        previous_ht_k.pop_back();
        // Insert the new calculated h_k value on the start of the vector
        previous_ht_k.insert(previous_ht_k.begin(), new_h[t]);
    }
    
    return new_h;
}

// [[Rcpp::export]]
NumericVector h_remove_k_cpp(std::vector<double>& theta, std::vector<double>& y, std::vector<double>& h, int k, int p, int q){
    std::vector<double> h_k_c = h_remove_k_cpp_call(theta, y, h, k, p, q);
    NumericVector h_k = NumericVector(h_k_c.begin(), h_k_c.end());
    return h_k;
}

std::vector<double> Lts_cpp_call(std::vector<double>& y, std::vector<double>& h){
    int T = y.size();
    std::vector<double> Lts(T);
    for(int i = 0; i < T; i++){
        Lts[i] = 1/sqrt(h[i] * 2 * PI) * exp(- pow(y[i], 2) / (2*h[i]) );
    }
    return Lts;
}

// [[Rcpp::export]]
NumericVector Lts_cpp(std::vector<double>& y, std::vector<double>& h){
    std::vector<double> Lts_c = Lts_cpp_call(y, h);
    NumericVector Lts = NumericVector(Lts_c.begin(), Lts_c.end());
    return Lts;
}

double mk_cpp_call(std::vector<double>& y, std::vector<double>& h, std::vector<double>& h_k){
    int T = y.size();
    
    // Indexes where h and h_k are actually different
    std::vector<double> tmp_index_vector;
    
    // Checks if the different values where already spotted.
    bool difference_spotted = false;
    int difference_start = 0;
    int difference_end = h.size();
        
    std::vector<double> Lts_num = Lts_cpp_call(y, h_k);
    std::vector<double> Lts_den = Lts_cpp_call(y, h);
    std::vector<double> ratio(T);
    for(int i = 0; i < T; i++){
        ratio[i] = Lts_num[i] / Lts_den[i];
    }
    
    return prod(ratio);
}

// [[Rcpp::export]]
double mk_cpp(std::vector<double>& y, std::vector<double>& h, std::vector<double>& h_k){
    return mk_cpp_call(y, h, h_k);
}

// Given a list of parameters sampled from the posterior distribution, compute the argument of the psi function for the divergence estimation
std::vector<double> psi_divergence_estimator_arguments_cpp_call(std::vector<std::vector<double>> thetas, std::vector<std::vector<double>> hs, std::vector<double> y, int k, int p, int q){
    int T = y.size();
    
    // thetas.size() = nrow(thetas)
    std::vector<double> S( thetas.size() );
    
    for(int i = 0; i < thetas.size(); i++){
        // Get the ith line from the matrices
        std::vector<double> theta = thetas[i];
        std::vector<double> h = hs[i];
        
        // Obtain the variance after removing the kth observation
        std::vector<double> h_k = h_remove_k_cpp_call(theta, y, h, k, p, q);
        
        // If k == T, m_k = 1, if not we call its respective function
        double m_k = 1;
        if(k < T)
            m_k = mk_cpp_call(y, h, h_k);
        
        // In order to obtain the likelihood contibution of point k, we create vectors of size 1
        std::vector<double> y_at_k(1);
        std::vector<double> h_at_k(1);
        y_at_k[0] = y[k-1];
        h_at_k[0] = h[k-1];
        double L_k = Lts_cpp_call(y_at_k, h_at_k)[0];
        
        S[i] = m_k / L_k;
    }
    double M_k = mean(S);
    
    std::vector<double> args = vector_divide(S, M_k);
    
    return args;
}

// [[Rcpp::export]]
NumericVector psi_divergence_estimator_arguments_cpp(NumericMatrix thetas, NumericMatrix hs, std::vector<double> y, int k, int p, int q){
    std::vector<std::vector<double>> thetas_c(thetas.nrow(), std::vector<double>(thetas.ncol()));
    std::vector<std::vector<double>> hs_c(hs.nrow(), std::vector<double>(hs.ncol()));
    for (int i = 0; i < thetas.nrow(); ++i) {
        for (int j = 0; j < thetas.ncol(); ++j) {
            thetas_c[i][j] = thetas(i, j);
        }
    }
    for (int i = 0; i < hs.nrow(); ++i) {
        for (int j = 0; j < hs.ncol(); ++j) {
            hs_c[i][j] = hs(i, j);
        }
    }

    std::vector<double> S_c = psi_divergence_estimator_arguments_cpp_call(thetas_c, hs_c, y, k, p, q);
    NumericVector S = NumericVector(S_c.begin(), S_c.end());
    
    return S;
}

// Kullback-Leibler
NumericVector KL(NumericVector x){
    NumericVector new_x(x.size());
    for(int i = 0; i < x.size(); i++){
        new_x[i] = -log(x[i]);
    }
    return new_x;
}
// Directed Divergence
NumericVector KL_rev(NumericVector x){
    NumericVector new_x(x.size());
    for(int i = 0; i < x.size(); i++){
        new_x[i] = x[i]*log(x[i]);
    }
    return new_x;
}
// Divergência J - Versão simétrica da KL (soma de KL e KL_rev)
NumericVector J(NumericVector x){
    NumericVector new_x(x.size());
    for(int i = 0; i < x.size(); i++){
        new_x[i] = (x[i]-1)*log(x[i]);
    }
    return new_x;
}
// Divergência Qui-quadrado
NumericVector chi2(NumericVector x){
    NumericVector new_x(x.size());
    for(int i = 0; i < x.size(); i++){
        new_x[i] = pow(x[i]-1, 2);
    }
    return new_x;
}
// Divergência Qui-quadrado reversa
NumericVector chi2_rev(NumericVector x){
    NumericVector new_x(x.size());
    for(int i = 0; i < x.size(); i++){
        new_x[i] = pow(x[i]-1, 2) / x[i];
    }
    return new_x;
}
// Versão simétrica da divergência Qui-quadrado (soma de chi2 e chi2_rev)
NumericVector chi2_sym(NumericVector x){
    NumericVector new_x(x.size());
    for(int i = 0; i < x.size(); i++){
        new_x[i] = (x[i]-1) * (pow(x[i],2)-1) / x[i];
    }
    return new_x;
}
// Norma variacional (Simétrica)
NumericVector L1(NumericVector x){
    NumericVector new_x(x.size());
    for(int i = 0; i < x.size(); i++){
        new_x[i] = std::abs(x[i] - 1)/2;
    }
    return new_x;
}
// Divergência de Hellinger (Simétrica)
NumericVector H(NumericVector x){
    NumericVector new_x(x.size());
    for(int i = 0; i < x.size(); i++){
        new_x[i] = pow(sqrt(x[i])-1, 2) / 2;
    }
    return new_x;
}
// Divergência de Jensen-Shannon (Versão simétrica e suavizada da KL)
NumericVector JS(NumericVector x){
    NumericVector new_x(x.size());
    for(int i = 0; i < x.size(); i++){
        new_x[i] = (x[i]*log(x[i]) - (x[i]+1)*log((x[i]+1)/2)) / 2;
    }
    return new_x;
}

// [[Rcpp::export]]
NumericMatrix psi_divergences_cpp(NumericMatrix thetas, NumericMatrix hs, std::vector<double> y, int p, int q, IntegerVector k_values, bool verbose){
    NumericMatrix divergences( k_values.size(), 9 );
    double KL_divergence, KL_rev_divergence, J_divergence, chi2_divergence, chi2_rev_divergence, chi2_sym_divergence, L1_divergence, H_divergence, JS_divergence;
    NumericVector psi_k_arg( thetas.nrow() );
    
    int k;
    for(int i = 0; i < k_values.size(); i++){
        k = k_values[i];
        psi_k_arg = psi_divergence_estimator_arguments_cpp(thetas, hs, y, k, p, q);
        
        KL_divergence = mean( KL(psi_k_arg) );
        KL_rev_divergence = mean( KL_rev(psi_k_arg) );
        J_divergence = mean( J(psi_k_arg) );
        chi2_divergence = mean( chi2(psi_k_arg) );
        chi2_rev_divergence = mean( chi2_rev(psi_k_arg) );
        chi2_sym_divergence = mean( chi2_sym(psi_k_arg) );
        L1_divergence = mean( L1(psi_k_arg) );
        H_divergence = mean( H(psi_k_arg) );
        JS_divergence = mean( JS(psi_k_arg) );

        divergences(i,0) = KL_divergence;
        divergences(i,1) = KL_rev_divergence;
        divergences(i,2) = J_divergence;
        divergences(i,3) = chi2_divergence;
        divergences(i,4) = chi2_rev_divergence;
        divergences(i,5) = chi2_sym_divergence;
        divergences(i,6) = L1_divergence;
        divergences(i,7) = H_divergence;
        divergences(i,8) = JS_divergence;
    }
    
    return divergences;
}

// [[Rcpp::depends(RcppParallel)]]
#include <RcppParallel.h>
using namespace RcppParallel;

struct PsiDivergenceEstimatorArgumentsCppParallel : public Worker{
    
    // Input values
    const RMatrix<double> thetas;
    const RMatrix<double> hs;
    const RVector<double> y;
    const int k;
    const int p;
    const int q;
    
    // Output vector
    RVector<double> S;
    
    // Initialize from Rcpp input and output matrices and vectors
    PsiDivergenceEstimatorArgumentsCppParallel(
        const NumericMatrix& thetas,
        const NumericMatrix& hs,
        const NumericVector& y,
        const int& k,
        const int& p,
        const int& q,
        NumericVector S
    ) : thetas(thetas), hs(hs), y(y), k(k), p(p), q(q), S(S) {}
    
    void operator()(std::size_t begin, std::size_t end){
        for(std::size_t i = begin; i < end; i++){
            int T = y.size();
            
            RMatrix<double>::Row theta_row = thetas.row(i);
            RMatrix<double>::Row h_row = hs.row(i);
            
            // Converts object from RMatrix<double>::Row and RVector<double> to std::vector<double>
            std::vector<double> theta_c( thetas.ncol() );
            std::vector<double> h_c( hs.ncol() );
            std::vector<double> y_c( y.length() );
            std::copy(theta_row.begin(), theta_row.end(), theta_c.begin());
            std::copy(h_row.begin(), h_row.end(), h_c.begin());
            std::copy(y.begin(), y.end(), y_c.begin());
            
            std::vector<double> h_k_c = h_remove_k_cpp_call(theta_c, y_c, h_c, k, p, q);

            // If k == T, m_k = 1, if not we call its respective function
            double m_k = 1;
            if(k < T)
                m_k = mk_cpp(y_c, h_c, h_k_c);

            // In order to obtain the likelihood contibution of point k, we create vectors of size 1
            std::vector<double> y_at_k(1);
            std::vector<double> h_at_k(1);
            y_at_k[0] = y_c[k-1];
            h_at_k[0] = h_c[k-1];
            double L_k = Lts_cpp_call(y_at_k, h_at_k)[0];
            
            S[i] = m_k / L_k;
        }
    }
};

// [[Rcpp::export]]
NumericVector psi_divergence_estimator_arguments_cpp_parallel(NumericMatrix thetas, NumericMatrix hs, NumericVector y, int k, int p, int q){
    
    // Allocate the vector we will return
    NumericVector S(thetas.nrow());
    
    // create the worker
    PsiDivergenceEstimatorArgumentsCppParallel worker_obj(thetas, hs, y, k, p, q, S);
    
    // Call the worker using the parallelFor function
    parallelFor(0, thetas.nrow(), worker_obj);
    
    // Convert the NumericVector to std::vector<double>
    std::vector<double> S_c( S.size() );
    std::copy(S.begin(), S.end(), S_c.begin());
    
    // Once obtained all m_k(theta^(s)) / L_k(theta^(s)), standardize using the M_k
    double M_k = mean( S_c );
    S_c = vector_divide(S_c, M_k);
    
    // Convert it back to NumericVector
    S = NumericVector(S_c.begin(), S_c.end());
    
    return S;
}

// [[Rcpp::export]]
NumericVector psi_divergences_cpp_parallel(NumericMatrix thetas, NumericMatrix hs, NumericVector y, int p, int q, IntegerVector k_values, bool verbose){
    NumericMatrix divergences( k_values.size(), 9 );
    double KL_divergence, KL_rev_divergence, J_divergence, chi2_divergence, chi2_rev_divergence, chi2_sym_divergence, L1_divergence, H_divergence, JS_divergence;
    NumericVector psi_k_arg( thetas.nrow() );
    
    int k;
    for(int i = 0; i < k_values.size(); i++){
        k = k_values[i];
        psi_k_arg = psi_divergence_estimator_arguments_cpp_parallel(thetas, hs, y, k, p, q);
        
        KL_divergence = mean( KL(psi_k_arg) );
        KL_rev_divergence = mean( KL_rev(psi_k_arg) );
        J_divergence = mean( J(psi_k_arg) );
        chi2_divergence = mean( chi2(psi_k_arg) );
        chi2_rev_divergence = mean( chi2_rev(psi_k_arg) );
        chi2_sym_divergence = mean( chi2_sym(psi_k_arg) );
        L1_divergence = mean( L1(psi_k_arg) );
        H_divergence = mean( H(psi_k_arg) );
        JS_divergence = mean( JS(psi_k_arg) );

        divergences(i,0) = KL_divergence;
        divergences(i,1) = KL_rev_divergence;
        divergences(i,2) = J_divergence;
        divergences(i,3) = chi2_divergence;
        divergences(i,4) = chi2_rev_divergence;
        divergences(i,5) = chi2_sym_divergence;
        divergences(i,6) = L1_divergence;
        divergences(i,7) = H_divergence;
        divergences(i,8) = JS_divergence;
    }
    
    return divergences;
}

// Obtain the vector of conditional variances for the GARCH model
VectorX128 compute_h(VectorX128 alpha, mp::float128 beta, VectorX128 y){
    int T = y.size();
    
    VectorX128 h(T);
    h[0] = alpha[0];
    for(int t = 1; t < T; t++)
        h[t] = alpha[0] + alpha[1] * mp::pow(y[t-1], 2) + beta * h[t-1];
    
    return h;
}

// Obtain the log-likelihood function (up to a constant) given that the conditional variances vector is yet to be aquired
mp::float128 log_likelihood_garch(VectorX128 alpha, mp::float128 beta, VectorX128 y){
    mp::float128 S(0);
    int T = y.size();
    
    VectorX128 h = compute_h(alpha, beta, y);
    for(int t = 0; t < T; t++){
        S += -log(h[t])/2 - mp::pow(y[t], 2)/(2*h[t]);
    }
    
    return S;
}

// Obtain the likelihood function (up to a constant) given that the conditional variances vector is yet to be aquired
mp::float128 likelihood_garch(VectorX128 alpha, mp::float128 beta, VectorX128 y){
    mp::float128 S = log_likelihood_garch(alpha, beta, y);
    S = mp::exp(S);
    return S;
}

// Obtain the log-likelihood function given that the conditional variances vector was already obtained
mp::float128 log_likelihood_garch_known_h(VectorX128 y, VectorX128 h){
    mp::float128 S(0);
    int T = y.size();
    for(int t = 0; t < T; t++){
        S += -log(h[t])/2 - mp::pow(y[t], 2)/(2*h[t]);
    }
    return S;
}
// Obtain the likelihood function given that the conditional variances vector was already obtained
mp::float128 likelihood_garch_known_h(VectorX128 y, VectorX128 h){
    mp::float128 S = log_likelihood_garch_known_h(y, h);
    S = mp::exp(S);
    return S;
}

// [[Rcpp::export]]
double log_likelihood_garch_cpp(NumericVector alpha, double beta, NumericVector y){
    VectorX128 alpha_cpp(2);
    alpha_cpp[0] = alpha[0]; alpha_cpp[1] = alpha[1];
    mp::float128 beta_cpp(beta);
    VectorX128 y_cpp(y.size());
    for(int t = 0; t < y.size(); t++)
        y_cpp[t] = y[t];

    mp::float128 res = log_likelihood_garch(alpha_cpp, beta_cpp, y_cpp);
    
    return res.convert_to<double>();
}

// [[Rcpp::export]]
NumericVector compute_h_cpp(NumericVector alpha, double beta, NumericVector y){
    VectorX128 alpha_cpp(2);
    alpha_cpp[0] = alpha[0]; alpha_cpp[1] = alpha[1];
    mp::float128 beta_cpp = beta;
    VectorX128 y_cpp(y.size());
    for(int t = 0; t < y.size(); t++)
        y_cpp[t] = y[t];
    
    VectorX128 h = compute_h(alpha_cpp, beta_cpp, y_cpp);
    
    NumericVector h_r(h.size());
    for(int t = 0; t < y.size(); t++)
        h_r[t] = h[t].convert_to<double>();
    
    return h_r;
}