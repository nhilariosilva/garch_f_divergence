// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::depends(BH)]]

#include <Rcpp.h>
#include <RcppEigen.h>

// C++ native libraries
#include <iostream>
#include <string>

using namespace Rcpp;

// Float point high precision packages (needed for the likelihood calculations - Values too small!)
#include <boost/multiprecision/float128.hpp>
#include <boost/multiprecision/mpfr.hpp>
namespace mp = boost::multiprecision;

// --------------------------------------- R functions ---------------------------------------
Function rtruncnorm("rtruncnorm");
Function rtmvn("rtmvn");
// -------------------------------------------------------------------------------------------

// ------------------------------------ Global Constants ------------------------------------
typedef Eigen::Matrix<mp::float128, Eigen::Dynamic, Eigen::Dynamic> MatrixX128;
typedef Eigen::Matrix<mp::float128, Eigen::Dynamic, 1> VectorX128; // A vector for eigen is simply a one column matrix
typedef std::vector<MatrixX128> TensorX128; // More like a list of matrix, but we call it Tensor for convenience
// unsuported in Rcpp
// typedef Eigen::Tensor<mp::float128, 3> TensorX128; // 3-dimensional tensor

typedef Eigen::Matrix<bool, Eigen::Dynamic, 1> VectorXbool; // A 0-1 vector
// ------------------------------------------------------------------------------------------

// ------------------------------------ Global Constants ------------------------------------

const double PI = 3.14159265358979323846;

// Define a global constant matrix via the creation and direct calling of a lambda function:
// []: Capture clause: No variables are captured from the global scope in this case
// (): The lambda function takes no arguments in this case

// Prior: [alpha0, alpha1] = alpha ~ N_2([0,0], 100*Id)
const VectorX128 mu_alpha = [](){
    VectorX128 m(2);
    m[0] = 0; m[1] = 0;
    return m;
}();
const MatrixX128 sigma_alpha = [](){
    MatrixX128 m(2,2);
    m(0,0) = 10000; m(0,1) = 0;
    m(1,0) = 0; m(1,1) = 10000;
    return m;
}();
const MatrixX128 sigma_alpha_inv = [](){
    MatrixX128 m(2,2);
    m(0,0) = 1/10000; m(0,1) = 0;
    m(1,0) = 0; m(1,1) = 1/10000;
    return m;
}();

// Prior: beta ~ N(0, 100)
const mp::float128 mu_beta = 0;
const mp::float128 sigma_beta = 10000; // Here, sigma_beta represents the variance actually
const mp::float128 sigma_beta_inv = 1/sigma_beta;
// ------------------------------------------------------------------------------------------

// --------------------- Basic Vector and Matrix functions ---------------------
mp::float128 round_float128(mp::float128 val, int precision){
    std::string precision_txt = "1e" + std::to_string(precision);
    mp::float128 factor(precision_txt);
    return round(val*factor) / factor;
}

// --------------------------------------------------------------- Distributions used ---------------------------------------------------------------
mp::float128 runiform(){
    NumericVector u = Rcpp::runif(1);
    return u[0];
}
mp::float128 rexponential(mp::float128 lambda = 1){
    NumericVector u_r = Rcpp::runif(1);
    mp::float128 u = u_r[0];
    mp::float128 x = -mp::log(1 - u);
    return x;
}

// Log of the density for the normal distribution
mp::float128 log_dnorm(mp::float128 x, mp::float128 mu, mp::float128 sd){
    mp::float128 var = mp::pow(sd, 2);
    mp::float128 log_d = -mp::log( (mp::float128) 2*PI )/2 - mp::log(sd) - mp::pow(x - mu, 2)/(2*var);
    return log_d;
}
// Density for the normal distribution
mp::float128 dnorm(mp::float128 x, mp::float128 mu, mp::float128 sd){
    mp::float128 log_d = log_dnorm(x, mu, sd);
    mp::float128 d = mp::exp(log_d);
    return d;
}

// Density (up to a constant) for the truncated normal distribution centered at zero
mp::float128 log_dnorm_trunc0(mp::float128 x, mp::float128 mu, mp::float128 sd){
    if(x < 0)
        return 0;
    return log_dnorm(x, mu, sd);
}
// Log of density (up to a constant) for the truncated normal distribution centered at zero
mp::float128 dnorm_trunc0(mp::float128 x, mp::float128 mu, mp::float128 sd){
    if(x < 0)
        return 0;
    return dnorm(x, mu, sd);
}
// Generate a sample from the truncated normal distribution centered at zero
mp::float128 rnorm_trunc0(mp::float128 mu, mp::float128 sigma){
    // NumericVector x = rtruncnorm(1, _["a"] = 0, _["b"] = 1000000, _["mean"] = mu.convert_to<double>(), _["sd"] = sigma.convert_to<double>());
    Function rnorm("rnorm");
    NumericVector x_r = rnorm(1, _["mean"] = mu.convert_to<double>(), _["sd"] = sigma.convert_to<double>());
    mp::float128 x;
    if(x_r[0] > 0)
        x = x_r[0];
    else
        x = -x_r[0];
    return x;
}

// Log of density for the multivariate normal distribution
mp::float128 log_dmvnorm(VectorX128 x, VectorX128 mean, MatrixX128 sigma){
    int k = x.size();
    mp::float128 det_sigma = sigma.determinant();
    MatrixX128 sigma_inv = sigma.inverse();
    
    mp::float128 quadratic_form = (x-mean).transpose() * sigma_inv * (x-mean);
    mp::float128 log_d = -mp::log( (mp::float128) 2*PI ) - mp::log(det_sigma)/2 - quadratic_form/2;
    
    return log_d;
}
// Density for the multivariate normal distribution
mp::float128 dmvnorm(VectorX128 x, VectorX128 mean, MatrixX128 sigma){    
    mp::float128 log_d = log_dmvnorm(x, mean, sigma);
    mp::float128 d = mp::exp(log_d);
    return d;
}

// Density (up to a constant) for the truncated in zero multivariate normal distribution
mp::float128 log_dmvnorm_trunc0(VectorX128 x, VectorX128 mean, MatrixX128 sigma){
    // If any coordinate is lesser than zero, returns the density zero
    for(int i = 0; i < x.size(); i++)
        if(x[i] < 0)
            return 0;
    return log_dmvnorm(x, mean, sigma);
}
// Density (up to a constant) for the truncated in zero multivariate normal distribution
mp::float128 dmvnorm_trunc0(VectorX128 x, VectorX128 mean, MatrixX128 sigma){
    // If any coordinate is lesser than zero, returns the density zero
    for(int i = 0; i < x.size(); i++)
        if(x[i] < 0)
            return 0;
    return dmvnorm(x, mean, sigma);
}


// Generate a sample from the truncated in zero multivariate normal distribution
VectorX128 rmvnorm_trunc0(VectorX128 mean, MatrixX128 sigma){
    NumericVector mean_r(mean.size());
    NumericVector lower_r(mean.size());
    NumericVector upper_r(mean.size());
    NumericMatrix sigma_r(sigma.rows(), sigma.cols());
    
    // Initialize the R objects as copies of the given vector and matrix
    for(int i = 0; i < mean.size(); i++){
        mean_r[i] = mean[i].convert_to<double>();
        lower_r[i] = 0;
        upper_r[i] = 1000000; // Very high upper limit
        for(int j = 0; j < sigma.cols(); j++)
            sigma_r(i,j) = sigma(i,j).convert_to<double>();
    }
    
    Function rtmvn("rtmvn");
    // Generate a single observation from the multivariate truncated normal distribution
    NumericVector sample = rtmvn(1, _["Mean"] = mean_r, _["Sigma"] = sigma_r, _["lower"] = lower_r, _["upper"] = upper_r);
    
    VectorX128 sample_cpp(sample.size());
    for(int i = 0; i < sample.size(); i++)
        sample_cpp[i] = sample[i];
    return sample_cpp;
}
// --------------------------------------------------------------------------------------------------------------------------------------------------

// ---------------------------------------------------------- Model Related Functions ----------------------------------------------------------
mp::float128 log_dprior_garch(VectorX128 alpha, mp::float128 beta){
    mp::float128 log_alpha_prior = log_dmvnorm_trunc0(alpha, mu_alpha, sigma_alpha);
    mp::float128 log_beta_prior = log_dnorm_trunc0(beta, mu_beta, mp::sqrt(sigma_beta));
    return log_alpha_prior + log_beta_prior;
}
mp::float128 dprior_garch(VectorX128 alpha, mp::float128 beta){
    mp::float128 log_p = log_dprior_garch(alpha, beta);
    mp::float128 p = mp::exp(log_p);
    return p;
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

// Obtain the posterior density given that the conditional variances vector is yet to be aquired
mp::float128 log_dposterior_garch(VectorX128 alpha, mp::float128 beta, VectorX128 y){
    mp::float128 log_d = log_dprior_garch(alpha, beta) + log_likelihood_garch(alpha, beta, y);
    return log_d;
}
// Obtain the posterior density given that the conditional variances vector is yet to be aquired
mp::float128 dposterior_garch(VectorX128 alpha, mp::float128 beta, VectorX128 y){
    mp::float128 log_d = log_dposterior_garch(alpha, beta, y);
    mp::float128 d = mp::exp(log_d);
    return d;
}

// Obtain the posterior density given that the conditional variances vector is yet to be aquired
mp::float128 log_dposterior_garch_known_h(VectorX128 alpha, mp::float128 beta, VectorX128 y, VectorX128 h){
    mp::float128 log_d = log_dprior_garch(alpha, beta) + log_likelihood_garch_known_h(y, h);
    return log_d;
}
// Obtain the posterior density given that the conditional variances vector was already obtained
mp::float128 dposterior_garch_known_h(VectorX128 alpha, mp::float128 beta, VectorX128 y, VectorX128 h){
    mp::float128 log_d = log_dposterior_garch_known_h(alpha, beta, y, h);
    mp::float128 d = mp::exp(log_d);
    return d;
}
// ---------------------------------------------------------------------------------------------------------------------------------------------


// ------------------------------------------------- Sampling the GARCH posterior distribution -------------------------------------------------
/* Update the value for the parameters of sigma_alpha_hat and mu_alpha_hat, the mean and variance matrix of the approximate conditional distribution for alpha
given the data and the previous sampled parameter values*/

void update_alpha_parameters(VectorX128 y, VectorX128 y2, VectorX128 prev_h, VectorX128 prev_alpha, mp::float128 prev_beta, MatrixX128& prev_sigma_alpha, VectorX128& prev_mu_alpha){
    int T = y.size();

    VectorX128 vts(T);
    vts[0] = mp::pow(y[0], 2);

    VectorX128 lts_star(T);
    lts_star[0] = 1;
    VectorX128 vts_star(T);
    vts_star[0] = 0;
    
    mp::float128 ht2;
    mp::float128 y2_ht2;
    ht2 = prev_h[0]*prev_h[0];
    
    MatrixX128 CLambda_invC(2,2);
    CLambda_invC(0,0) = 1 / (prev_h[0]*prev_h[0]); // (lts_star[0]*lts_star[0]) / (prev_h[0]*prev_h[0]) - The only element of CLambda_invC that is not zero at time t = 1
    
    VectorX128 CLambda_invv(2);
    CLambda_invv[0] = y2[0] / ht2; // lts_star[0] * y2[0]/ht2 - The only element of CLambda_invv that is not zero at time t = 1
    
    // Populates the vectors necessary to obtain the mean and variance of the candidate distribution given the previous sampled values
    for(int t = 1; t < T; t++){
        vts[t] = y2[t];
        lts_star[t] = 1 + prev_beta * lts_star[t-1];
        vts_star[t] = vts[t-1] + prev_beta * vts_star[t-1];
        
        ht2 = prev_h[t]*prev_h[t];
        CLambda_invC(0,0) += (lts_star[t]*lts_star[t]) / ht2;
        CLambda_invC(0,1) += (lts_star[t]*vts_star[t]) / ht2;
        CLambda_invC(1,1) += (vts_star[t]*vts_star[t]) / ht2;
        
        y2_ht2 = y2[t]/ht2;
        CLambda_invv[0] += lts_star[t] * y2_ht2;
        CLambda_invv[1] += vts_star[t] * y2_ht2;
    }
    CLambda_invC(1,0) = CLambda_invC(0,1);
    CLambda_invC = CLambda_invC / 2;
    CLambda_invv = CLambda_invv / 2;
    
    MatrixX128 prev_sigma_alpha_inv = CLambda_invC + sigma_alpha_inv;
    prev_sigma_alpha = prev_sigma_alpha_inv.inverse();
    prev_mu_alpha = prev_sigma_alpha * (CLambda_invv + sigma_alpha_inv * mu_alpha);
}

/*
Receives the parameters of the candidate posterior conditional distribution for alpha. Returns a sampled value for alpha and also updates the values of the parameters for the next candidate
posterior conditional distribution for alpha
*/
VectorX128 sample_posterior_alpha(int q, VectorX128 y, VectorX128 y2, VectorX128 prev_h, VectorX128 prev_alpha, mp::float128 prev_beta, MatrixX128 prev_sigma_alpha, VectorX128 prev_mu_alpha, MatrixX128& new_sigma_alpha, VectorX128& new_mu_alpha, VectorX128& new_h, VectorX128& acceptance_probs_alpha, VectorXbool& accepted_alpha, MatrixX128& candidates_alpha, MatrixX128& means_alpha, TensorX128& sigmas_alpha){
    
    // Saves the information for the study of the posterior sample (The parameters in q correspond to the distribution from where obs q was sampled)
    means_alpha(q,0) = prev_mu_alpha[0];
    means_alpha(q,1) = prev_mu_alpha[1];
    sigmas_alpha[q] = prev_sigma_alpha;
    
    // Sample from the updated alpha distribution
    VectorX128 new_alpha = rmvnorm_trunc0(prev_mu_alpha, prev_sigma_alpha);
    candidates_alpha(q,0) = new_alpha[0];
    candidates_alpha(q,1) = new_alpha[1];
    
    // Obtain the new corresponding conditional variances vector
    new_h = compute_h(new_alpha, prev_beta, y);
    
    // Obtain the parameters for the distribution of (alpha | new_alpha, prev_beta, y)
    update_alpha_parameters(y, y2, new_h, new_alpha, prev_beta, new_sigma_alpha, new_mu_alpha);

    // Acceptance probability for the new sampled value
    mp::float128 log_p_newalpha_prevbeta = log_dposterior_garch_known_h(new_alpha, prev_beta, y, new_h);
    mp::float128 log_p_prevalpha_prevbeta = log_dposterior_garch_known_h(prev_alpha, prev_beta, y, prev_h);
    mp::float128 log_q_prevalpha = log_dmvnorm(prev_alpha, new_mu_alpha, new_sigma_alpha);
    mp::float128 log_q_newalpha = log_dmvnorm(new_alpha, prev_mu_alpha, prev_sigma_alpha);
    
    mp::float128 log_acceptance_prob = log_p_newalpha_prevbeta - log_p_prevalpha_prevbeta + log_q_prevalpha - log_q_newalpha;
        
    acceptance_probs_alpha[q] = mp::exp(log_acceptance_prob);

    if(log_acceptance_prob >= 0){
        accepted_alpha[q] = 1;
        return new_alpha;
    }else{
        // U <= p -> -log U >= -log p -> X >= -log p; X ~ Exp(1)
        if(rexponential(1) >= -log_acceptance_prob){
            accepted_alpha[q] = 1;
            return new_alpha;
        }else{
            accepted_alpha[q] = 0;
            return prev_alpha;
        }
    }
}

void update_beta_parameters(VectorX128 y, VectorX128 y2, VectorX128 prev_h, VectorX128 prev_alpha, mp::float128 prev_beta, mp::float128& prev_sigma_beta, mp::float128& prev_mu_beta){
    int T = y.size();

    VectorX128 vts(T);
    vts[0] = y2[0];
    
    VectorX128 prev_delta(T);
    prev_delta[0] = 0;
    
    VectorX128 prev_z(T);
    prev_z[0] = vts[0] - prev_alpha[0];
    
    VectorX128 prev_r(T);
    prev_r[0] = prev_z[0];
    
    mp::float128 ht2;
    mp::float128 DeltaLambda_invDelta(0);
    mp::float128 DeltaLambda_invr(0);
    
    // Populates the vectors necessary to obtain the mean and variance of the candidate distribution given the previous sampled values
    for(int t = 1; t < T; t++){
        vts[t] = y2[t];
        prev_z[t] = vts[t] - prev_alpha[0] - (prev_alpha[1] + prev_beta) * vts[t-1] + prev_beta * prev_z[t-1];
        prev_delta[t] = vts[t-1] - prev_z[t-1] + prev_beta * prev_delta[t-1];
        prev_r[t] = prev_z[t] + prev_beta * prev_delta[t];
        
        ht2 = prev_h[t]*prev_h[t];
        DeltaLambda_invDelta += prev_delta[t]*prev_delta[t] / ht2;
        DeltaLambda_invr += prev_delta[t]*prev_r[t] / ht2;
    }
    DeltaLambda_invDelta = DeltaLambda_invDelta / 2;
    DeltaLambda_invr = DeltaLambda_invr / 2;
    
    mp::float128 prev_sigma_beta_inv = DeltaLambda_invDelta + sigma_beta_inv;
    prev_sigma_beta = 1/prev_sigma_beta_inv;
    prev_mu_beta = prev_sigma_beta * (DeltaLambda_invr + sigma_beta_inv * mu_beta);
}

/*
Receives the parameters of the candidate posterior conditional distribution for beta. Returns a sampled value for beta and also updates the values of the parameters for the next candidate
posterior conditional distribution for beta
*/
mp::float128 sample_posterior_beta(int q, VectorX128 y, VectorX128 y2, VectorX128 prev_h, VectorX128 prev_alpha, mp::float128 prev_beta, mp::float128 prev_sigma_beta, mp::float128 prev_mu_beta, mp::float128& new_sigma_beta, mp::float128& new_mu_beta, VectorX128& new_h, VectorX128& acceptance_probs_beta, VectorXbool& accepted_beta, VectorX128& candidates_beta){
    // Sample from the updated alpha distribution
    mp::float128 new_beta = rnorm_trunc0(prev_mu_beta, mp::sqrt(prev_sigma_beta));
    candidates_beta[q] = new_beta;
    
    // Obtain the new corresponding conditional variances vector
    new_h = compute_h(prev_alpha, new_beta, y);

    // Obtain the parameters for the distribution of (alpha | new_alpha, prev_beta, y)
    update_beta_parameters(y, y2, new_h, prev_alpha, new_beta, new_sigma_beta, new_mu_beta);
    
    // Acceptance probability for the new sampled value
    mp::float128 log_p_prevalpha_newbeta = log_dposterior_garch_known_h(prev_alpha, new_beta, y, new_h);
    mp::float128 log_p_prevalpha_prevbeta = log_dposterior_garch_known_h(prev_alpha, prev_beta, y, prev_h);
    mp::float128 log_q_prevbeta = log_dnorm_trunc0(prev_beta, new_mu_beta, mp::sqrt(new_sigma_beta));
    mp::float128 log_q_newbeta = log_dnorm_trunc0(new_beta, prev_mu_beta, mp::sqrt(prev_sigma_beta));
    
    mp::float128 log_acceptance_prob = log_p_prevalpha_newbeta - log_p_prevalpha_prevbeta + log_q_prevbeta - log_q_newbeta;
    acceptance_probs_beta[q] = mp::exp(log_acceptance_prob);
    
    if(log_acceptance_prob >= 0){
        accepted_beta[q] = 1;
        return new_beta;
    }else{
        // Jogar moeda para aceitar ou não a amostra!
        if(rexponential(1) >= -log_acceptance_prob){
            accepted_beta[q] = 1;
            return new_beta;
        }else{
            accepted_beta[q] = 0;
            return prev_beta;
        }
    }
}

void sample_posterior_garch(int Q, VectorX128 y, VectorX128& alpha0, mp::float128& beta0,
                            MatrixX128& sample_alpha, VectorX128& sample_beta,
                            VectorX128& acceptance_probs_alpha, VectorX128& acceptance_probs_beta, VectorXbool& accepted_alpha, VectorXbool& accepted_beta,
                            MatrixX128& candidates_alpha, VectorX128& candidates_beta,
                            MatrixX128& means_alpha, TensorX128& sigmas_alpha,
                            MatrixX128& hs
                           ){
    VectorX128 h0 = compute_h(alpha0, beta0, y);
    
    VectorX128 y2(y.size());
    for(int t = 0; t < y.size(); t++)
        y2[t] = y[t]*y[t];
    
    // Obtain the candidate posterior distribution parameters for the first sampled alpha
    MatrixX128 sigma_alpha0(2,2);
    VectorX128 mu_alpha0(2);
    update_alpha_parameters(y, y2, h0, alpha0, beta0, sigma_alpha0, mu_alpha0);
    
    mp::float128 sigma_beta0(1);
    mp::float128 mu_beta0(0);
    update_beta_parameters(y, y2, h0, alpha0, beta0, sigma_beta0, mu_beta0);
    
    // Variables that shall change during the iterations
    MatrixX128 new_sigma_alpha(2,2);
    VectorX128 new_mu_alpha(2);
    mp::float128 new_sigma_beta;
    mp::float128 new_mu_beta;
    VectorX128 new_h(y.size());
    
    VectorX128 new_alpha;
    mp::float128 new_beta;
    for(int q = 0; q < Q; q++){
        // ------------------------------------------ Sampling alpha ------------------------------------------
        // Obtain a single sample from the candidate conditional posterior distribution for alpha
        new_alpha = sample_posterior_alpha(q, y, y2, h0, alpha0, beta0, sigma_alpha0, mu_alpha0, new_sigma_alpha, new_mu_alpha, new_h, acceptance_probs_alpha, accepted_alpha, candidates_alpha, means_alpha, sigmas_alpha);
        // Save the sampled value on the data matrix
        sample_alpha(q,0) = new_alpha[0];
        sample_alpha(q,1) = new_alpha[1];
        
        // Updates the values for the next iteration if the sample was accepted
        if(accepted_alpha[q] == 1){
            alpha0 = new_alpha;
            sigma_alpha0 = new_sigma_alpha;
            mu_alpha0 = new_mu_alpha;
            h0 = new_h;
        }
        // ----------------------------------------------------------------------------------------------------
        
        // ------------------------------------------ Sampling beta ------------------------------------------
        // Obtain a single sample from the candidate conditional posterior distribution for alpha
        new_beta = sample_posterior_beta(q, y, y2, h0, alpha0, beta0, sigma_beta0, mu_beta0, new_sigma_beta, new_mu_beta, new_h, acceptance_probs_beta, accepted_beta, candidates_beta);
        // Save the sampled value on the data matrix
        sample_beta[q] = new_beta;
        
        // Updates the values for the next iteration if the sample was accepted
        if(accepted_beta[q] == 1){
            beta0 = new_beta;
            sigma_beta0 = new_sigma_beta;
            mu_beta0 = new_mu_beta;
            h0 = new_h;
        }
        
        // ---------------------------------------------------------------------------------------------------
        
        // ----------------------------------------- Saving h values -----------------------------------------
        // Saves the current h values i.e. if the sample was not accepted, save the previous values since the chain won't update
        for(int t = 0; t < y.size(); t++)
            hs(q,t) = h0[t];
        // ---------------------------------------------------------------------------------------------------   
    }
}

// [[Rcpp::export]]
List sample_posterior_garch_cpp(NumericVector y, int Q, NumericVector alpha0, double beta0){

    int T = y.size();

    // Converts the variables from R types to Eigen C++ types
    VectorX128 alpha0_cpp(2);
    alpha0_cpp[0] = alpha0[0]; alpha0_cpp[1] = alpha0[1];
    mp::float128 beta0_cpp = beta0;
    VectorX128 y_cpp(T);
    for(int t = 0; t < T; t++)
        y_cpp[t] = y[t];
    
    // Create the data objects that will store the sampled posterior values
    MatrixX128 sample_alpha(Q,2);
    VectorX128 sample_beta(Q);
    VectorX128 acceptance_probs_alpha(Q);
    VectorX128 acceptance_probs_beta(Q);
    VectorXbool accepted_alpha(Q);
    VectorXbool accepted_beta(Q);
    
    MatrixX128 candidates_alpha(Q,2);
    VectorX128 candidates_beta(Q);
    
    MatrixX128 means_alpha(Q,2);
    TensorX128 sigmas_alpha(Q);
    
    MatrixX128 hs(Q,T);
    
    // Call the iterative posterior sampler for a posterior sample of size Q
    sample_posterior_garch(Q, y_cpp, alpha0_cpp, beta0_cpp, sample_alpha, sample_beta, acceptance_probs_alpha, acceptance_probs_beta, accepted_alpha, accepted_beta, candidates_alpha, candidates_beta, means_alpha, sigmas_alpha, hs);
    
    // Formats back the results to R variables
    NumericMatrix sample_alpha_r(Q,2);
    NumericVector sample_beta_r(Q);
    NumericVector acceptance_probs_alpha_r(Q); NumericVector accepted_alpha_r(Q);
    NumericVector acceptance_probs_beta_r(Q); NumericVector accepted_beta_r(Q);
    NumericMatrix candidates_alpha_r(Q,2); 
    NumericVector candidates_beta_r(Q);
    
    NumericMatrix means_alpha_r(Q,2);
    NumericVector sigmas_alpha_r(Q*2*2); // Q*4
    
    NumericMatrix h_r(Q,T);
    
    for(int q = 0; q < Q; q++){
        sample_alpha_r(q,0) = sample_alpha(q,0).convert_to<double>(); sample_alpha_r(q,1) = sample_alpha(q,1).convert_to<double>();
        sample_beta_r[q] = sample_beta[q].convert_to<double>();
        acceptance_probs_alpha_r[q] = acceptance_probs_alpha[q].convert_to<double>();
        accepted_alpha_r[q] = accepted_alpha[q];
        acceptance_probs_beta_r[q] = acceptance_probs_beta[q].convert_to<double>();
        accepted_beta_r[q] = accepted_beta[q];
        
        candidates_alpha_r(q,0) = candidates_alpha(q,0).convert_to<double>();
        candidates_alpha_r(q,1) = candidates_alpha(q,1).convert_to<double>();
        candidates_beta_r[q] = candidates_beta[q].convert_to<double>();
        
        means_alpha_r(q,0) = means_alpha(q,0).convert_to<double>();
        means_alpha_r(q,1) = means_alpha(q,1).convert_to<double>();
        
        // Populate the tensor this way so it does not mess its order when converted via the Dimension(Q,2,2) function
        sigmas_alpha_r[q] = sigmas_alpha[q](0,0).convert_to<double>();
        sigmas_alpha_r[q+Q] = sigmas_alpha[q](0,1).convert_to<double>();
        sigmas_alpha_r[q+2*Q] = sigmas_alpha[q](1,0).convert_to<double>();
        sigmas_alpha_r[q+3*Q] = sigmas_alpha[q](1,1).convert_to<double>();
        
        for(int t = 0; t < T; t++)
            h_r(q,t) = hs(q,t).convert_to<double>();
    }
    sigmas_alpha_r.attr("dim") = Dimension(Q, 2, 2); // Set the dimension of the R vector so it gets converted to an R array object automatically
    
    List L = List::create(
        _["alpha"] = sample_alpha_r,
        _["beta"] = sample_beta_r,
        _["acceptance_probs_alpha"] = acceptance_probs_alpha_r,
        _["accepted_alpha"] = accepted_alpha_r,
        _["acceptance_probs_beta"] = acceptance_probs_beta_r,
        _["accepted_beta"] = accepted_beta_r,
        _["candidates_alpha"] = candidates_alpha_r,
        _["candidates_beta"] = candidates_beta_r,
        _["means_alpha"] = means_alpha_r,
        _["sigmas_alpha"] = sigmas_alpha_r,
        _["h"] = h_r
    );
    
    return L;
}

// ---------------------------------------------------------------------------------------------------------------------------------------------


// [[Rcpp::export]]
void test(){
    mp::float128 a = 3;
    mp::float128 b = 6;
    if(a >= b)
        Rcout << "AAA" << "\n";
    else
        Rcout << "BBB" << "\n";
}

// [[Rcpp::export]]
double dmvnorm_cpp(NumericVector x, NumericVector mean, NumericMatrix sigma){
    int k = x.size();
    VectorX128 x_cpp(k);
    VectorX128 mean_cpp(k);
    MatrixX128 sigma_cpp(k,k);
    for(int i = 0; i < k; i++){
        x_cpp[i] = x[i];
        mean_cpp[i] = mean[i];
        for(int j = 0; j < k; j++)
            sigma_cpp(i,j) = sigma(i,j);
    }
    mp::float128 res = dmvnorm(x_cpp, mean_cpp, sigma_cpp);
    
    return res.convert_to<double>();
}


// [[Rcpp::export]]
double likelihood_garch_cpp(NumericVector alpha, double beta, NumericVector y){

    VectorX128 alpha_cpp(2);
    alpha_cpp[0] = alpha[0]; alpha_cpp[1] = alpha[1];
    mp::float128 beta_cpp(beta);
    VectorX128 y_cpp(y.size());
    for(int t = 0; t < y.size(); t++)
        y_cpp[t] = y[t];

    mp::float128 res = likelihood_garch(alpha_cpp, beta_cpp, y_cpp);
    
    return res.convert_to<double>();
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


// --------------------------------------------------------------------------------------------------------------------------------------------------