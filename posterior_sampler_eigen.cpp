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
Function rdirichlet("rdirichlet");
Function ddirichlet("ddirichlet");
Function rlnorm("rlnorm");
Function dlnorm("dlnorm");
// -------------------------------------------------------------------------------------------

// ------------------------------------ Global Types ------------------------------------
typedef Eigen::Matrix<mp::float128, Eigen::Dynamic, Eigen::Dynamic> MatrixX128;
typedef Eigen::Matrix<mp::float128, Eigen::Dynamic, 1> VectorX128; // A vector for eigen is simply a one column matrix
typedef std::vector<MatrixX128> TensorX128; // More like a list of matrix, but we call it Tensor for convenience
// unsuported in Rcpp
// typedef Eigen::Tensor<mp::float128, 3> TensorX128; // 3-dimensional tensor

typedef Eigen::Matrix<bool, Eigen::Dynamic, 1> VectorXbool; // A 0-1 vector
// ------------------------------------------------------------------------------------------

// ------------------------------------ Global Constants ------------------------------------

const double PI = 3.14159265358979323846;

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

// Generate a single sample from the Dirichlet distribution with concentration vector a
VectorX128 rdirichlet0(VectorX128 a){
    // Calibration vector for R function rdirichlet
    NumericVector a_r( a.size() );
    
    // Pass C++ vector values to R value before calling the R sampler
    for(int i = 0; i < a.size(); i++){
        a_r[i] = a[i].convert_to<double>();
    }

    // Pull R function from R environment and call it
    Function rdirichlet("rdirichlet");
    NumericVector sample = rdirichlet(1, _["alpha"] = a_r);

    VectorX128 sample_cpp(sample.size());
    // Convert function output back to a C++ object
    for(int i = 0; i < sample.size(); i++)
        sample_cpp[i] = sample[i];

    return sample_cpp;
}

// Generate a single sample from the Lognormal distribution with a given mu and sigma parameters
mp::float128 rlognormal0(mp::float128 meanlog, mp::float128 sdlog){   
    // Pull R function from R environment and call it
    Function rlnorm("rlnorm");
    NumericVector sample = rlnorm(1, _["meanlog"] = meanlog.convert_to<double>(), _["sdlog"] = sdlog.convert_to<double>());
    mp::float128 sample_cpp = sample[0];
    return sample_cpp;
}

// --------------------------------------------------------------- GARCH(p,q) related functions ---------------------------------------------------------------

// Obtain the vector of conditional variances for the GARCH model
VectorX128 compute_h(VectorX128 alpha, VectorX128 beta, VectorX128 y){
    int T = y.size();

    // Order of GARCH(p,q) model
    int p = alpha.size()-1;
    int q = beta.size();

    mp::float128 alpha_terms;
    mp::float128 beta_terms;
    VectorX128 h(T);
    
    h[0] = alpha[0];
    for(int t = 1; t < T; t++){
        alpha_terms = 0;
        beta_terms = 0;
        for(int i = 1; i <= p; i++)
            if (t - i >= 0) {
                alpha_terms += alpha[i] * mp::pow(y[t-i], 2);
            }
        for(int j = 1; j <= q; j++)
            if (t - j >= 0) {
                beta_terms += beta[j-1] * h[t-j];
            }
        h[t] = alpha[0] + alpha_terms + beta_terms;
    }
    
    return h;
}

// Obtain the log-likelihood function (up to a constant) given that the conditional variances vector is yet to be aquired
mp::float128 log_likelihood_garch(VectorX128 alpha, VectorX128 beta, VectorX128 y){
    mp::float128 S(0);
    int T = y.size();
    VectorX128 h = compute_h(alpha, beta, y);
    for(int t = 0; t < T; t++){
        S += -log(h[t])/2 - mp::pow(y[t], 2)/(2*h[t]);
    }
    return S;
}
// [[Rcpp::export]]
double log_likelihood_garch_cpp(NumericVector alpha, NumericVector beta, NumericVector y){
    // Order of GARCH(p,q) model
    int p = alpha.size()-1;
    int q = beta.size();
    
    VectorX128 alpha_cpp(1+p);
    VectorX128 beta_cpp(q);
    
    for(int i = 0; i < 1+p; i++)
        alpha_cpp[i] = alpha[i];
    for(int j = 0; j < q; j++)
        beta_cpp[j] = beta[j];
    
    VectorX128 y_cpp(y.size());
    for(int t = 0; t < y.size(); t++)
        y_cpp[t] = y[t];

    mp::float128 res = log_likelihood_garch(alpha_cpp, beta_cpp, y_cpp);
    
    return res.convert_to<double>();
}


// Obtain the likelihood function (up to a constant) given that the conditional variances vector is yet to be aquired
mp::float128 likelihood_garch(VectorX128 alpha, VectorX128 beta, VectorX128 y){
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

// --------------------------------------------------------------- Functions related to GARCH posterior ---------------------------------------------------------------
// Considering priors
// alpha0 ~ Lognormal( mu = log(1), sigma = 5 )
// (alpha_1, ..., \alpha_p, \beta_1, ..., \beta_q) ~ Dirichlet(1, ..., 1, 1, ..., 1)

// Receives given alpha^r and beta^r vectors. Obtain the posterior distribution considering the priors above
mp::float128 log_posterior_density_dirichlet(VectorX128 alpha_r, VectorX128 beta_r, VectorX128 y){  
    // GARCH(p,q) dimensions
    int p = alpha_r.size()-1;
    int q = beta_r.size();
    
    NumericVector alphabeta_prior_argument_r(1 + p + q);
    mp::float128 sum_alphabeta;

    sum_alphabeta = 0;
    for(int i = 0; i < p+q; i++){
        if(i < p)
            alphabeta_prior_argument_r[i] = alpha_r[i+1].convert_to<double>();
        else
            alphabeta_prior_argument_r[i] = beta_r[i-p].convert_to<double>();
        // Sums the value of the current parameter into the total
        sum_alphabeta += alphabeta_prior_argument_r[i];
    }
    // The last position of the Dirichlet component vector is one minus all the other parameters
    alphabeta_prior_argument_r[p+q] = (1-sum_alphabeta).convert_to<double>();
    
    NumericVector prior_concentration_r(1 + p + q);
    prior_concentration_r.fill(1.0);

    Function ddirichlet("ddirichlet");
    Function dlnorm("dlnorm");

    // Fix prior_log_density_alpha0
    mp::float128 prior_log_density_alpha0 = as<double>( dlnorm(alpha_r[0].convert_to<double>(), _["meanlog"] = std::log(1.0), _["sdlog"] = 5.0, _["log"] = true) );

    double prior_density_alphabeta = as<double>( ddirichlet(alphabeta_prior_argument_r, prior_concentration_r) );
    mp::float128 prior_log_density_alphabeta = std::log( prior_density_alphabeta );

    mp::float128 prior_log_density = prior_log_density_alpha0 + prior_log_density_alphabeta;
    mp::float128 ell = log_likelihood_garch(alpha_r, beta_r, y);

    return (prior_log_density + ell);
}


mp::float128 log_candidate_density_dirichlet(VectorX128 alpha_star, VectorX128 beta_star, VectorX128 alpha_r, VectorX128 beta_r, VectorX128 y, mp::float128 tau, mp::float128 sigma_alpha0){
    Function ddirichlet("ddirichlet");
    Function dlnorm("dlnorm");

    // GARCH(p,q) dimensions
    int p = alpha_star.size()-1;
    int q = beta_star.size();
    
    double alpha0_star_r = alpha_star[0].convert_to<double>();
    double mean_log_star = std::log( alpha_r[0].convert_to<double>() ); 
    double sd_log_star = sigma_alpha0.convert_to<double>();

    mp::float128 q_log_density_alpha0 = Rcpp::as<double>( dlnorm( alpha0_star_r, _["meanlog"] = mean_log_star, _["sdlog"] = sd_log_star, _["log"] = true ) );    

    NumericVector alphabeta_candidate_argument_r(1 + p + q);
    NumericVector candidate_concentration_r(1 + p + q);
    
    mp::float128 sum_alphabeta_candidate;
    mp::float128 sum_alphabeta_r;
    
    sum_alphabeta_candidate = 0;
    sum_alphabeta_r = 0;    
    for(int i = 0; i < p+q; i++){
        if(i < p){
            alphabeta_candidate_argument_r[i] = alpha_star[i+1].convert_to<double>();
            candidate_concentration_r[i] = (alpha_r[i+1] * tau).convert_to<double>();
            sum_alphabeta_r += alpha_r[i+1];
        }else{
            alphabeta_candidate_argument_r[i] = beta_star[i-p].convert_to<double>();
            candidate_concentration_r[i] = (beta_r[i-p] * tau).convert_to<double>();
            sum_alphabeta_r += beta_r[i-p];
        }
        // Sums the value of the current parameter into the total
        sum_alphabeta_candidate += alphabeta_candidate_argument_r[i];
    }
    alphabeta_candidate_argument_r[p+q] = (1-sum_alphabeta_candidate).convert_to<double>();
    candidate_concentration_r[p+q] = ((1-sum_alphabeta_r) * tau).convert_to<double>();

    mp::float128 q_log_density_alphabeta = std::log(Rcpp::as<double>( ddirichlet(alphabeta_candidate_argument_r, candidate_concentration_r) ));

    mp::float128 q_log_density = q_log_density_alpha0 + q_log_density_alphabeta;

    return q_log_density;
}

void sample_posterior_garch_dirichlet(int Q, VectorX128 y, mp::float128 tau, mp::float128 sigma_alpha0, VectorX128& alpha_r, VectorX128& beta_r,
                                      MatrixX128& sample_alpha, MatrixX128& sample_beta,
                                      VectorX128& log_acceptance_probs,
                                      MatrixX128& candidates_alpha, MatrixX128& candidates_beta,
                                      MatrixX128& hs){
    // Alternative posterior sampler. Here, we consider
    // alpha0 ~ Log-normal(mu = 1, sigma2 = 5^2)
    // alpha1, beta1, gamma ~ Dirichlet(1,1,1)
    // We take a MH posterior sampler, considering candidate distributions at step r+1 are given by
    // alpha0* ~ Log-normal(mu = log(alpha0^r), sigma2 = 0.05)
    // (alpha1*, beta1*, 1-alpha1*-beta1*) ~ Dirichlet( alpha1^r tau, beta1^r tau, (1-alpha1^r-beta1^r) tau )

    Function ddirichlet("ddirichlet");
    Function dlnorm("dlnorm");
    
    // GARCH(p,q) dimensions
    int p = alpha_r.size()-1;
    int q = beta_r.size();
    
    // VectorX128 h0 = compute_h(alpha_r, beta_r, y);
    VectorX128 current_h = compute_h(alpha_r, beta_r, y);
    VectorX128 new_h(y.size());

    VectorX128 concentration_tau(1+p+q);
    VectorX128 alphabeta_star_raw(p+q);
    VectorX128 alphabeta_star(p+q);

    mp::float128 alpha0_star;
    VectorX128 alpha_star(1+p);
    VectorX128 beta_star(q);

    mp::float128 floor_val = 0.0001;
    mp::float128 slack;

    mp::float128 posterior_log_ratio;
    mp::float128 candidate_log_ratio;
    mp::float128 log_accept_prob;

    mp::float128 raw_prob;
    mp::float128 sum_raw_probs;
    
    for(int m = 0; m < Q; m++){
        // Sample alpha0 from a log-normal with mean alpha0^r and sd sigma_alpha0
        alpha0_star = rlognormal0( mp::log( alpha_r[0] ), sigma_alpha0);

        sum_raw_probs = 0;
        // Populate the proposal concentrations and prevents the optimizer to have negative or concentrations too close to zero
        for(int i = 0; i < p+q; i++){
            if(i < p)
                raw_prob = alpha_r[i+1];
            else
                raw_prob = beta_r[i-p];
            
            concentration_tau[i] = std::max( raw_prob * tau, (mp::float128)0.001 );
            sum_raw_probs += raw_prob;
        }
        concentration_tau[p+q] = std::max( (1 - sum_raw_probs) * tau, (mp::float128)0.001 );
        
        // Sample (alpha1, beta1) from the Dirichlet proposal distribution
        alphabeta_star_raw = rdirichlet0( concentration_tau );

        // Transform the sampled alpha1, ..., alphap, beta1, ..., betaq to a tighter interval (avoid parameters equal to zero)
        slack = 1.0 - ( (1+p+q) * floor_val);
        alphabeta_star = floor_val + (slack * alphabeta_star_raw.array());

        alpha_star[0] = alpha0_star;
        for(int i = 0; i < p; i++)
            alpha_star[i+1] = alphabeta_star[i];
        for(int j = p; j < p+q; j++)
            beta_star[j-p] = alphabeta_star[j];

        // Obtain the acceptance probability for the new sampled observation
        posterior_log_ratio = log_posterior_density_dirichlet(alpha_star, beta_star, y) - log_posterior_density_dirichlet(alpha_r, beta_r, y);
        candidate_log_ratio = log_candidate_density_dirichlet(alpha_r, beta_r, alpha_star, beta_star, y, tau, sigma_alpha0) - log_candidate_density_dirichlet(alpha_star, beta_star, alpha_r, beta_r, y, tau, sigma_alpha0);
        log_accept_prob = posterior_log_ratio + candidate_log_ratio;

        // Throw a coin to accept or reject sampled observation
        mp::float128 log_u = mp::log( runiform() );

        // Accept new sample
        if(log_u <= log_accept_prob){
            // Obtain the corresponding conditional variances vector for new sampled observation
            new_h = compute_h(alpha_star, beta_star, y);
            // Update current_h
            current_h = new_h;
            
            // Save the values into the proper objects
            for(int i = 0; i < 1+p; i++)
                sample_alpha(m,i) = alpha_star[i];
            for(int j = 0; j < q; j++)
                sample_beta(m,j) = beta_star[j];
            for(int t = 0; t < y.size(); t++)
                hs(m,t) = new_h[t];
            
            // Update the current state to the new sampled observation
            alpha_r = alpha_star;
            beta_r = beta_star;
        // Reject new sample
        }else{
            // Save the values into the proper objects
            for(int i = 0; i < 1+p; i++)
                sample_alpha(m,i) = alpha_r[i];
            for(int j = 0; j < q; j++)
                sample_beta(m,j) = beta_r[j];
            for(int t = 0; t < y.size(); t++)
                hs(m,t) = current_h[t];
        }

        log_acceptance_probs[m] = log_accept_prob;
        for(int i = 0; i < 1+p; i++)
            candidates_alpha(m,i) = alpha_star[i];
        for(int j = 0; j < q; j++)
            candidates_beta(m,j) = beta_star[j];
    }
}

// [[Rcpp::export]]
List sample_posterior_garch_dirichlet_cpp(NumericVector y, int Q, NumericVector alpha0, NumericVector beta0, double tau, double sigma_alpha0){
    // Function to call sample_posterior_garch_dirichlet from the R environment
    int T = y.size();

    // GARCH(p,q) dimensions
    int p = alpha0.size()-1;
    int q = beta0.size();
    
    mp::float128 tau_cpp = tau;
    mp::float128 sigma_alpha0_cpp = sigma_alpha0;
    
    // Converts the variables from R types to Eigen C++ types
    VectorX128 alpha0_cpp(1+p);
    for(int i = 0; i < 1+p; i++)
        alpha0_cpp[i] = alpha0[i];
    VectorX128 beta0_cpp(q);
    for(int j = 0; j < q; j++)
        beta0_cpp[j] = beta0[j];
    VectorX128 y_cpp(T);
    for(int t = 0; t < T; t++)
        y_cpp[t] = y[t];
    
    // Create the data objects that will store the sampled posterior values
    MatrixX128 sample_alpha(Q,1+p);
    MatrixX128 sample_beta(Q,q);
    VectorX128 log_acceptance_probs(Q);
    
    MatrixX128 candidates_alpha(Q,1+p);
    MatrixX128 candidates_beta(Q,q);
    
    MatrixX128 hs(Q,T);

    sample_posterior_garch_dirichlet(Q, y_cpp, tau_cpp, sigma_alpha0_cpp, alpha0_cpp, beta0_cpp, sample_alpha, sample_beta, log_acceptance_probs, candidates_alpha, candidates_beta, hs);

    // Formats back the results to R variables
    NumericMatrix sample_alpha_r(Q,1+p);
    NumericMatrix sample_beta_r(Q,q);
    NumericVector log_acceptance_probs_r(Q);
    NumericMatrix candidates_alpha_r(Q,1+p); 
    NumericMatrix candidates_beta_r(Q,q);
    NumericMatrix hs_r(Q,T);

    for(int m = 0; m < Q; m++){
        for(int i = 0; i < 1+p; i++){
            sample_alpha_r(m,i) = sample_alpha(m,i).convert_to<double>();
            candidates_alpha_r(m,i) = candidates_alpha(m,i).convert_to<double>();
        }
        for(int j = 0; j < q; j++){
            sample_beta_r(m,j) = sample_beta(m,j).convert_to<double>();
            candidates_beta_r(m,j) = candidates_beta(m,j).convert_to<double>();
        }
        log_acceptance_probs_r[m] = log_acceptance_probs[m].convert_to<double>();
        for(int t = 0; t < T; t++)
            hs_r(m,t) = hs(m,t).convert_to<double>();
    }

    List L = List::create(
        _["alpha"] = sample_alpha_r,
        _["beta"] = sample_beta_r,
        _["log_acceptance_probs"] = log_acceptance_probs_r,
        _["candidates_alpha"] = candidates_alpha_r,
        _["candidates_beta"] = candidates_beta_r,
        _["h"] = hs_r
    );
    
    return L;
}



