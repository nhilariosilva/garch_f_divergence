# Basic and plotting packages
library(ggplot2)
library(caTools) # Running average
library(cowplot)
library(latex2exp)
t = theme(plot.title = element_text(size=26, hjust=0.5),
          axis.title = element_text(size=20),
          axis.text = element_text(size=16),
          legend.title = element_text(size = 20),
          legend.text = element_text(size = 16),
          plot.subtitle = element_text(size = 18, face="bold"))
theme_set(theme_minimal()+t)
options(repr.plot.width=16, repr.plot.height=6)

suppressMessages( library(gtools) )

# Utils packages
suppressMessages( library(dplyr) ) # Pipeline
suppressMessages( library(snow) ) # Parallel apply, sapply, lapply
suppressMessages( library(pbapply) ) # Progressbar apply, sapply, lapply
suppressMessages( library(pbmcapply) ) # Progress bar in for loop (needs to be installed manually: https://cran.r-project.org/web/packages/pbmcapply/index.html)

# Bayesian packages
suppressMessages( library(rstan) )
suppressMessages( library(sns) ) # Effective Sample Size MCMC
suppressMessages( library(coda) ) # gelman.diag

suppressMessages( library(TSA) )

suppressMessages( library(bayesGARCH) )
data(dem2gbp)

suppressMessages( library(Rmpfr) )

# Integrate R with C++
suppressMessages( library(Rcpp) )

suppressMessages( library(mvtnorm) )
suppressMessages( library(tmvmixnorm) )
suppressMessages( library(truncnorm) )

# Sampler for the GARCH(p,q) model
sample_garch <- function(T, alpha, beta) {
  # Extract the intercept (alpha0) and calculate p (ARCH lags)
  alpha0 <- alpha[1]
  p <- length(alpha) - 1
  q <- length(beta)
  
  # Initialize vectors to store the series and variances
  y <- numeric(T)
  h <- numeric(T)
  
  # Initialize the first observation
  h[1] <- alpha0
  y[1] <- rnorm(1, mean = 0, sd = sqrt(h[1]))
  
  # Loop through the remaining time steps
  for (t in 2:T) {
    # Start with the unconditional variance constant
    h_t <- alpha0
    
    # Add ARCH terms (alpha * y_{t-i}^2)
    if (p > 0) {
      for (i in 1:p) {
        if (t - i > 0) {
          # Shift index by +1 because alpha[1] is the intercept
          h_t <- h_t + alpha[i + 1] * (y[t - i]^2)
        }
      }
    }
    
    # Add GARCH terms (beta * h_{t-j})
    if (q > 0) {
      for (j in 1:q) {
        if (t - j > 0) {
          h_t <- h_t + beta[j] * h[t - j]
        }
      }
    }
    
    # Store the variance and generate the new observation
    h[t] <- h_t
    y[t] <- rnorm(1, mean = 0, sd = sqrt(h[t]))
  }
  
  return(list(y = y, h = h))
}

# Get the maximum likelihood estimation for the GARCH(p,q) model
garch_emv <- function(y, alpha0, beta0) {
  len_alpha <- length(alpha0)
  len_beta <- length(beta0)
  
  # Combine initial guesses into a single parameter vector
  theta0 <- c(alpha0, beta0)
  total_params <- len_alpha + len_beta
  
  # Objective function
  log_lik <- function(theta, y) {
    alpha <- theta[1:len_alpha]
    
    if(len_beta > 0)
      beta <- theta[(len_alpha + 1):total_params]
    else
      beta <- numeric(0)
    
    ell <- log_likelihood_garch_cpp(alpha, beta, y)
    return(ell)
  }
  
  # Generate dynamic bounds so it works for any p and q
  lower_bounds <- rep(0.0001, total_params)
  upper_bounds <- rep(Inf, total_params)
  
  BFGS_model <- optim(par = theta0, 
                      fn = log_lik, 
                      method = "L-BFGS-B", 
                      hessian = TRUE, 
                      control = list(fnscale = -1),
                      lower = lower_bounds, 
                      upper = upper_bounds, 
                      y = y)
  
  return(BFGS_model)
}

# Compila os códigos do amostrador MCMC para o GARCH(1,1)
start <- Sys.time()
Sys.setenv(PKG_LIBS = "-lmpfr -lgmp -lquadmath", PKG_CXXFLAGS="-O3 -ffast-math")
sourceCpp("posterior_sampler_eigen.cpp")
Sys.time() - start


# Scenario 1: alpha0 = 0.1, alpha1 = 0.1, beta1 = 0.8
nsim <- 500
sample_sizes <- c(500, 750, 1000)
alpha <- c(0.1, 0.1)
beta <- c(0.8)

set.seed(10)
summ_colnames <- c("T", "replica", "parameter", "mean", "median", "sd", "2.5%", "25%", "50%", "75%", "97.5%")
summ_rownames <- c("alpha0", "alpha1", "beta")
scenario1_summaries <- data.frame()

start <- Sys.time()
for(T in sample_sizes){
  for(i in 1:nsim){
    yh <- sample_garch(T, alpha = alpha, beta = beta)
    y <- yh$y
    h <- yh$h
    
    fit <- sample_posterior_garch_dirichlet_cpp(y, 10000, alpha0 = c(0.5, 0.3), beta0 = c(0.3), tau = 500, sigma_alpha0 = 0.05)
    final_alpha <- fit$alpha
    final_beta <- fit$beta
    
    summ <- data.frame(
      c(T,T,T),
      c(i,i,i),
      c("alpha0","alpha1","beta1"),
      c(mean(final_alpha[,1]), mean(final_alpha[,2]), mean(final_beta)),
      c(median(final_alpha[,1]), median(final_alpha[,2]), median(final_beta)),
      c(sd(final_alpha[,1]), sd(final_alpha[,2]), sd(final_beta)),
      c(quantile(final_alpha[,1], 0.025), quantile(final_alpha[,2], 0.025), quantile(final_beta, 0.025)),
      c(quantile(final_alpha[,1], 0.25), quantile(final_alpha[,2], 0.25), quantile(final_beta, 0.25)),
      c(quantile(final_alpha[,1], 0.5), quantile(final_alpha[,2], 0.5), quantile(final_beta, 0.5)),
      c(quantile(final_alpha[,1], 0.75), quantile(final_alpha[,2], 0.75), quantile(final_beta, 0.75)),
      c(quantile(final_alpha[,1], 0.975), quantile(final_alpha[,2], 0.975), quantile(final_beta, 0.975))
    )
    colnames(summ) <- summ_colnames
    scenario1_summaries <- rbind(scenario1_summaries, summ)
  }
}
Sys.time() - start
write.csv(scenario1_summaries, "scenario1_simulations.csv", row.names = FALSE)

scenario1_500_alpha0 <- scenario1_summaries[(scenario1_summaries$T == 500) & (scenario1_summaries$parameter == "alpha0") ,]
scenario1_750_alpha0 <- scenario1_summaries[(scenario1_summaries$T == 750) & (scenario1_summaries$parameter == "alpha0") ,]
scenario1_1000_alpha0 <- scenario1_summaries[(scenario1_summaries$T == 1000) & (scenario1_summaries$parameter == "alpha0") ,]
mean( (scenario1_500_alpha0$`2.5%` < 0.1) & (scenario1_500_alpha0$`97.5%` > 0.1) )
mean( (scenario1_750_alpha0$`2.5%` < 0.1) & (scenario1_750_alpha0$`97.5%` > 0.1) )
mean( (scenario1_1000_alpha0$`2.5%` < 0.1) & (scenario1_1000_alpha0$`97.5%` > 0.1) )

scenario1_500_alpha <- scenario1_summaries[(scenario1_summaries$T == 500) & (scenario1_summaries$parameter == "alpha1") ,]
scenario1_750_alpha <- scenario1_summaries[(scenario1_summaries$T == 750) & (scenario1_summaries$parameter == "alpha1") ,]
scenario1_1000_alpha <- scenario1_summaries[(scenario1_summaries$T == 1000) & (scenario1_summaries$parameter == "alpha1") ,]
mean( (scenario1_500_alpha$`2.5%` < 0.1) & (scenario1_500_alpha$`97.5%` > 0.1) )
mean( (scenario1_750_alpha$`2.5%` < 0.1) & (scenario1_750_alpha$`97.5%` > 0.1) )
mean( (scenario1_1000_alpha$`2.5%` < 0.1) & (scenario1_1000_alpha$`97.5%` > 0.1) )

scenario1_500_beta <- scenario1_summaries[(scenario1_summaries$T == 500) & (scenario1_summaries$parameter == "beta1") ,]
scenario1_750_beta <- scenario1_summaries[(scenario1_summaries$T == 750) & (scenario1_summaries$parameter == "beta1") ,]
scenario1_1000_beta <- scenario1_summaries[(scenario1_summaries$T == 1000) & (scenario1_summaries$parameter == "beta1") ,]
mean( (scenario1_500_beta$`2.5%` < 0.8) & (scenario1_500_beta$`97.5%` > 0.8) )
mean( (scenario1_750_beta$`2.5%` < 0.8) & (scenario1_750_beta$`97.5%` > 0.8) )
mean( (scenario1_1000_beta$`2.5%` < 0.8) & (scenario1_1000_beta$`97.5%` > 0.8) )

# Scenario 2: alpha0 = 0.1, alpha1 = 0.7, beta1 = 0.2
nsim <- 500
sample_sizes <- c(500, 750, 1000)
alpha <- c(0.1, 0.7)
beta <- c(0.2)

set.seed(10)
summ_colnames <- c("T", "replica", "parameter", "mean", "median", "sd", "2.5%", "25%", "50%", "75%", "97.5%")
summ_rownames <- c("alpha0", "alpha1", "beta")
scenario2_summaries <- data.frame()

start <- Sys.time()
for(T in sample_sizes){
  for(i in 1:nsim){
    yh <- sample_garch(T, alpha = alpha, beta = beta)
    y <- yh$y
    h <- yh$h
    
    fit <- sample_posterior_garch_dirichlet_cpp(y, 10000, alpha0 = c(0.5, 0.3), beta0 = c(0.3), tau = 500, sigma_alpha0 = 0.05)
    final_alpha <- fit$alpha
    final_beta <- fit$beta
    
    summ <- data.frame(
      c(T,T,T),
      c(i,i,i),
      c("alpha0","alpha1","beta1"),
      c(mean(final_alpha[,1]), mean(final_alpha[,2]), mean(final_beta)),
      c(median(final_alpha[,1]), median(final_alpha[,2]), median(final_beta)),
      c(sd(final_alpha[,1]), sd(final_alpha[,2]), sd(final_beta)),
      c(quantile(final_alpha[,1], 0.025), quantile(final_alpha[,2], 0.025), quantile(final_beta, 0.025)),
      c(quantile(final_alpha[,1], 0.25), quantile(final_alpha[,2], 0.25), quantile(final_beta, 0.25)),
      c(quantile(final_alpha[,1], 0.5), quantile(final_alpha[,2], 0.5), quantile(final_beta, 0.5)),
      c(quantile(final_alpha[,1], 0.75), quantile(final_alpha[,2], 0.75), quantile(final_beta, 0.75)),
      c(quantile(final_alpha[,1], 0.975), quantile(final_alpha[,2], 0.975), quantile(final_beta, 0.975))
    )
    colnames(summ) <- summ_colnames
    scenario2_summaries <- rbind(scenario2_summaries, summ)
  }
}
Sys.time() - start
write.csv(scenario2_summaries, "scenario2_simulations.csv", row.names = FALSE)

# Scenario 3: alpha0 = 0.1, alpha1 = 0.3, beta1 = 0.3
nsim <- 500
sample_sizes <- c(500, 750, 1000)
alpha <- c(0.1, 0.3)
beta <- c(0.3)

set.seed(10)
summ_colnames <- c("T", "replica", "parameter", "mean", "median", "sd", "2.5%", "25%", "50%", "75%", "97.5%")
summ_rownames <- c("alpha0", "alpha1", "beta")
scenario3_summaries <- data.frame()

start <- Sys.time()
for(T in sample_sizes){
  for(i in 1:nsim){
    yh <- sample_garch(T, alpha = alpha, beta = beta)
    y <- yh$y
    h <- yh$h
    
    fit <- sample_posterior_garch_dirichlet_cpp(y, 10000, alpha0 = c(0.5, 0.3), beta0 = c(0.3), tau = 500, sigma_alpha0 = 0.05)
    final_alpha <- fit$alpha
    final_beta <- fit$beta
    
    summ <- data.frame(
      c(T,T,T),
      c(i,i,i),
      c("alpha0","alpha1","beta1"),
      c(mean(final_alpha[,1]), mean(final_alpha[,2]), mean(final_beta)),
      c(median(final_alpha[,1]), median(final_alpha[,2]), median(final_beta)),
      c(sd(final_alpha[,1]), sd(final_alpha[,2]), sd(final_beta)),
      c(quantile(final_alpha[,1], 0.025), quantile(final_alpha[,2], 0.025), quantile(final_beta, 0.025)),
      c(quantile(final_alpha[,1], 0.25), quantile(final_alpha[,2], 0.25), quantile(final_beta, 0.25)),
      c(quantile(final_alpha[,1], 0.5), quantile(final_alpha[,2], 0.5), quantile(final_beta, 0.5)),
      c(quantile(final_alpha[,1], 0.75), quantile(final_alpha[,2], 0.75), quantile(final_beta, 0.75)),
      c(quantile(final_alpha[,1], 0.975), quantile(final_alpha[,2], 0.975), quantile(final_beta, 0.975))
    )
    colnames(summ) <- summ_colnames
    scenario3_summaries <- rbind(scenario3_summaries, summ)
  }
}
Sys.time() - start
write.csv(scenario3_summaries, "scenario3_simulations.csv", row.names = FALSE)

# 
# # Scenario 4: alpha0 = 0.5, alpha1 = 0.2, beta1 = 0.5
# nsim <- 500
# sample_sizes <- c(500, 750, 1000)
# alpha <- c(0.5, 0.2)
# beta <- c(0.5)
# 
# set.seed(10)
# summ_colnames <- c("T", "replica", "parameter", "mean", "median", "sd", "2.5%", "25%", "50%", "75%", "97.5%")
# summ_rownames <- c("alpha0", "alpha1", "beta")
# scenario4_summaries <- data.frame()
# 
# start <- Sys.time()
# for(T in sample_sizes){
#   for(i in 1:nsim){
#     yh <- sample_garch(T, alpha = alpha, beta = beta)
#     y <- yh$y
#     h <- yh$h
#     
#     fit <- sample_posterior_garch_dirichlet_cpp(y, 10000, alpha0 = c(0.5, 0.3), beta0 = c(0.3), tau = 500, sigma_alpha0 = 0.05)
#     final_alpha <- fit$alpha
#     final_beta <- fit$beta
#     
#     summ <- data.frame(
#       c(T,T,T),
#       c(i,i,i),
#       c("alpha0","alpha1","beta1"),
#       c(mean(final_alpha[,1]), mean(final_alpha[,2]), mean(final_beta)),
#       c(median(final_alpha[,1]), median(final_alpha[,2]), median(final_beta)),
#       c(sd(final_alpha[,1]), sd(final_alpha[,2]), sd(final_beta)),
#       c(quantile(final_alpha[,1], 0.025), quantile(final_alpha[,2], 0.025), quantile(final_beta, 0.025)),
#       c(quantile(final_alpha[,1], 0.25), quantile(final_alpha[,2], 0.25), quantile(final_beta, 0.25)),
#       c(quantile(final_alpha[,1], 0.5), quantile(final_alpha[,2], 0.5), quantile(final_beta, 0.5)),
#       c(quantile(final_alpha[,1], 0.75), quantile(final_alpha[,2], 0.75), quantile(final_beta, 0.75)),
#       c(quantile(final_alpha[,1], 0.975), quantile(final_alpha[,2], 0.975), quantile(final_beta, 0.975))
#     )
#     colnames(summ) <- summ_colnames
#     scenario4_summaries <- rbind(scenario4_summaries, summ)
#   }
# }
# Sys.time() - start
# write.csv(scenario4_summaries, "scenario4_simulations.csv", row.names = FALSE)
