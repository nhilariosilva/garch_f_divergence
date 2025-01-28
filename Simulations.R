# Basic and plotting packages
library(ggplot2)
library(caTools) # Running average
library(cowplot)
library(latex2exp)

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
suppressMessages( library(Rmpfr) )

# Integrate R with C++
suppressMessages( library(Rcpp) )

suppressMessages( library(mvtnorm) )
suppressMessages( library(tmvmixnorm) )
suppressMessages( library(truncnorm) )


# ----- Rcpp Code compilation -----
# Code for parallelized divergence computing
Sys.setenv(PKG_LIBS = "-lmpfr -lgmp -lquadmath")
sourceCpp("divergence_parallel.cpp")

# Code to sample the posterior for the GARCH(1,1) via MCMC (ARDIA - 2008)
Sys.setenv(PKG_LIBS = "-lmpfr -lgmp -lquadmath")
sourceCpp("multifloat_precision_eigen.cpp")

# Amostrador do modelo GARCH(1,1)
sample_garch <- function(T, alpha0, alpha1, beta){
    h <- alpha0
    y <- rnorm(1, 0, sqrt(h))
    for(t in 2:T){
        h[t] <- alpha0 + alpha1 * y[t-1]^2 + beta * h[t-1]
        y[t] <- rnorm(1, 0, sqrt(h[t]))
    }
    return(list(y = y, h = h))
}

# Obtém a estimativa de máxima verossimilhança para o modelo GARCH(1,1)
garch_emv <- function(y, alpha0, beta0){
    theta0 <- c(alpha0, beta0)
    log_lik <- function(theta, y){
        alpha <- theta[1:2]
        beta <- theta[3]
        ell <- log_likelihood_garch_cpp(alpha, beta, y)
        return(ell)
    }
    BFGS_model <- optim(theta0, log_lik, method="L-BFGS-B", hessian=TRUE, control=list(fnscale=-1),
                        lower=c(0.0001,0.0001,0.0001), upper=c(Inf, Inf, Inf), y=y)
    return(BFGS_model)
}

# Utiliza o pacote rstan para a obtenção da distribuição a posteriori
fit_garch_ardia <- function(y, niter = 10000, burnin = 5000, nchains = 2, seed = 42, emv_guess = c(0.5, 0.5, 0.5)){
    T = length(y)
    
    alpha_guess <- emv_guess[1:2]
    beta_guess <- emv_guess[3]
    emv <- garch_emv(y, alpha0 = alpha_guess, beta0 = beta_guess)
    alpha0_emv <- emv$par[1]
    alpha1_emv <- emv$par[2]
    beta_emv <- emv$par[3]
    
    # -------------------------------- Valores iniciais de cada cadeia --------------------------------
    alpha0 <- c(alpha0_emv, alpha1_emv)
    beta0 <- beta_emv
    
    # -------------------------------- Fit the Bayesian model --------------------------------
    sample_alpha_beta <- sample_posterior_garch_cpp(y, niter, alpha0, beta0)
    
    # -------------------------------- Extract the posterior distribution samples --------------------------------
    sample_alpha0 <- sample_alpha_beta$alpha[,1]
    sample_alpha1 <- sample_alpha_beta$alpha[,2]
    sample_beta <- sample_alpha_beta$beta
    sample_h <- sample_alpha_beta$h
    
    # -------------------------------- Amostras com o burnin já retirado --------------------------------
    final_alpha0 <- sample_alpha0[(burnin+1):niter]
    final_alpha1 <- sample_alpha1[(burnin+1):niter]
    final_beta <- sample_beta[(burnin+1):niter]
    final_h <- sample_h[(burnin+1):niter,]
    
    shuffle_indexes <- sample(1:length(final_alpha0))
    final_alpha0 <- final_alpha0[shuffle_indexes]
    final_alpha1 <- final_alpha1[shuffle_indexes]
    final_beta <- final_beta[shuffle_indexes]
    final_h <- final_h[shuffle_indexes,]
    
    summ_colnames <- c("mean", "median", "sd", "2.5%", "25%", "50%", "75%", "97.5%")
    summ_rownames <- c("alpha0", "alpha1", "beta")
    summ <- data.frame(
        c(mean(final_alpha0), mean(final_alpha1), mean(final_beta)),
        c(median(final_alpha0), median(final_alpha1), median(final_beta)),
        c(sd(final_alpha0), sd(final_alpha1), sd(final_beta)),
        c(quantile(final_alpha0, 0.025), quantile(final_alpha1, 0.025), quantile(final_beta, 0.025)),
        c(quantile(final_alpha0, 0.25), quantile(final_alpha1, 0.25), quantile(final_beta, 0.25)),
        c(quantile(final_alpha0, 0.5), quantile(final_alpha1, 0.5), quantile(final_beta, 0.5)),
        c(quantile(final_alpha0, 0.75), quantile(final_alpha1, 0.75), quantile(final_beta, 0.75)),
        c(quantile(final_alpha0, 0.975), quantile(final_alpha1, 0.975), quantile(final_beta, 0.975))
    )
    colnames(summ) <- summ_colnames
    rownames(summ) <- summ_rownames
    
    list(
        "alpha0_complete" = sample_alpha0,
        "alpha1_complete" = sample_alpha1,
        "beta_complete" = sample_beta,
        "h_complete" = sample_h,
        "alpha0" = final_alpha0,
        "alpha1" = final_alpha1,
        "beta" = final_beta,
        "h" = final_h,
        "summary" = summ,
        "fit_chain1" = sample_alpha_beta,
        "emv" = emv
    )
}

garch_div_sim <- function(nsim, T, alpha0, alpha1, beta, disturbed = NULL, L = 5, niter = 10000, burnin = 5000, seed = 42){
    set.seed(seed)
    
    divergences <- list()
    summaries <- list()
    emvs <- list()
    ys <- list()
    sampling_times <- c()
    divergence_times <- c()
    total_times <- c()
    
    pb <- progressBar(min = 0, max = nsim)
    for(i in 1:nsim){
        # -------------------------------- Sample from the GARCH model --------------------------------
        yh <- sample_garch(T, alpha0, alpha1, beta)
        y <- yh$y
        h <- yh$h
        
        # -------------------------------- Disturb all indexes required --------------------------------
        for(j in disturbed){
            y[j] <- y[j] + L*sqrt(h[j])*sign(y[j])
        }
        
        # -------------------------------- Fit the Bayesian model --------------------------------
        main_start <- Sys.time()
        
        start <- Sys.time()
        fit_obj <- fit_garch_ardia(y, niter = niter, burnin = burnin, seed = seed+1, emv_guess = c(alpha0, alpha1, beta))
        sampling_time <- Sys.time() - start
        
        # -------------------------------- Extract parameters --------------------------------
        thetas <- cbind(fit_obj$alpha0, fit_obj$alpha1, fit_obj$beta)
        hs <- fit_obj$h
        colnames(thetas) <- c("alpha0", "alpha1", "beta")
        
        summaries[[i]] <- fit_obj$summary
        
        # -------------------------------- Obtain the divergences --------------------------------
        start <- Sys.time()
        div <- psi_divergences_cpp_parallel(thetas, hs, y, p = 1, q = 1, k_values = 1:T, verbose = FALSE)
        divergence_time <- Sys.time() - start
        
        total_time <- Sys.time() - main_start
        
        # -------------------------------- Save the relevant values --------------------------------
        sampling_times[i] <- sampling_time
        divergence_times[i] <- divergence_time
        total_times[i] <- total_time
        colnames(div) <- c("KL", "KL_rev", "J", "chi2", "chi2_rev", "chi2_sym", "L1", "H", "JS")
        divergences[[i]] <- div
        ys[[i]] <- y
        emvs[[i]] <- fit_obj$emv
        
        # Invoke the Garbage collector for memory maintenance
        invisible(gc())
        setTxtProgressBar(pb, i)
    }
    close(pb)
    
    return(list(
        "ys" = ys,
        "divergences" = divergences,
        "summaries" = summaries,
        "emvs" = emvs,
        "sampling_times" = sampling_times,
        "divergence_times" = divergence_times,
        "total_times" = total_times,
        "pars" = c(alpha0, alpha1, beta)
    ))
}

# ----- Simulations -----
nsim <- 250
niter <- 10000
burnin <- 5000
seed <- 200

# Scenario 1 (alpha_0 = 0.1)
sim1.1_1000 <- garch_div_sim(nsim = nsim, T = 1000, alpha0 = 0.1, alpha1 = 0.1, beta = 0.85, disturbed = NULL, L = 5,
                            niter = niter, burnin = burnin, seed = seed)
save("sim1.1_1000", file = "Final Simulations/sim1.1.RData")

sim1.2_1000 <- garch_div_sim(nsim = nsim, T = 1000, alpha0 = 0.1, alpha1 = 0.1, beta = 0.6, disturbed = NULL, L = 5,
                            niter = niter, burnin = burnin, seed = seed)
save("sim1.2_1000", file = "Final Simulations/sim1.2.RData")

sim1.3_1000 <- garch_div_sim(nsim = nsim, T = 1000, alpha0 = 0.1, alpha1 = 0.1, beta = 0.35, disturbed = NULL, L = 5,
                            niter = niter, burnin = burnin, seed = seed)
save("sim1.4_1000", file = "Final Simulations/sim1.6.RData")

sim1.4_1000 <- garch_div_sim(nsim = nsim, T = 1000, alpha0 = 0.1, alpha1 = 0.1, beta = 0.1, disturbed = NULL, L = 5,
                            niter = niter, burnin = burnin, seed = seed)
save("sim1.6_1000", file = "Final Simulations/sim1.4.RData")

sim1.5_1000 <- garch_div_sim(nsim = nsim, T = 1000, alpha0 = 0.1, alpha1 = 0.35, beta = 0.6, disturbed = NULL, L = 5,
                            niter = niter, burnin = burnin, seed = seed)
save("sim1.5_1000", file = "Final Simulations/sim1.5.RData")

sim1.6_1000 <- garch_div_sim(nsim = nsim, T = 1000, alpha0 = 0.1, alpha1 = 0.35, beta = 0.35, disturbed = NULL, L = 5,
                            niter = niter, burnin = burnin, seed = seed)
save("sim1.6_1000", file = "Final Simulations/sim1.6.RData")

sim1.7_1000 <- garch_div_sim(nsim = nsim, T = 1000, alpha0 = 0.1, alpha1 = 0.35, beta = 0.1, disturbed = NULL, L = 5,
                            niter = niter, burnin = burnin, seed = seed)
save("sim1.7_1000", file = "Final Simulations/sim1.7.RData")

sim1.8_1000 <- garch_div_sim(nsim = nsim, T = 1000, alpha0 = 0.1, alpha1 = 0.6, beta = 0.35, disturbed = NULL, L = 5,
                            niter = niter, burnin = burnin, seed = seed)
save("sim1.8_1000", file = "Final Simulations/sim1.8.RData")

sim1.9_1000 <- garch_div_sim(nsim = nsim, T = 1000, alpha0 = 0.1, alpha1 = 0.6, beta = 0.1, disturbed = NULL, L = 5,
                            niter = niter, burnin = burnin, seed = seed)
save("sim1.9_1000", file = "Final Simulations/sim1.9.RData")

sim1.10_1000 <- garch_div_sim(nsim = nsim, T = 1000, alpha0 = 0.1, alpha1 = 0.85, beta = 0.1, disturbed = NULL, L = 5,
                            niter = niter, burnin = burnin, seed = seed)
save("sim1.10_1000", file = "Final Simulations/sim1.10.RData")

# Scenario 2 (alpha_0 = 1.0)
sim2.1_1000 <- garch_div_sim(nsim = nsim, T = 1000, alpha0 = 1.0, alpha1 = 0.1, beta = 0.85, disturbed = NULL, L = 5,
                            niter = niter, burnin = burnin, seed = seed)
save("sim2.1_1000", file = "Final Simulations/sim2.1.RData")

sim2.2_1000 <- garch_div_sim(nsim = nsim, T = 1000, alpha0 = 1.0, alpha1 = 0.1, beta = 0.6, disturbed = NULL, L = 5,
                            niter = niter, burnin = burnin, seed = seed)
save("sim2.2_1000", file = "Final Simulations/sim2.2.RData")

sim2.3_1000 <- garch_div_sim(nsim = nsim, T = 1000, alpha0 = 1.0, alpha1 = 0.1, beta = 0.35, disturbed = NULL, L = 5,
                            niter = niter, burnin = burnin, seed = seed)
save("sim2.4_1000", file = "Final Simulations/sim2.6.RData")

sim2.4_1000 <- garch_div_sim(nsim = nsim, T = 1000, alpha0 = 1.0, alpha1 = 0.1, beta = 0.1, disturbed = NULL, L = 5,
                            niter = niter, burnin = burnin, seed = seed)
save("sim2.6_1000", file = "Final Simulations/sim2.4.RData")

sim2.5_1000 <- garch_div_sim(nsim = nsim, T = 1000, alpha0 = 1.0, alpha1 = 0.35, beta = 0.6, disturbed = NULL, L = 5,
                            niter = niter, burnin = burnin, seed = seed)
save("sim2.5_1000", file = "Final Simulations/sim2.5.RData")

sim2.6_1000 <- garch_div_sim(nsim = nsim, T = 1000, alpha0 = 1.0, alpha1 = 0.35, beta = 0.35, disturbed = NULL, L = 5,
                            niter = niter, burnin = burnin, seed = seed)
save("sim2.6_1000", file = "Final Simulations/sim2.6.RData")

sim2.7_1000 <- garch_div_sim(nsim = nsim, T = 1000, alpha0 = 1.0, alpha1 = 0.35, beta = 0.1, disturbed = NULL, L = 5,
                            niter = niter, burnin = burnin, seed = seed)
save("sim2.7_1000", file = "Final Simulations/sim2.7.RData")

sim2.8_1000 <- garch_div_sim(nsim = nsim, T = 1000, alpha0 = 1.0, alpha1 = 0.6, beta = 0.35, disturbed = NULL, L = 5,
                            niter = niter, burnin = burnin, seed = seed)
save("sim2.8_1000", file = "Final Simulations/sim2.8.RData")

sim2.9_1000 <- garch_div_sim(nsim = nsim, T = 1000, alpha0 = 1.0, alpha1 = 0.6, beta = 0.1, disturbed = NULL, L = 5,
                            niter = niter, burnin = burnin, seed = seed)
save("sim2.9_1000", file = "Final Simulations/sim2.9.RData")

sim2.10_1000 <- garch_div_sim(nsim = nsim, T = 1000, alpha0 = 1.0, alpha1 = 0.85, beta = 0.1, disturbed = NULL, L = 5,
                            niter = niter, burnin = burnin, seed = seed)
save("sim2.10_1000", file = "Final Simulations/sim2.10.RData")

# Scenario 3 (alpha_0 = 2.0)
sim3.1_1000 <- garch_div_sim(nsim = nsim, T = 1000, alpha0 = 2.0, alpha1 = 0.1, beta = 0.85, disturbed = NULL, L = 5,
                            niter = niter, burnin = burnin, seed = seed)
save("sim3.1_1000", file = "Final Simulations/sim3.1.RData")

sim3.2_1000 <- garch_div_sim(nsim = nsim, T = 1000, alpha0 = 2.0, alpha1 = 0.1, beta = 0.6, disturbed = NULL, L = 5,
                            niter = niter, burnin = burnin, seed = seed)
save("sim3.2_1000", file = "Final Simulations/sim3.2.RData")

sim3.3_1000 <- garch_div_sim(nsim = nsim, T = 1000, alpha0 = 2.0, alpha1 = 0.1, beta = 0.35, disturbed = NULL, L = 5,
                            niter = niter, burnin = burnin, seed = seed)
save("sim3.4_1000", file = "Final Simulations/sim3.6.RData")

sim3.4_1000 <- garch_div_sim(nsim = nsim, T = 1000, alpha0 = 2.0, alpha1 = 0.1, beta = 0.1, disturbed = NULL, L = 5,
                            niter = niter, burnin = burnin, seed = seed)
save("sim3.6_1000", file = "Final Simulations/sim3.4.RData")

sim3.5_1000 <- garch_div_sim(nsim = nsim, T = 1000, alpha0 = 2.0, alpha1 = 0.35, beta = 0.6, disturbed = NULL, L = 5,
                            niter = niter, burnin = burnin, seed = seed)
save("sim3.5_1000", file = "Final Simulations/sim3.5.RData")

sim3.6_1000 <- garch_div_sim(nsim = nsim, T = 1000, alpha0 = 2.0, alpha1 = 0.35, beta = 0.35, disturbed = NULL, L = 5,
                            niter = niter, burnin = burnin, seed = seed)
save("sim3.6_1000", file = "Final Simulations/sim3.6.RData")

sim3.7_1000 <- garch_div_sim(nsim = nsim, T = 1000, alpha0 = 2.0, alpha1 = 0.35, beta = 0.1, disturbed = NULL, L = 5,
                            niter = niter, burnin = burnin, seed = seed)
save("sim3.7_1000", file = "Final Simulations/sim3.7.RData")

sim3.8_1000 <- garch_div_sim(nsim = nsim, T = 1000, alpha0 = 2.0, alpha1 = 0.6, beta = 0.35, disturbed = NULL, L = 5,
                            niter = niter, burnin = burnin, seed = seed)
save("sim3.8_1000", file = "Final Simulations/sim3.8.RData")

sim3.9_1000 <- garch_div_sim(nsim = nsim, T = 1000, alpha0 = 2.0, alpha1 = 0.6, beta = 0.1, disturbed = NULL, L = 5,
                            niter = niter, burnin = burnin, seed = seed)
save("sim3.9_1000", file = "Final Simulations/sim3.9.RData")

sim3.10_1000 <- garch_div_sim(nsim = nsim, T = 1000, alpha0 = 2.0, alpha1 = 0.85, beta = 0.1, disturbed = NULL, L = 5,
                            niter = niter, burnin = burnin, seed = seed)
save("sim3.10_1000", file = "Final Simulations/sim3.10.RData")
