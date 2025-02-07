import numpy as np

def jackknife(data):
    n = len(data)
    jackknife_samples = np.empty((n, n-1))
    
    for i in range(n):
        jackknife_samples[i] = np.delete(data, i)
    
    return jackknife_samples

def jackknife_stats(data):
    jackknife_samples = jackknife(data)
    jackknife_means = np.mean(jackknife_samples, axis=1)
    mean = np.mean(jackknife_means)
    error = np.sqrt((len(data) - 1) * np.mean((jackknife_means - mean) ** 2))
    
    return mean, error


def grab(var):
  return var.detach().cpu().numpy()

def compute_autocorrelation(samples, max_lag=None):

    x = np.asarray(samples, dtype=float)
    N = len(x)
    if max_lag is None or max_lag >= N:
        max_lag = N - 1

    mean_x = np.mean(x)
    var_x = np.var(x)
    if np.isclose(var_x, 0.0):
        return np.ones(max_lag + 1)

    autocorr = np.zeros(max_lag + 1)
    for lag in range(max_lag + 1):
        x_front = x[:N - lag] - mean_x
        x_back  = x[lag:] - mean_x
        cov_k = np.sum(x_front * x_back) / N
        autocorr[lag] = cov_k / var_x
    
    return autocorr

def compute_ess(samples, max_lag=None):

    x = np.asarray(samples, dtype=float)
    N = len(x)
    autocorr = compute_autocorrelation(x, max_lag=max_lag)
    
    tau_int = 1.0
    for k in range(1, len(autocorr)):
        if autocorr[k] < 0:
            break
        tau_int += 2.0 * autocorr[k]

    ess = N / tau_int
    return ess, tau_int