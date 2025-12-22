#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 15:42:43 2025

@author: lahtine9
"""
import pandas as pd
import numpy as np
from scipy.stats import kstest
from scipy.stats import pearsonr, spearmanr
import statsmodels

def safe_corr(x, y, method="pearson"):
    # convert to arrays
    x = np.array(x)
    y = np.array(y)

    # remove NaNs pairwise
    mask = ~np.isnan(x) & ~np.isnan(y)
    x_clean = x[mask]
    y_clean = y[mask]
    
    total_samples = len(x_clean)

    # if not enough data, return NaN
    if len(x_clean) < 2:
        return np.nan, np.nan

    # compute correlation
    if method == "pearson":
        corr_coef, p = pearsonr(x_clean, y_clean)
        
        return corr_coef, p, total_samples
    elif method == "spearman":
        corr_coef, p = spearmanr(x_clean, y_clean)
        return corr_coef, p, total_samples
    else:
        raise ValueError("method must be 'pearson' or 'spearman'")
        
def linear_regression(annotation_scores, significant_features):
    
    F = significant_features.to_numpy()

    dims = np.shape(F)
    
    F_normalized = np.zeros(dims)
    
    for dim in range(dims[1]):
        
        f = F[:,dim]
        
        f_u = np.mean(f)
        
        f_std = np.std(f)
        
        f_norm = (f - f_u) / f_std
        
        F_normalized[:,dim] = f_norm
    
    
    scores = annotation_scores.to_numpy()
    
    M = np.linalg.pinv(F_normalized)@scores
    
    predicted_scores = F_normalized@M
    
    LR_coeff = {"M": M, "Feature name": significant_features.columns}
    LR_coeff_df = pd.DataFrame(LR_coeff)
    
    return predicted_scores, LR_coeff_df
    

def select_correlation_method_ks(df, distance_threshold=0.05):
    """
    For each column in a DataFrame, decide whether to use Pearson or Spearman
    correlation based on Kolmogorov–Smirnov normality testing.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with numeric features.
    alpha : float
        Significance level for the KS-test (default 0.05).
    
    Returns
    -------
    pandas.DataFrame
        Table with KS test p-values and recommended correlation method.
    """
    results = []

    for col in df.columns:
        data = df[col].dropna()  # remove missing values
        
        if len(data) < 20:
            results.append({
                "metric": col,
                "reason": "n<20",
                "ks_p": np.nan,
                "method": "spearman"
            })
            continue

        # Standardize before KS test (mean=0, std=1)
        standardized = (data - data.mean()) / data.std(ddof=0)

        # KS test against standard normal
        stat, pval = kstest(standardized, 'norm')

        # Normal if p > alpha
        if stat <= distance_threshold:
            method = "pearson"
        else:
            method = "spearman"

        results.append({
            "metric": col,
            "ks_stat": stat,
            "ks_p": pval,
            "method": method
        })

    return pd.DataFrame(results)

def holm_bonferroni(p, alpha=0.05):
    """
    Holm-Bonferroni correction for multiple hypothesis testing.
    
    Parameters
    ----------
    p : array-like
        Vector of p-values.
    alpha : float
        Significance level (default = 0.05).
    
    Returns
    -------
    siglevels : np.ndarray
        Holm-Bonferroni significance thresholds for each test.
    h : np.ndarray
        Hypothesis test decisions (1 = reject H0, 0 = fail to reject).
    adjusted_p : np.ndarray
        Adjusted p-values.
    """

    p = np.asarray(p).flatten()  # Ensure input is a 1D array
    m = len(p)

    # Sort p-values
    psort = np.sort(p)
    i = np.argsort(p)

    # Holm-Bonferroni significance thresholds
    siglev = np.zeros(m)
    for k in range(m):
        siglev[k] = alpha / (m - k)

    # Adjusted p-values
    adjusted_p = np.zeros(m)
    for k in range(m):
        j = np.arange(0, k+1)  # 0..k
        adjusted_p[k] = min(1, np.max((m - j) * psort[j]))

    # Hypothesis test decisions
    h = np.zeros(m, dtype=int)
    a = np.argmax(psort > siglev)  # first index where p > threshold
    if psort[a] > siglev[a]:
        h[i[:a]] = 1
    else:
        h[:] = 1

    # Reorder siglevels and adjusted_p to match original input order
    siglevels = np.zeros(m)
    adjusted_ps = np.zeros(m)
    for k in range(m):
        siglevels[i[k]] = siglev[k]
        adjusted_ps[i[k]] = adjusted_p[k]

    return siglevels, h, adjusted_ps

    

def replication_experiment(annotations, replication_metrics):
    
    correlation_methods_ks = select_correlation_method_ks(replication_metrics)
    correlation_methods_ks_annotations = select_correlation_method_ks(annotations)
    print(correlation_methods_ks)
    
    
    valence_results = {}
    arousal_results = {}
    
    valence_results["correlation_coefficient"] = []
    valence_results["correlation_method"] = []
    valence_results["metric"] = []
    valence_results["correlation_p_value"] = []
    valence_results["correlation_significance"] = []
    valence_results["total_samples"] = []
    
    arousal_results["correlation_coefficient"] = []
    arousal_results["correlation_method"] = []
    arousal_results["metric"] = []
    arousal_results["correlation_p_value"] = []
    arousal_results["correlation_significance"] = []
    arousal_results["total_samples"] = []
    
    
    for metric in replication_metrics.columns:
        selected_correlation_data = correlation_methods_ks.loc[correlation_methods_ks["metric"] == metric]
        
        correlation_method_metric = selected_correlation_data["method"].item()
        
        for annotation_target in annotations.columns:
            
            correlation_method_annotation = correlation_methods_ks_annotations.loc[correlation_methods_ks_annotations["metric"] == annotation_target]["method"].item()
        
            if correlation_method_annotation == "spearman":
                correlation_method = "spearman"
                
            else:
                correlation_method = selected_correlation_data["method"].item()
                
        
            if correlation_method == "pearson":
                
                corr_coef, p, total_samples = safe_corr(replication_metrics[metric], annotations[annotation_target], method="pearson")
                
            if correlation_method == "spearman":
                
                corr_coef, p, total_samples = safe_corr(replication_metrics[metric], annotations[annotation_target], method="spearman")

            
            if annotation_target == "valence":
                
                valence_results["correlation_coefficient"].append(corr_coef)
                valence_results["correlation_method"].append(correlation_method)
                valence_results["metric"].append(metric)
                valence_results["correlation_p_value"].append(p)
                valence_results["correlation_significance"].append(p < 0.05)
                valence_results["total_samples"].append(total_samples)
                
            if annotation_target == "arousal":
                
                arousal_results["correlation_coefficient"].append(corr_coef)
                arousal_results["correlation_method"].append(correlation_method)
                arousal_results["metric"].append(metric)
                arousal_results["correlation_p_value"].append(p)
                arousal_results["correlation_significance"].append(p < 0.05)
                arousal_results["total_samples"].append(total_samples)

            
        
    
    return pd.DataFrame(valence_results), pd.DataFrame(arousal_results)
        
    
    
def exploration_experiment(annotations, exploration_metrics):
    
    correlation_methods_ks = select_correlation_method_ks(exploration_metrics)
    correlation_methods_ks_annotations = select_correlation_method_ks(annotations)
    print(correlation_methods_ks)
    
    
    valence_results = {}
    arousal_results = {}
    
    valence_results["correlation_coefficient"] = []
    valence_results["correlation_method"] = []
    valence_results["metric"] = []
    valence_results["correlation_p_value"] = []
    valence_results["correlation_significant_after_correction"] = []
    valence_results["correlation_p_value_bonferri_corrected"] = []
    valence_results["correlation_bonferri_siglevels"] = []
    valence_results["total_samples"] = []
    
    arousal_results["correlation_coefficient"] = []
    arousal_results["correlation_method"] = []
    arousal_results["metric"] = []
    arousal_results["correlation_p_value"] = []
    arousal_results["correlation_significant_after_correction"] = []
    arousal_results["correlation_p_value_bonferri_corrected"] = []
    arousal_results["correlation_bonferri_siglevels"] = []
    arousal_results["total_samples"] = []
    
    
    for metric in exploration_metrics.columns:
        selected_correlation_data = correlation_methods_ks.loc[correlation_methods_ks["metric"] == metric]
        
        correlation_method_metric = selected_correlation_data["method"].item()
        
        for annotation_target in annotations.columns:
            
            correlation_method_annotation = correlation_methods_ks_annotations.loc[correlation_methods_ks_annotations["metric"] == annotation_target]["method"].item()
        
            if correlation_method_annotation == "spearman":
                correlation_method = "spearman"
                
            else:
                correlation_method = selected_correlation_data["method"].item()
        
            if correlation_method == "pearson":
                
                corr_coef, p, total_samples = safe_corr(exploration_metrics[metric], annotations[annotation_target], method="pearson")
                
            if correlation_method == "spearman":
                
                corr_coef, p, total_samples = safe_corr(exploration_metrics[metric], annotations[annotation_target], method="spearman")

            
            if annotation_target == "valence":
                
                valence_results["correlation_coefficient"].append(corr_coef)
                valence_results["correlation_method"].append(correlation_method)
                valence_results["metric"].append(metric)
                valence_results["correlation_p_value"].append(p)
                valence_results["total_samples"].append(total_samples)
                
            if annotation_target == "arousal":
                
                arousal_results["correlation_coefficient"].append(corr_coef)
                arousal_results["correlation_method"].append(correlation_method)
                arousal_results["metric"].append(metric)
                arousal_results["correlation_p_value"].append(p)
                arousal_results["total_samples"].append(total_samples)
                
                
    
    valence_siglevels, valence_h, valence_adjustedps = holm_bonferroni(valence_results["correlation_p_value"])
    
    valence_results["correlation_significant_after_correction"] = valence_h
    valence_results["correlation_p_value_bonferri_corrected"] = valence_adjustedps
    valence_results["correlation_bonferri_siglevels"] = valence_siglevels
    
    arousal_siglevels, arousal_h, arousal_adjustedps = holm_bonferroni(arousal_results["correlation_p_value"])
    
    arousal_results["correlation_significant_after_correction"] = arousal_h
    arousal_results["correlation_p_value_bonferri_corrected"] = arousal_adjustedps
    arousal_results["correlation_bonferri_siglevels"] = arousal_siglevels

    
    return pd.DataFrame(valence_results), pd.DataFrame(arousal_results)



def linear_regression_experiment(annotations, significant_features):
    
    significant_features_nan_imputated = significant_features.fillna(significant_features.mean(numeric_only=True))
    
    predicted_scores, LR_coeff_df = linear_regression(annotations, significant_features_nan_imputated)

    corr_coef, p, total_samples = safe_corr(annotations, predicted_scores, method="pearson")
    
    return pd.DataFrame({"correlation_coefficient": [corr_coef], "p": [p], "total_samples": [total_samples]}), LR_coeff_df.round(3)

        
    
    
    