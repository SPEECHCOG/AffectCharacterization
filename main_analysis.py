#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 13:19:59 2025

@author: lahtine9
"""

import numpy as np
import sys
import os
import argparse
import pandas as pd
import prosodic_features
import voice_quality_features
import experiments
import soundfile as sf
import librosa
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial import distance


SMALLER_SIZE = 10
SMALL_SIZE = 15
MEDIUM_SIZE = 20
MEDIUM_LARGE_SIZE = 25
BIG_SIZE = 35
BIGGER_SIZE = 50
width = 24
height = 10

plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_LARGE_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_LARGE_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIG_SIZE)    # legend fontsize
plt.rc('figure', titlesize=SMALL_SIZE)  # fontsize of the figure title
plt.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Arial']})


def plot_valence_LR_with_arousal_split_results(results, quantiles=None, indice_intersections=None, low_ar_label_corr=None, high_ar_label_corr=None):
    """
    Visualize 3×N results:
        results[0] = low-arousal correlations
        results[1] = high-arousal correlations
        results[2] = cosine distances
    
    Parameters
    ----------
    results : np.ndarray (3 × N)
        Array containing:
            row 0: correlations low arousal
            row 1: correlations high arousal
            row 2: cosine distances
    quantiles : list or array (optional)
        X-axis tick positions. If None, evenly spaced 0..1.
    """
    cwd = os.getcwd()
    results = np.array(results)
    
    low_corr = results[:,0]
    high_corr = results[:,1]
    cos_dist = results[:,2]

    n = np.shape(results)[0]

    if quantiles is None:
        quantiles = np.linspace(0.05, 0.95, n)

    fig, ax, = plt.subplots(figsize=(20, 8))
    
    fig2, ax2, = plt.subplots(figsize=(20, 8))

    # Plot correlation lines + points
    ax.plot(quantiles, low_corr, label="Low arousal sample group")
    ax.scatter(quantiles, low_corr, marker="o", s=np.multiply(low_ar_label_corr,1700))
    ax.plot(quantiles, high_corr, label="High arousal sample group")
    ax.scatter(quantiles, high_corr, marker="o", s=np.multiply(high_ar_label_corr,1700))
    
    ax2.plot(quantiles, low_ar_label_corr, label="Low arousal sample group")
    ax2.plot(quantiles, high_ar_label_corr, label="High arousal sample group")
    ax2.scatter(quantiles, low_ar_label_corr, marker="o")
    ax2.scatter(quantiles, high_ar_label_corr, marker="o",)
    

    # Vertical connector lines with cosine distance
    for i, q in enumerate(quantiles):
        
        if i == 0:
        
            ax.plot([q, q], [low_corr[i], high_corr[i]], 
                color="gray", linestyle="--", linewidth=1, label="LR coefficient distance")
            
        else:
            ax.plot([q, q], [low_corr[i], high_corr[i]], 
                color="gray", linestyle="--", linewidth=1)
            
        corr_diff = high_corr[i] - low_corr[i]
        
        # Position cosine distance text slightly to the right of the connector
        y_mid = (low_corr[i] + high_corr[i]) / 2
        ax.text(q + 0.005, y_mid, f"{cos_dist[i]:.2f}", fontsize=MEDIUM_SIZE, va="center")


    # Labels and formatting
    ax.set_xticks(quantiles)
    ax.set_yticks(np.arange(0.12, 0.375, 0.025))
    ax.set_xlabel("Arousal split quantile threshold")
    ax.set_ylabel("Predicted valence-valence correlation")
    #ax.set_title("Valence modelling correlations across arousal quantile splits")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    
    ax2.legend()
    ax2.set_ylabel("Valence-arousal correlation")
    ax2.set_xlabel("Arousal split quantile threshold")
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.set_yticks(np.arange(0.02, 0.2, 0.025))
    ax2.set_xticks(quantiles)
    
    legend = ax.legend()
    for line in legend.get_lines():
        line.set_linewidth(3.0)
        
    legend = ax2.legend()
    for line in legend.get_lines():
        line.set_linewidth(3.0)
    
    plt.tight_layout()
    plt.show()
    
    fig.savefig(cwd+"//valence_LR_results_high_low_arousal_splits.pdf", bbox_inches="tight",format="pdf")
    fig2.savefig(cwd+"//valence_arousal_corr_results_high_low_arousal_splits.pdf", bbox_inches="tight",format="pdf")

    
def collect_and_combine_significant_findings(correlation_df, significance_name, correlation_name, p_value_name):
    
        conditions = [
            correlation_df[p_value_name] < 0.001,
            correlation_df[p_value_name] < 0.01,
            correlation_df[p_value_name] < 0.05
        ]
        
        choices = ["***", "**", "*"]
        
        
        
        correlation_df["correlation_coef_str"] = (
           correlation_df[correlation_name].round(3).astype(str) +   # format coef
            np.select(conditions, choices, default="")
            )
        
        
        symbol_map = {"spearman": "\u03C1", "pearson": "r"}
        
        correlation_df["correlation_coef_str"] = (
            correlation_df["correlation_method"].map(symbol_map).fillna("")
            + "="
            + correlation_df["correlation_coef_str"].astype(str)
        )
            
        correlation_df["correlation_coef_str"].loc[correlation_df[significance_name].astype(bool) == False] = "-"

        return correlation_df
    

if __name__ == '__main__':
    
    ###########################################################################
    
    #Initial arguments if the script is executed directly
    if len(sys.argv) < 2:
        
        #absolute dir to the data files
        working_dir = "/Volumes/T9/LP_HP_TP_combined/Master_datasets/FinnAffect_Kielipankki"
        data_storage_dir = working_dir+"/annotations_and_metadata/"
        wavs_storage_dir = working_dir+"/wavs/"
        txts_storage_dir = working_dir+"/txts/"
        txts_filtered_storage_dir = working_dir+"/txts_filtered/"
        alignments_storage_dir = working_dir+"/webmaus_alignments/"

      
    #Initial arguments if the script is executed from the commandline
    else:
        
        CLI=argparse.ArgumentParser()
        CLI.add_argument("--dataset_storage_dir", nargs=1, type=str)
        CLI.add_argument("--wavs_storage_dir", nargs=1, type=str)    
        CLI.add_argument("--txts_storage_dir", nargs=1, type=str)    
        CLI.add_argument("--txts_filtered_storage_dir", nargs=1, type=str)    
        CLI.add_argument("--alignments_storage_dir", nargs=1, type=str)    
        
        args = CLI.parse_args()
        data_storage_dir = args.dataset_storage_dir[0]
        wavs_storage_dir = args.wavs_storage_dir[0]
        txt_storage_dir = args.txts_storage_dir[0]
        txt_filtered_storage_dir = args.txts_filtered_storage_dir[0]
        alignments_storage_dir = args.alignments_storage_dir[0]

    #get current working directory
    cwd = os.getcwd()
        
    #Read in all label types and metadata as pandas dataframes
    print("Reading in data")   
    valence_continuous_normalized = pd.read_csv(cwd+"/valence_normalized.csv")
    valence_continuous_normalized = valence_continuous_normalized.set_index("Unnamed: 0")
    
    arousal_continuous_normalized = pd.read_csv(cwd+"/arousal_normalized.csv")
    arousal_continuous_normalized = arousal_continuous_normalized.set_index("Unnamed: 0")
    
    #THE FOLLOWING DATA FILES ARE ONLY AVAILABLE THROUGH THE OFFICIAL FINNAFFECT CORPUS FROM KIELIPANKKI
    #metadata = pd.read_csv(data_storage_dir+"metadata.csv")
    #metadata = metadata.set_index("Unnamed: 0")
    
    #annotation_timestamps_data = pd.read_csv(data_storage_dir+"annotation_timestamps.csv")
    #annotation_timestamps_data = annotation_timestamps_data.set_index("Unnamed: 0")
    
    #Read in sample ids, in this case the annotated only ids (N = 12000)
    #all samples (N = 1474728)
    annotated_only_ids = valence_continuous_normalized[~valence_continuous_normalized["mean"].isna()].index.to_list()
    GS_indices = valence_continuous_normalized.dropna().index.to_list()
    all_ids = valence_continuous_normalized.index.to_list()
    
    
    #Loop through all the annotated samples and read sample audio, transcription
    #and alignment files, if the features are yet not computed
    #NOTE: For this you need the whole FinnAffect corpus, including metadata
    if not os.path.isfile(cwd+"//"+"all_features_df.csv"):
    
        
        all_features = {}
        utt_ids = []
        
        
        all_features["valence"] = []
        all_features["arousal"] = []
        
        
        all_phones = []
        
        for i, utt_id in enumerate(annotated_only_ids):
            
            print("Utterance ID: "+str(utt_id))
            print(str(i)+"/"+str(len(annotated_only_ids)))
            
            #all affect labels
            valence_label_cont_normalized = valence_continuous_normalized.iloc[[utt_id]]
            all_features["valence"].append(valence_label_cont_normalized["mean"].item())
            arousal_label_cont_normalized = arousal_continuous_normalized.iloc[[utt_id]]
            all_features["arousal"].append(arousal_label_cont_normalized["mean"].item())
            
            #sample filepaths
            wav_filepath = wavs_storage_dir+"/"+str(utt_id)+".wav"
            txt_filepath = txts_storage_dir+"/"+str(utt_id)+".txt"
            txt_filtered_filepath = txts_filtered_storage_dir+"/"+str(utt_id)+".txt"
            webmaus_phonetic_alignment_filepath = alignments_storage_dir+"/"+str(utt_id)+".TextGrid"
            
            utt_ids.append(utt_id)
            
            #read the audio sample and store original sampling rate
            with sf.SoundFile(wav_filepath) as f:
                original_sr = f.samplerate
            
            y, sr = librosa.load(wav_filepath, sr=original_sr)
            
            #compute features from audio sample
            F0_features = prosodic_features.F0_features(y, sr)
            mfcc_features = voice_quality_features.MFCC_based_features(y, webmaus_phonetic_alignment_filepath, sr)
            opensmile_features = voice_quality_features.opensmile_based_features(y, sr)
            rhythm_features = prosodic_features.rhythm_features(webmaus_phonetic_alignment_filepath)
            VSA_features = voice_quality_features.vowel_space_features(wav_filepath, webmaus_phonetic_alignment_filepath)
            formant_features = voice_quality_features.formant_features(wav_filepath, webmaus_phonetic_alignment_filepath)
            
            #combine all features into one dict structure
            combined_sample_data = {**F0_features, **mfcc_features, **opensmile_features, **rhythm_features, **VSA_features}
            
            #init keys for samples
            for key, value in combined_sample_data.items():
                
                if key not in all_features.keys():
                    all_features[key] = []
                    
                all_features[key].append(value)
        
        #get speaker metadata 
        all_features["utt_id"] = utt_ids
        speakers = metadata["speaker id"].iloc[utt_ids]
        speaker_gender = metadata["speaker gender"]
    
        #combine features and speaker metadata into one dataframe
        all_features_df = pd.DataFrame.from_dict(all_features)
        all_features_df.set_index("utt_id", inplace=True)
        all_labels_df = all_features_df[["valence", "arousal"]]
        all_features_df.drop(["valence", "arousal"], axis=1, inplace=True)
        all_features_df["speaker_gender"] = speaker_gender
        
        
        #select features that will be mean normalized by speaker gender info
        features_for_gender_normalization = ["f0_log_mean", "f0_log_var", 
                                             "f0_range", "f0_log_range",
                                             'VSA',
                                             'a_F1_mean', 
                                             'a_F2_mean',
                                             'a_F3_mean', 
                                             'a_F4_mean', 
                                             'i_F1_mean', 
                                             'i_F2_mean',
                                             'i_F3_mean', 
                                             'i_F4_mean', 
                                             'u_F1_mean',
                                             'u_F2_mean',
                                             'u_F3_mean',
                                             'u_F4_mean'
                                             ]
        
        #store normalized featuredata into a separate list
        normalized_feature_names = []
        
        for feature in features_for_gender_normalization:
            
            normalized_feature_names.append(feature+"_norm")
            
            
        all_features_df[normalized_feature_names] = all_features_df.groupby("speaker_gender")\
            [features_for_gender_normalization].transform(
                lambda x: (x - x.mean())
                )
            
        all_features_df['f0_log_range'] = all_features_df['f0_log_range'].replace([np.inf, -np.inf], np.nan)
        all_features_df['f0_log_range_norm'] = all_features_df['f0_log_range_norm'].replace([np.inf, -np.inf], np.nan)
            
        all_labels_df.to_csv(cwd+"//"+"all_labels.csv")
        all_features_df.to_csv(cwd+"//"+"all_features_df.csv")
     
    #if the features are computed, jump straight into the analysis part 
    else:
        
        all_labels_df = pd.read_csv(cwd+"//"+"all_labels.csv", index_col="utt_id")
        #all_labels_df.drop(["utt_id"], axis=1, inplace=True)
        all_features_df = pd.read_csv(cwd+"//"+"all_features_df.csv", index_col="utt_id")
        #all_features_df.drop(["utt_id"], axis=1, inplace=True)
    
    #select features for the replication experiment
    replication_study_measures = ['f0_log_mean_norm', 
                                  'f0_log_var_norm', 
                                  'f0_log_range_norm',
                                  'mfcc_0_vowel_var',
                                  'mfcc_1_vowel_mean',
                                  'speaking_rate']
    
    
    #select features for the exploration experiment
    exploration_study_measures = ['jitterLocal_sma3nz_mean', 
                                  'jitterLocal_sma3nz_var', 
                                  'shimmerLocaldB_sma3nz_mean',
                                  'shimmerLocaldB_sma3nz_var',
                                  'HNRdBACF_sma3nz_mean',
                                  'HNRdBACF_sma3nz_var',
                                  'logRelF0-H1-H2_sma3nz_mean',
                                  'logRelF0-H1-H2_sma3nz_var',
                                  'mfcc_1_vowel_var',
                                  'break_duration_mean',
                                  'voicing_to_frames_ratio',
                                  'articulation_rate',
                                  'VSA_norm',
                                  'a_F1_mean_norm', 
                                  'a_F2_mean_norm',
                                  'a_F3_mean_norm', 
                                  'a_F4_mean_norm', 
                                  'i_F1_mean_norm', 
                                  'i_F2_mean_norm',
                                  'i_F3_mean_norm', 
                                  'i_F4_mean_norm', 
                                  'u_F1_mean_norm',
                                  'u_F2_mean_norm',
                                  'u_F3_mean_norm',
                                  'u_F4_mean_norm'
                                  ]
    
    
    
    replication_features_df = all_features_df[replication_study_measures]
    exploration_features_df = all_features_df[exploration_study_measures]
    
    
    # EXPERIMENT 1: REPLICATE AND EXPLORE PREVIOUSLY FOUND ACOUSTIC AND PHONETIC MEASURES RELATED TO AFFECt
    #
    
    replication_valence_results, replication_arousal_results = experiments.replication_experiment(all_labels_df, replication_features_df)
    exploration_valence_results, exploration_arousal_results = experiments.exploration_experiment(all_labels_df, exploration_features_df)
    
    
    replication_valence_results.to_csv(cwd+"//replication_valence_results.csv")
    replication_arousal_results.to_csv(cwd+"//replication_arousal_results.csv")
    exploration_valence_results.to_csv(cwd+"//exploration_valence_results.csv")
    exploration_arousal_results.to_csv(cwd+"//exploration_arousal_results.csv")
    
    ##############################################################################
    
    # EXPERIMENT 2: PERFORM LINEAR REGRESSION USING THE SIGNIFICANT METRICS FOUND IN EXPERIMENT 1 AND
    # MEASURE CORRELATION BETWEEN PREDICTED AROUSAL / VALENCe TO GT AROUSAL / VALENCE
    
    arousal_rep_significant_metrics = replication_arousal_results[replication_arousal_results["correlation_significance"] == True]["metric"]
    arousal_exp_significant_metrics = exploration_arousal_results[exploration_arousal_results["correlation_significant_after_correction"] == 1]["metric"]
    
    arousal_significant_metrics = pd.concat([arousal_rep_significant_metrics, arousal_exp_significant_metrics])
    arousal_significant_features_df = all_features_df[arousal_significant_metrics]
    
    valence_rep_significant_metrics = replication_valence_results[replication_valence_results["correlation_significance"] == True]["metric"]
    valence_exp_significant_metrics = exploration_valence_results[exploration_valence_results["correlation_significant_after_correction"] == 1]["metric"]
    
    valence_significant_metrics = pd.concat([valence_rep_significant_metrics, valence_exp_significant_metrics])
    
    valence_significant_features_df = all_features_df[valence_significant_metrics]
    
    valence_LR_experiment_results, valence_LR_coeff_df = experiments.linear_regression_experiment(all_labels_df["valence"], valence_significant_features_df)
    arousal_LR_experiment_results, arousal_LR_coeff_df = experiments.linear_regression_experiment(all_labels_df["arousal"], arousal_significant_features_df)
    
    valence_LR_experiment_results[valence_LR_coeff_df["Feature name"]] = valence_LR_coeff_df["M"] 
    arousal_LR_experiment_results[arousal_LR_coeff_df["Feature name"]] = arousal_LR_coeff_df["M"] 
    
    valence_LR_experiment_results["annotator"] = "mean"
    arousal_LR_experiment_results["annotator"] = "mean"
    
    valence_LR_experiment_results.set_index("annotator")
    arousal_LR_experiment_results.set_index("annotator")
    
    valence_LR_experiment_results.to_csv(cwd+"//valence_LR_experiment_results.csv")
    arousal_LR_experiment_results.to_csv(cwd+"//arousal_LR_experiment_results.csv")
    
    ##############################################################################
    
    # EXPERIMENT 3: REPEAT EXPERIMENT 1, BUT USE THE GOLD STANDARD ANNOTATED SAMPLES ONLY, 
    # AND COMPARE FINDINGS 
    
    all_GS_labels_df = all_labels_df.loc[GS_indices]
    replication_GS_features_df = replication_features_df.loc[GS_indices]
    exploration_GS_features_df = exploration_features_df.loc[GS_indices] 
    
    replication_GS_valence_results, replication_GS_arousal_results = experiments.replication_experiment(all_GS_labels_df, replication_GS_features_df)
    exploration_GS_valence_results, exploration_GS_arousal_results = experiments.exploration_experiment(all_GS_labels_df, exploration_GS_features_df)
    
    
    arousal_GS_rep_significant_metrics = replication_GS_arousal_results[replication_GS_arousal_results["correlation_significance"] == True]["metric"]
    arousal_GS_exp_significant_metrics = exploration_GS_arousal_results[exploration_GS_arousal_results["correlation_significant_after_correction"] == 1]["metric"]
    
    arousal_GS_significant_metrics = pd.concat([arousal_GS_rep_significant_metrics, arousal_GS_exp_significant_metrics])
    arousal_GS_significant_features_df = all_features_df[arousal_GS_significant_metrics]
    
    valence_GS_rep_significant_metrics = replication_GS_valence_results[replication_GS_valence_results["correlation_significance"] == True]["metric"]
    valence_GS_exp_significant_metrics = exploration_GS_valence_results[exploration_GS_valence_results["correlation_significant_after_correction"] == 1]["metric"]
    
    valence_GS_significant_metrics = pd.concat([valence_GS_rep_significant_metrics, valence_GS_exp_significant_metrics])
    
    valence_GS_significant_features_df = all_features_df[valence_GS_significant_metrics]
       
    
    replication_GS_valence_results.to_csv(cwd+"//replication_GS_valence_results.csv")
    replication_GS_arousal_results.to_csv(cwd+"//replication_GS_arousal_results.csv")
    exploration_GS_valence_results.to_csv(cwd+"//exploration_GS_valence_results.csv")
    exploration_GS_arousal_results.to_csv(cwd+"//exploration_GS_arousal_results.csv")
    
    
    # EXPERIMENT 4: REPEAT EXPERIMENT 2, BUT WITH INDIVIDUAL ANNOTATOR SAMPLES
    
    annotators = ["a1", "a2", "a3", "a4", "a5"]
    
    annotator_valence_LR_results = []
    annotator_arousal_LR_results = []
    
    for annotator in annotators:
        
        annotator_valence_labels = valence_continuous_normalized[annotator].dropna()
        annotator_valence_significant_features = valence_significant_features_df.loc[annotator_valence_labels.index]
        annotator_valence_LR_experiment_results, annotator_valence_LR_coeff_df = experiments.linear_regression_experiment(annotator_valence_labels, annotator_valence_significant_features)
        annotator_valence_LR_experiment_results[annotator_valence_LR_coeff_df["Feature name"]] = annotator_valence_LR_coeff_df["M"] 
        annotator_valence_LR_experiment_results["annotator"] = annotator
        
        
        annotator_arousal_labels = arousal_continuous_normalized[annotator].dropna()
        annotator_arousal_significant_features = arousal_significant_features_df.loc[annotator_arousal_labels.index]
        annotator_arousal_LR_experiment_results, annotator_arousal_LR_coeff_df = experiments.linear_regression_experiment(annotator_arousal_labels, annotator_arousal_significant_features)
        annotator_arousal_LR_experiment_results[annotator_arousal_LR_coeff_df["Feature name"]] = annotator_arousal_LR_coeff_df["M"]  
        annotator_arousal_LR_experiment_results["annotator"] = annotator
        
        valence_row_df = annotator_valence_LR_experiment_results.copy()
        arousal_row_df = annotator_arousal_LR_experiment_results.copy()
        
        annotator_valence_LR_results.append(valence_row_df)
        annotator_arousal_LR_results.append(arousal_row_df)
        
    
    annotator_valence_LR_results_df = pd.concat(annotator_valence_LR_results, ignore_index=True)
    annotator_arousal_LR_results_df = pd.concat(annotator_arousal_LR_results, ignore_index=True)
    
    annotator_valence_LR_results_df.set_index("annotator")
    annotator_arousal_LR_results_df.set_index("annotator")
    
    annotator_valence_LR_results_df.to_csv((cwd+"//annotator_LR_valence_results.csv"))
    annotator_arousal_LR_results_df.to_csv((cwd+"//annotator_LR_arousal_results.csv"))
    
    
    
    # EXPERIMENT 5: REPEAT EXPERIMENT 1, BUT WITH INDIVIDUAL ANNOTATOR SAMPLES
    
    annotators = ["a1", "a2", "a3", "a4", "a5"]
    
    
    all_replication_valence_results = {}
    all_replication_arousal_results = {}
    all_exploration_valence_results = {}
    all_exploration_arousal_results = {}
    
    for annotator in annotators:
        
        annotator_valence_labels = valence_continuous_normalized[annotator].dropna()
        annotator_arousal_labels = arousal_continuous_normalized[annotator].dropna()
        
        annotator_labels_df = pd.DataFrame()
        annotator_labels_df["valence"] = annotator_valence_labels
        annotator_labels_df["arousal"] = annotator_arousal_labels
        
        annotator_replication_features_df = replication_features_df.loc[annotator_valence_labels.index]
        annotator_exploration_features_df = exploration_features_df.loc[annotator_valence_labels.index]

        annotator_replication_valence_results, annotator_replication_arousal_results = experiments.replication_experiment(annotator_labels_df, annotator_replication_features_df)
        annotator_exploration_valence_results, annotator_exploration_arousal_results = experiments.exploration_experiment(annotator_labels_df, annotator_exploration_features_df)
         
        all_replication_valence_results[annotator] = annotator_replication_valence_results
        all_replication_arousal_results[annotator] = annotator_replication_arousal_results
        
        all_exploration_valence_results[annotator] = annotator_exploration_valence_results
        all_exploration_arousal_results[annotator] = annotator_exploration_arousal_results
        
        
    combined_repl_val = pd.concat(all_replication_valence_results, axis=0)
    combined_repl_ar = pd.concat(all_replication_arousal_results, axis=0)
    combined_expl_val = pd.concat(all_exploration_valence_results, axis=0)
    combined_expl_ar = pd.concat(all_exploration_arousal_results, axis=0)
    
    combined_repl_val.to_csv((cwd+"//annotator_results_replication_valence.csv"))
    combined_repl_ar.to_csv((cwd+"//annotator_results_replication_arousal.csv"))
    combined_expl_val.to_csv((cwd+"//annotator_results_exploration_valence.csv"))
    combined_expl_ar.to_csv((cwd+"//annotator_results_exploration_arousal.csv"))
    
    valence_replication_final_results = {}
    arousal_replication_final_results = {}
    valence_exploration_final_results = {}
    arousal_exploration_final_results = {}
    
    replication_metrics = []
    exploration_metrics = []
    
    
    for annotator in annotators:
        
        
        ###REPLICATION
        annotator_repl_valence = combined_repl_val.loc[annotator]
        
        
        annotator_repl_valence = collect_and_combine_significant_findings(annotator_repl_valence, 
                                                                          significance_name="correlation_significance", 
                                                                          correlation_name="correlation_coefficient", 
                                                                          p_value_name="correlation_p_value")
        

        annotator_repl_arousal = combined_repl_ar.loc[annotator]
        annotator_repl_arousal = collect_and_combine_significant_findings(annotator_repl_arousal, 
                                                                          significance_name="correlation_significance", 
                                                                          correlation_name="correlation_coefficient", 
                                                                          p_value_name="correlation_p_value")
        
        replication_metrics.append(annotator_repl_valence["metric"])
        replication_metrics.append(annotator_repl_arousal["metric"])
        
        valence_replication_final_results[annotator] = annotator_repl_valence.set_index("metric")["correlation_coef_str"].to_dict()
        arousal_replication_final_results[annotator] = annotator_repl_arousal.set_index("metric")["correlation_coef_str"].to_dict()
        
        ###
        
        
        ##EXPLORATION
        
        ###
        annotator_expl_valence = combined_expl_val.loc[annotator]
        annotator_expl_valence = collect_and_combine_significant_findings(annotator_expl_valence, 
                                                                          significance_name="correlation_significant_after_correction", 
                                                                          correlation_name="correlation_coefficient", 
                                                                          p_value_name="correlation_p_value_bonferri_corrected")
    


        annotator_expl_arousal = combined_expl_ar.loc[annotator]
        annotator_expl_arousal = collect_and_combine_significant_findings(annotator_expl_arousal, 
                                                                          significance_name="correlation_significant_after_correction", 
                                                                          correlation_name="correlation_coefficient", 
                                                                          p_value_name="correlation_p_value_bonferri_corrected")
        
        exploration_metrics.append(annotator_expl_valence["metric"])
        exploration_metrics.append(annotator_expl_arousal["metric"])
        
        valence_exploration_final_results[annotator] = annotator_expl_valence.set_index("metric")["correlation_coef_str"].to_dict()
        arousal_exploration_final_results[annotator] = annotator_expl_arousal.set_index("metric")["correlation_coef_str"].to_dict()
        
        ###
        
        
    ###REPLICATION MEAN ANNOTATION RESULTS
    
    
    replication_valence_results_collected = collect_and_combine_significant_findings(replication_valence_results, 
                                                                      significance_name="correlation_significance", 
                                                                      correlation_name="correlation_coefficient", 
                                                                      p_value_name="correlation_p_value")
    replication_arousal_results_collected = collect_and_combine_significant_findings(replication_arousal_results, 
                                                                      significance_name="correlation_significance", 
                                                                      correlation_name="correlation_coefficient", 
                                                                      p_value_name="correlation_p_value")


    
    replication_metrics.append(replication_valence_results_collected["metric"])
    replication_metrics.append(replication_arousal_results_collected["metric"])
    
    valence_replication_final_results["mean"] = replication_valence_results_collected.set_index("metric")["correlation_coef_str"].to_dict()
    arousal_replication_final_results["mean"] = replication_arousal_results_collected.set_index("metric")["correlation_coef_str"].to_dict()
    
    ###
    
    
    ##EXPLORATION MEAN ANNOTATION RESULTS
    


    exploration_valence_results_collected = collect_and_combine_significant_findings(exploration_valence_results, 
                                                                      significance_name="correlation_significant_after_correction", 
                                                                      correlation_name="correlation_coefficient", 
                                                                      p_value_name="correlation_p_value_bonferri_corrected")
    exploration_arousal_results_collected = collect_and_combine_significant_findings(exploration_arousal_results, 
                                                                      significance_name="correlation_significant_after_correction", 
                                                                      correlation_name="correlation_coefficient", 
                                                                      p_value_name="correlation_p_value_bonferri_corrected")
    
    exploration_metrics.append(exploration_valence_results_collected["metric"])
    exploration_metrics.append(exploration_arousal_results_collected["metric"])
    
    valence_exploration_final_results["mean"] = exploration_valence_results_collected.set_index("metric")["correlation_coef_str"].to_dict()
    arousal_exploration_final_results["mean"] = exploration_arousal_results_collected.set_index("metric")["correlation_coef_str"].to_dict()
    
    ###
    
    ###REPLICATION GS MEAN ANNOTATION RESULTS
    
    
    replication_valence_results_GS_collected = collect_and_combine_significant_findings(replication_GS_valence_results, 
                                                                      significance_name="correlation_significance", 
                                                                      correlation_name="correlation_coefficient", 
                                                                      p_value_name="correlation_p_value")
    replication_arousal_results_GS_collected = collect_and_combine_significant_findings(replication_GS_arousal_results, 
                                                                      significance_name="correlation_significance", 
                                                                      correlation_name="correlation_coefficient", 
                                                                      p_value_name="correlation_p_value")


    
    replication_metrics.append(replication_valence_results_GS_collected["metric"])
    replication_metrics.append(replication_arousal_results_GS_collected["metric"])
    
    valence_replication_final_results["mean (GS)"] = replication_valence_results_GS_collected.set_index("metric")["correlation_coef_str"].to_dict()
    arousal_replication_final_results["mean (GS)"] = replication_arousal_results_GS_collected.set_index("metric")["correlation_coef_str"].to_dict()
    
    ###
    
    
    ##EXPLORATION GS MEAN ANNOTATION RESULTS
    


    exploration_valence_results_GS_collected = collect_and_combine_significant_findings(exploration_GS_valence_results, 
                                                                      significance_name="correlation_significant_after_correction", 
                                                                      correlation_name="correlation_coefficient", 
                                                                      p_value_name="correlation_p_value_bonferri_corrected")
    exploration_arousal_results_GS_collected = collect_and_combine_significant_findings(exploration_GS_arousal_results, 
                                                                      significance_name="correlation_significant_after_correction", 
                                                                      correlation_name="correlation_coefficient", 
                                                                      p_value_name="correlation_p_value_bonferri_corrected")
    
    exploration_metrics.append(exploration_valence_results_GS_collected["metric"])
    exploration_metrics.append(exploration_arousal_results_GS_collected["metric"])
    
    valence_exploration_final_results["mean (GS)"] = exploration_valence_results_GS_collected.set_index("metric")["correlation_coef_str"].to_dict()
    arousal_exploration_final_results["mean (GS)"] = exploration_arousal_results_GS_collected.set_index("metric")["correlation_coef_str"].to_dict()
    
    ###
    
    
        
    valence_replication_final_results_df = pd.DataFrame(valence_replication_final_results)
    arousal_replication_final_results_df = pd.DataFrame(arousal_replication_final_results)
    valence_exploration_final_results_df = pd.DataFrame(valence_exploration_final_results)
    arousal_exploration_final_results_df = pd.DataFrame(arousal_exploration_final_results)
    
    valence_replication_final_results_df.to_csv((cwd+"//replication_valence_final_results.csv"))
    arousal_replication_final_results_df.to_csv((cwd+"//replication_arousal_final_results.csv"))
    valence_exploration_final_results_df.to_csv((cwd+"//exploration_valence_final_results.csv"))
    arousal_exploration_final_results_df.to_csv((cwd+"//exploration_arousal_final_results.csv"))
    
    
    
    # EXPERIMENT 6: PERFORM LINEAR REGRESSION USING THE SIGNIFICANT METRICS FOUND IN EXPERIMENT 2 USING THE GS DATASET AND
    # MEASURE CORRELATION BETWEEN PREDICTED AROUSAL / VALENCE TO GT AROUSAL / VALENCE
    
    
    
    valence_GS_LR_experiment_results, valence_GS_LR_coeff_df = experiments.linear_regression_experiment(all_GS_labels_df["valence"], valence_significant_features_df.loc[all_GS_labels_df.index])
    arousal_GS_LR_experiment_results, arousal_GS_LR_coeff_df = experiments.linear_regression_experiment(all_GS_labels_df["arousal"], arousal_significant_features_df.loc[all_GS_labels_df.index])
    
    valence_GS_LR_experiment_results[valence_GS_LR_coeff_df["Feature name"]] = valence_GS_LR_coeff_df["M"] 
    arousal_GS_LR_experiment_results[arousal_GS_LR_coeff_df["Feature name"]] = arousal_GS_LR_coeff_df["M"] 
    
    valence_GS_LR_experiment_results["annotator"] = "mean (GS)"
    arousal_GS_LR_experiment_results["annotator"] = "mean (GS)"
    
    valence_GS_LR_experiment_results.set_index("annotator")
    arousal_GS_LR_experiment_results.set_index("annotator")
    
    valence_GS_LR_experiment_results.to_csv(cwd+"//valence_GS_LR_experiment_results.csv")
    arousal_GS_LR_experiment_results.to_csv(cwd+"//arousal_GS_LR_experiment_results.csv")
        
        
    # EXPERIMENT 7: PERFORM LINEAR REGRESSION FOR VALENCE USING THE SIGNIFICANT METRICS FOUND IN EXPERIMENT 1
    # CONTROLLING THE AROUSAL, I.E. PREDICT VALENCE USING SAMPLES FROM HIGH AND LOW AROUSAL REGIONS WITH
    # INCREASINGLY LARGE SAMPLE COUNTS
    
    
    starting_point_percentiles = np.linspace(0.05, 0.95, num=13)
    
    quantile_range_LR_valence_results = []
    
    valence_range_arousal_correlations = []
    
    index_intersection = []
    
    
    for quantile in starting_point_percentiles:
        
        arousal_low_ids = all_labels_df.loc[all_labels_df["arousal"] <= all_labels_df["arousal"].quantile(q=quantile)].index
        arousal_high_ids = all_labels_df.loc[all_labels_df["arousal"] > all_labels_df["arousal"].quantile(q=1-quantile)].index
        
        index_intersection.append(len(arousal_low_ids.intersection(arousal_high_ids)))
        
        quantile_low_ar_labels = all_labels_df.loc[arousal_low_ids]
        
        quantile_low_ar_valence_labels = all_labels_df["valence"][arousal_low_ids]
        quantile_low_ar_valence_significant_features_df = valence_significant_features_df.loc[arousal_low_ids]
        valence_low_ar_quantile_LR_results, valence_low_ar_quantile_LR_coeff_df = experiments.linear_regression_experiment(quantile_low_ar_valence_labels, quantile_low_ar_valence_significant_features_df)
        
        quantile_high_ar_labels = all_labels_df.loc[arousal_high_ids]
        
        quantile_high_ar_valence_labels = all_labels_df["valence"][arousal_high_ids]
        quantile_high_ar_valence_significant_features_df = valence_significant_features_df.loc[arousal_high_ids]
        valence_high_ar_quantile_LR_results, valence_high_ar_quantile_LR_coeff_df = experiments.linear_regression_experiment(quantile_high_ar_valence_labels, quantile_high_ar_valence_significant_features_df)
        
        low_ar_label_corr = quantile_low_ar_labels.corr()["valence"]["arousal"]
        high_ar_label_corr = quantile_high_ar_labels.corr()["valence"]["arousal"]
        
        
        quantile_range_LR_valence_results.append([valence_low_ar_quantile_LR_results, valence_low_ar_quantile_LR_coeff_df, valence_high_ar_quantile_LR_results, valence_high_ar_quantile_LR_coeff_df, low_ar_label_corr, high_ar_label_corr ])
    
    quantile_range_rs = []
    quantile_range_rs_diff = []
    quantile_range_ps = []
    quantile_range_Ns = []
    quantile_range_cosine = []
    
    quantile_range_low_label_corrs = []
    quantile_range_high_label_corrs = []
    
    for LR_low_results, LR_low_coeffs, LR_high_results, LR_high_coeffs, low_ar_label_corr, high_ar_label_corr in quantile_range_LR_valence_results:
        
        quantile_range_rs.append([np.round(LR_low_results["correlation_coefficient"].item(),3), np.round(LR_high_results["correlation_coefficient"].item(),3), distance.cosine(LR_low_coeffs["M"], LR_high_coeffs["M"])])
        
        quantile_range_ps.append([np.round(LR_low_results["p"].item(),3), np.round(LR_high_results["p"].item(),3)])
        
        quantile_range_Ns.append([LR_low_results["total_samples"].item(), LR_high_results["total_samples"].item()])    
    
        quantile_range_rs_diff.append(np.round(LR_low_results["correlation_coefficient"].item(),3) - np.round(LR_high_results["correlation_coefficient"].item(),3))
        
        quantile_range_cosine.append(distance.cosine(LR_low_coeffs["M"], LR_high_coeffs["M"]))
        
        quantile_range_low_label_corrs.append(low_ar_label_corr)
        quantile_range_high_label_corrs.append(high_ar_label_corr)
        
        
        
    plot_valence_LR_with_arousal_split_results(quantile_range_rs, quantiles=starting_point_percentiles, indice_intersections=index_intersection, low_ar_label_corr = quantile_range_low_label_corrs, high_ar_label_corr=quantile_range_high_label_corrs)
    


    
    # COMBINE LINEAR REGRESSION RESULTS INTO ONE TABLE
    
    
    valence_LR_final_results = pd.concat([annotator_valence_LR_results_df, valence_LR_experiment_results, valence_GS_LR_experiment_results])
    arousal_LR_final_results = pd.concat([annotator_arousal_LR_results_df, arousal_LR_experiment_results, arousal_GS_LR_experiment_results])
    
    valence_LR_final_results.set_index("annotator", inplace=True)
    arousal_LR_final_results.set_index("annotator", inplace=True)
    
    valence_LR_final_results["correlation_coefficient"] = valence_LR_final_results["correlation_coefficient"].round(3)
    arousal_LR_final_results["correlation_coefficient"] = arousal_LR_final_results["correlation_coefficient"].round(3)
    
    #valence_LR_final_results.transpose()
    #arousal_LR_final_results.transpose()
    
    valence_LR_final_results.transpose().to_csv(cwd+"//valence_LR_final_results.csv")
    arousal_LR_final_results.transpose().to_csv(cwd+"//arousal_LR_final_results.csv")    