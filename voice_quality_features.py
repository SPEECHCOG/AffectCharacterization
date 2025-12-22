#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 19 13:47:12 2025

@author: lahtine9
"""

import librosa
import opensmile
import numpy as np
import parselmouth
import pandas as pd
import tgt
import os
import matplotlib.pyplot as plt

smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.LowLevelDescriptors)

def plot_triangle(points, save_dir, filename="triangle.png"):
    """
    Plot a triangle given three 2D points and save the image.

    Parameters
    ----------
    points : list of tuples
        List of three (x, y) coordinate pairs, e.g. [(0,0), (1,0), (0,1)]
    save_dir : str
        Directory where the plot should be saved.
    filename : str, optional
        Name of the saved image file (default: "triangle.png")
    """
    if len(points) != 3:
        raise ValueError("Exactly three points are required")

    # Unpack points
    xs, ys = zip(*points)

    # Close the triangle by repeating the first point
    xs_closed = list(xs) + [xs[0]]
    ys_closed = list(ys) + [ys[0]]

    # Create plot
    fig, ax = plt.subplots()
    ax.plot(xs_closed, ys_closed, marker="o", linestyle="-", color="b")
    
    point_labels = ["a", "i", "u"]
    
    # Label the points
    for i, (x, y) in enumerate(points):
        ax.text(x, y, point_labels[i], fontsize=12, ha="right", va="bottom")

    # Axis scaling: linear with ticks visible
    ax.set_xlabel("F1")
    ax.set_ylabel("F2")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True)

    # Ensure directory exists
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)

    # Save and close
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Triangle plot saved to: {save_path}")
    
def safe_get_value(formants, idx, t):
    v = formants.get_value_at_time(idx, t)
    return np.round(v) if v is not None else np.nan

def get_formants(filepath, frame_length = 0.025):
    sound = parselmouth.Sound(filepath)
    
    #This command takes five arguments: the time step, 
    #the maximum number of formants, the maximum hertz, the window length, 
    #and the dynamic range (in decibels)
    formants = parselmouth.praat.call(sound, "To Formant (burg)", 0, 5, 5500, frame_length, 50.0)
    
    df = []
    
    for t in formants.ts():    
        
        unit = {'time': t,
                'f1': safe_get_value(formants, 1, t),
                'f2': safe_get_value(formants, 2, t),
                'f3': safe_get_value(formants, 3, t),
                'f4': safe_get_value(formants, 4, t)}
        
        df.append(unit)
        
    df = pd.DataFrame(df)

    return df


def compute_2D_triangle_area(p1, p2, p3):
    
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    return 0.5 * abs(x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))


def check_for_vowels(interval_text):
    
    vowels_xsampa = {"a", "e", "i", "o", "u", "y", "2", "9", "{"}
    
    
    for xsampa_symbol in vowels_xsampa:
        
        if xsampa_symbol in interval_text:
            return True
        

    return False

def MFCC_based_features(sample, sample_textgrid_file, sr, n_fft = 512):
    
    
    mfcc = librosa.feature.mfcc(y=sample, sr=sr, n_fft=n_fft)
    hop_length = n_fft // 4
    mfcc_frames = np.shape(mfcc)[1]
    
    
    mfcc_timesteps = librosa.frames_to_time(np.arange(mfcc.shape[1]),
                                       sr=sr,
                                       hop_length=hop_length,
                                       n_fft=n_fft)
    
    
    if not os.path.isfile(sample_textgrid_file):
        vowel_mfcc0s = [np.nan]
        vowel_mfcc1s = [np.nan]
        vowel_mfcc2s = [np.nan]
        vowel_mfcc3s = [np.nan]
        
    else:
        grid = tgt.io.read_textgrid(filename=sample_textgrid_file)
        phonetic_tier = grid.get_tier_by_name("MAU")
        vowel_mfcc0s = []
        vowel_mfcc1s = []
        vowel_mfcc2s = []
        vowel_mfcc3s = []
        
        
        for interval in phonetic_tier:
            
            if check_for_vowels(interval.text):
                
                start_time = interval.start_time
                end_time = interval.end_time
                
                start_time_index, end_time_index = find_mfcc_time_range(mfcc_timesteps, start_time, end_time)
                
                if start_time_index > end_time_index:
                    start_time_index, end_time_index = end_time_index, start_time_index
                    
                end_time_index = end_time_index + 1    
                
                vowel_mfcc0s = vowel_mfcc0s + mfcc[0, start_time_index:end_time_index].tolist()
                vowel_mfcc1s = vowel_mfcc1s + mfcc[1, start_time_index:end_time_index].tolist()
                vowel_mfcc2s = vowel_mfcc2s + mfcc[2, start_time_index:end_time_index].tolist()
                vowel_mfcc3s = vowel_mfcc3s + mfcc[3, start_time_index:end_time_index].tolist()
            
    
    return {"mfcc_0_mean": np.mean(mfcc[0,:]),
            "mfcc_0_var": np.var(mfcc[0,:]),
            "mfcc_1_mean": np.mean(mfcc[1,:]),
            "mfcc_1_var": np.var(mfcc[1,:]),
            "mfcc_2_mean": np.mean(mfcc[2,:]),
            "mfcc_2_var": np.var(mfcc[2,:]),
            "mfcc_3_mean": np.mean(mfcc[3,:]),
            "mfcc_3_var": np.var(mfcc[3,:]),
            "mfcc_0_vowel_mean": np.mean(vowel_mfcc0s),
            "mfcc_0_vowel_var": np.var(vowel_mfcc0s),
            "mfcc_1_vowel_mean": np.mean(vowel_mfcc1s),
            "mfcc_1_vowel_var": np.var(vowel_mfcc1s),
            "mfcc_2_vowel_mean": np.mean(vowel_mfcc2s),
            "mfcc_2_vowel_var": np.var(vowel_mfcc2s),
            "mfcc_3_vowel_mean": np.mean(vowel_mfcc3s),
            "mfcc_3_vowel_var": np.var(vowel_mfcc3s)}



def opensmile_based_features(sample, sr):
    
    
    opensmile_result = smile.process_signal(sample, sr)
    
    return {"jitterLocal_sma3nz_mean": opensmile_result["jitterLocal_sma3nz"].mean(),
            "jitterLocal_sma3nz_var": opensmile_result["jitterLocal_sma3nz"].var(),
            "shimmerLocaldB_sma3nz_mean": opensmile_result["shimmerLocaldB_sma3nz"].mean(),
            "shimmerLocaldB_sma3nz_var": opensmile_result["shimmerLocaldB_sma3nz"].var(),
            "HNRdBACF_sma3nz_mean": opensmile_result["HNRdBACF_sma3nz"].mean(),
            "HNRdBACF_sma3nz_var": opensmile_result["HNRdBACF_sma3nz"].var(),
            "logRelF0-H1-H2_sma3nz_mean": opensmile_result["logRelF0-H1-H2_sma3nz"].mean(),
            "logRelF0-H1-H2_sma3nz_var": opensmile_result["logRelF0-H1-H2_sma3nz"].var(),
            "logRelF0-H1-A3_sma3nz_mean": opensmile_result["logRelF0-H1-A3_sma3nz"].mean(),
            "logRelF0-H1-A3_sma3nz_var": opensmile_result["logRelF0-H1-A3_sma3nz"].var()}
            

def find_formant_time_range(formants, start_time, end_time):
    
    df_closest_start = formants.iloc[(formants["time"]-start_time).abs().argsort()[:1]]
    df_closest_end = formants.iloc[(formants["time"]-end_time).abs().argsort()[:1]]
    
    
    return df_closest_start.index.item(), df_closest_end.index.item()

def find_mfcc_time_range(mfcc_timesteps, start_time, end_time):
    
    mfcc_timesteps_df = pd.DataFrame(mfcc_timesteps, columns = ["time"])
    
    df_closest_start = mfcc_timesteps_df.iloc[(mfcc_timesteps_df["time"]-start_time).abs().argsort()[:1]]
    df_closest_end = mfcc_timesteps_df.iloc[(mfcc_timesteps_df["time"]-end_time).abs().argsort()[:1]]
    
    
    return df_closest_start.index.item(), df_closest_end.index.item()

def vowel_space_features(sample_audio_filepath, sample_textgrid_file):
    
    sample_name = sample_textgrid_file.split("/")[-1].split(".")[0]
    
    formants = get_formants(sample_audio_filepath)
    
    if not os.path.isfile(sample_textgrid_file):
        return {"VSA": np.nan,
                "a_F1_mean": np.nan,
                "a_F2_mean": np.nan,
                "i_F1_mean": np.nan,
                "i_F2_mean": np.nan,
                "u_F1_mean": np.nan,
                "u_F2_mean": np.nan,
                "a_F3_mean": np.nan,
                "a_F4_mean": np.nan,
                "i_F3_mean": np.nan,
                "i_F4_mean": np.nan,
                "u_F3_mean": np.nan,
                "u_F4_mean": np.nan,
                "a_F1_var": np.nan,
                "a_F2_var": np.nan,
                "i_F1_var": np.nan,
                "i_F2_var": np.nan,
                "u_F1_var": np.nan,
                "u_F2_var": np.nan,
                "a_F3_var": np.nan,
                "a_F4_var": np.nan,
                "i_F3_var": np.nan,
                "i_F4_var": np.nan,
                "u_F3_var": np.nan,
                "u_F4_var": np.nan}
    
    
    grid = tgt.io.read_textgrid(filename=sample_textgrid_file)
    phonetic_tier = grid.get_tier_by_name("MAU")
    
    a_F1 = []
    a_F2 = []
    a_F3 = []
    a_F4 = []
    i_F1 = []
    i_F2 = []
    i_F3 = []
    i_F4 = []
    u_F1 = []
    u_F2 = []
    u_F3 = []
    u_F4 = []
    
    for interval in phonetic_tier:
        
        if "a" in interval.text:
            
            start_time = interval.start_time
            end_time = interval.end_time
            
            start_time_index, end_time_index = find_formant_time_range(formants, start_time, end_time)
            
            a_F1.append(formants["f1"].iloc[start_time_index:end_time_index+1].mean())
            a_F2.append(formants["f2"].iloc[start_time_index:end_time_index+1].mean())
            a_F3.append(formants["f3"].iloc[start_time_index:end_time_index+1].mean())
            a_F4.append(formants["f4"].iloc[start_time_index:end_time_index+1].mean())
            
        if "i" in interval.text:
            
            start_time = interval.start_time
            end_time = interval.end_time
            
            start_time_index, end_time_index = find_formant_time_range(formants, start_time, end_time)
            
            i_F1.append(formants["f1"].iloc[start_time_index:end_time_index+1].mean())
            i_F2.append(formants["f2"].iloc[start_time_index:end_time_index+1].mean())
            i_F3.append(formants["f3"].iloc[start_time_index:end_time_index+1].mean())
            i_F4.append(formants["f4"].iloc[start_time_index:end_time_index+1].mean())
            
        if "u" in interval.text:
            
            start_time = interval.start_time
            end_time = interval.end_time
            
            start_time_index, end_time_index = find_formant_time_range(formants, start_time, end_time)
            
            u_F1.append(formants["f1"].iloc[start_time_index:end_time_index+1].mean())
            u_F2.append(formants["f2"].iloc[start_time_index:end_time_index+1].mean())
            u_F3.append(formants["f3"].iloc[start_time_index:end_time_index+1].mean())
            u_F4.append(formants["f4"].iloc[start_time_index:end_time_index+1].mean())



    

    a_F1_mean = np.mean(a_F1)
    a_F2_mean = np.mean(a_F2)
    a_F3_mean = np.mean(a_F3)
    a_F4_mean = np.mean(a_F4)
    
    i_F1_mean = np.mean(i_F1)
    i_F2_mean = np.mean(i_F2)
    i_F3_mean = np.mean(i_F3)
    i_F4_mean = np.mean(i_F4)
    
    u_F1_mean = np.mean(u_F1)
    u_F2_mean = np.mean(u_F2)
    u_F3_mean = np.mean(u_F3)
    u_F4_mean = np.mean(u_F4)
    
    a_F1_var = np.var(a_F1)
    a_F2_var = np.var(a_F2)
    a_F3_var = np.var(a_F3)
    a_F4_var = np.var(a_F4)
    
    i_F1_var = np.var(i_F1)
    i_F2_var = np.var(i_F2)
    i_F3_var = np.var(i_F3)
    i_F4_var = np.var(i_F4)
    
    u_F1_var = np.var(u_F1)
    u_F2_var = np.var(u_F2)
    u_F3_var = np.var(u_F3)
    u_F4_var = np.var(u_F4)
    
    if not any(np.isnan((a_F1_mean, a_F2_mean) + (i_F1_mean, i_F2_mean)+ (u_F1_mean, u_F2_mean))):
    
        VSA = compute_2D_triangle_area((a_F1_mean, a_F2_mean), (i_F1_mean, i_F2_mean), (u_F1_mean, u_F2_mean))
        
        plot_triangle([(a_F1_mean, a_F2_mean), (i_F1_mean, i_F2_mean), (u_F1_mean, u_F2_mean)], "/Volumes/T9/testi/VSA_plots", sample_name+".png")
        
        
    else:
        
        VSA = np.nan
    
    
    return {"VSA": VSA,
            "a_F1_mean": a_F1_mean,
            "a_F2_mean": a_F2_mean,
            "a_F3_mean": a_F3_mean,
            "a_F4_mean": a_F4_mean,
            "i_F1_mean": i_F1_mean,
            "i_F2_mean": i_F2_mean,
            "i_F3_mean": i_F3_mean,
            "i_F4_mean": i_F4_mean,
            "u_F1_mean": u_F1_mean,
            "u_F2_mean": u_F2_mean,
            "u_F3_mean": u_F3_mean,
            "u_F4_mean": u_F4_mean,
            "a_F1_var": a_F1_var,
            "a_F2_var": a_F2_var,
            "a_F3_var": a_F3_var,
            "a_F4_var": a_F4_var,
            "i_F1_var": i_F1_var,
            "i_F2_var": i_F2_var,
            "i_F3_var": i_F3_var,
            "i_F4_var": i_F4_var,
            "u_F1_var": u_F1_var,
            "u_F2_var": u_F2_var,
            "u_F3_var": u_F3_var,
            "u_F4_var": u_F4_var}




def formant_features(sample_audio_filepath, sample_textgrid_file):
    
    sample_name = sample_textgrid_file.split("/")[-1].split(".")[0]
    
    formants = get_formants(sample_audio_filepath)
    
    if not os.path.isfile(sample_textgrid_file):
        return {"F1_mean": np.nan,
                "F2_mean": np.nan,
                "F3_mean": np.nan,
                "F4_mean": np.nan,
                "F1_var": np.nan,
                "F2_var": np.nan,
                "F3_var": np.nan,
                "F4_var": np.nan}
    
    
    grid = tgt.io.read_textgrid(filename=sample_textgrid_file)
    phonetic_tier = grid.get_tier_by_name("MAU")
    
    F1 = []
    F2 = []
    F3 = []
    F4 = []
    
    for interval in phonetic_tier:
        
        if check_for_vowels(interval.text):
            
            start_time = interval.start_time
            end_time = interval.end_time
            
            start_time_index, end_time_index = find_formant_time_range(formants, start_time, end_time)
            
            F1.append(formants["f1"].iloc[start_time_index:end_time_index+1].mean())
            F2.append(formants["f2"].iloc[start_time_index:end_time_index+1].mean())
            F3.append(formants["f3"].iloc[start_time_index:end_time_index+1].mean())
            F4.append(formants["f4"].iloc[start_time_index:end_time_index+1].mean())
        
    

    F1_mean = np.mean(F1)
    F2_mean = np.mean(F2)
    F3_mean = np.mean(F3)
    F4_mean = np.mean(F4)
    
    F1_var = np.var(F1)
    F2_var = np.var(F2)
    F3_var = np.var(F3)
    F4_var = np.var(F4)

    
    
    
    return {"F1_mean": F1_mean,
            "F2_mean": F2_mean,
            "F3_mean": F3_mean,
            "F4_mean": F4_mean,
            "F1_var": F1_var,
            "F2_var": F2_var,
            "F3_var": F3_var,
            "F4_var": F4_var}
    
def get_phone_characters(sample_textgrid_file):
    
    sample_name = sample_textgrid_file.split("/")[-1].split(".")[0]
    
    if not os.path.isfile(sample_textgrid_file):
        return {"all_phones": []}
    
    
    grid = tgt.io.read_textgrid(filename=sample_textgrid_file)
    phonetic_tier = grid.get_tier_by_name("MAU")
    
    phones = []
    
    for interval in phonetic_tier:
        
       phones.append(interval.text)

    return {"all_phones": phones}