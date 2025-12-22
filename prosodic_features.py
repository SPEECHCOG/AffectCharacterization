#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 16 11:37:04 2025

@author: lahtine9
"""

import librosa
import numpy as np
import tgt
import os

def F0_features(sample, sr, fmin=70, fmax=550, frame_size_ms=93):
    
    frame_size_samples = int(sr*(frame_size_ms/1000))
    
    f0, voicing, voicing_probability = librosa.pyin(y=sample, sr=sr, fmin=fmin, fmax=fmax, frame_length=frame_size_samples)
    
    return {"f0_mean": np.nanmean(f0),
            "f0_log_mean": np.nanmean(np.log(f0)),
            "f0_var": np.nanvar(f0),
            "f0_log_var": np.nanvar(np.log(f0)),
            "f0_range": (np.nanmax(f0) - np.nanmin(f0)),
            "f0_log_range": (np.nanmax(np.log(f0) - np.nanmin(np.log(f0)))),
            "voicing_prob_mean": np.mean(voicing_probability),
            "voicing_freq": np.sum(voicing),
            "total_voicing_frames": len(voicing),
            "voicing_to_frames_ratio": np.sum(voicing)/len(voicing)}

    
    
    
def check_for_vowels(interval_text):
    
    vowels_xsampa = {"a", "e", "i", "o", "u", "y", "2", "9", "{"}
    
    
    for xsampa_symbol in vowels_xsampa:
        
        if xsampa_symbol in interval_text:
            return True
        

    return False
    
    

def rhythm_features(sample_textgrid_file):
    
    if not os.path.isfile(sample_textgrid_file):
        return {"vowel_tiers_count": np.nan,
                "consonant_tiers_count": np.nan,
                "break_tiers_count": np.nan,
                "phone_duration_mean": np.nan,
                "break_duration_mean": np.nan,
                "phone_duration_var": np.nan,
                "break_duration_var": np.nan,
                "speech_total_duration": np.nan,
                "articulation_rate": np.nan,
                "speaking_rate": np.nan}
    
    
    grid = tgt.io.read_textgrid(filename=sample_textgrid_file)
    phonetic_tier = grid.get_tier_by_name("MAU")
    
    vowel_tiers = []
    consonant_tiers = []
    break_tiers = []
    
    phone_durations = []
    break_durations = []
    
    for interval in phonetic_tier:
        
        if interval.text == "<p:>":
            break_tiers.append(interval)
            break_durations.append(interval.duration())
            continue
        
        if check_for_vowels(interval.text):
            vowel_tiers.append(interval)
        else:
            consonant_tiers.append(interval)


        phone_durations.append(interval.duration())    
    
    
    if len(vowel_tiers) == 0:
        vowel_tiers = [np.nan]
        
    if len(consonant_tiers) == 0:
        consonant_tiers = [np.nan]
        
    if len(break_tiers) == 0:
        break_tiers = [np.nan]
        break_durations = [np.nan]
        
    if len(phone_durations) == 0:
        phone_durations = [np.nan]
        
    
    return {"vowel_tiers_count": len(vowel_tiers),
            "consonant_tiers_count": len(consonant_tiers),
            "break_tiers_count": len(break_tiers),
            "phone_duration_mean": np.mean(phone_durations),
            "break_duration_mean": np.mean(break_durations),
            "phone_duration_var": np.var(phone_durations),
            "break_duration_var": np.var(break_durations),
            "speech_total_duration": np.sum(phone_durations)+np.sum(break_durations),
            "articulation_rate": (len(phone_durations)/np.sum(phone_durations)),
            "speaking_rate": (len(phone_durations)/(np.sum(phone_durations)+np.sum(break_durations)))}


