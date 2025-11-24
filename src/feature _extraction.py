import os
import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import skew, kurtosis # Import for non-linear statistics
import mne 

# --- CONFIGURATION (Must match the pipeline setup) ---
DATA_DIR = '/home/mohamed/Uni Stuff/Signals and Systems Project/EEG_Sleep_Staging_DSP/data' 
LOAD_PATH = os.path.join(DATA_DIR, 'processed_epochs_and_labels.npz')
SAVE_PATH = os.path.join(DATA_DIR, 'feature_matrix.csv')

# DSP PARAMETERS
F_HIGH_PASS_CORRECTION = 1.0 
F_LOW_PASS = 40.0 

# Frequency Bands (5 Bands)
BAND_LIMITS = {
    "Delta": [1.0, 4], 
    "Theta": [4, 8],
    "Alpha": [8, 13],
    "Beta": [13, 30],
    "Sigma": [11, 16] 
}


# --- 2. LOAD CLEANED EPOCHS AND METADATA ---
try:
    print("--- 2.1 Loading Cleaned Data ---")
    data = np.load(LOAD_PATH, allow_pickle=True)
    
    X_data = data['X_data'] # 3D array (Epochs, Channels, Samples)
    Y_labels = data['Y_labels']
    SFREQ = data['sfreq'].item() 
    CH_NAMES = data['ch_names']
    
    # We ignore the subject_IDs field if it exists, as we are doing a random split.
    
except Exception as e:
    print(f"\n[CRITICAL ERROR]: Could not load data from {LOAD_PATH}. Details: {e}")
    print("Please ensure 01_data_pipeline.py has run successfully and created the .npz file.")
    exit()


# --- 3. FEATURE EXTRACTION FUNCTION (PHASE 3: DSP CORE) ---
def calculate_hjorth(data):
    """Calculates Hjorth Parameters: Activity (variance) and Mobility (mean freq)."""
    activity = np.var(data)
    mobility = np.sqrt(np.var(np.diff(data)) / activity) if activity > 0 else 0
    return activity, mobility

def calculate_spectral_entropy(psd, total_power):
    """Calculates Spectral Entropy (measures flatness/randomness of the power spectrum)."""
    if total_power == 0:
        return 0
    # Normalize power spectrum for probability distribution
    prob_dist = psd / total_power
    # Entropy formula: -sum(p * log2(p))
    entropy = -np.sum(prob_dist * np.log2(prob_dist + 1e-12))
    return entropy

def extract_all_features(data_matrix, sfreq, ch_names, band_limits):
    
    n_epochs, n_channels, n_samples = data_matrix.shape
    feature_list = []
    
    for epoch_idx in range(n_epochs):
        features_per_epoch = []
        
        for ch_idx in range(n_channels):
            data_epoch = data_matrix[epoch_idx, ch_idx, :]
            
            # --- TIME DOMAIN FEATURES ---
            act, mob = calculate_hjorth(data_epoch)
            features_per_epoch.extend([act, mob])
            
            # --- FREQUENCY DOMAIN FEATURES (BAND POWER) ---
            freqs, psd_epoch = welch(
                data_epoch, fs=sfreq, nperseg=int(sfreq * 2), noverlap=int(sfreq)
            )
            
            # Sum of power across the relevant spectrum for normalization
            total_power = np.sum(psd_epoch[(freqs >= F_HIGH_PASS_CORRECTION) & (freqs <= F_LOW_PASS)])
            
            # Add Spectral Entropy (Non-Linear Feature)
            features_per_epoch.append(calculate_spectral_entropy(psd_epoch, total_power))

            # Extract Band Power Features
            for f_min, f_max in band_limits.values():
                idx_band = (freqs >= f_min) & (freqs <= f_max)
                band_power = np.sum(psd_epoch[idx_band])
                
                features_per_epoch.append(band_power / total_power if total_power > 0 else 0.0)
                
        feature_list.append(features_per_epoch)

    # 3. Create the 2D Feature Matrix (X)
    # Total features: 3 channels * (2 Hjorth + 1 Entropy + 5 Band Powers) = 24 features!
    base_features = ['Activity', 'Mobility', 'Entropy']
    band_names = list(band_limits.keys())
    
    column_names = []
    for ch in ch_names:
        column_names.extend([f"{ch}_{feat}" for feat in base_features])
        column_names.extend([f"{ch}_{band}" for band in band_names])
        
    feature_df = pd.DataFrame(feature_list, columns=column_names)
    
    return feature_df.values


# --- 4. APPLICATION AND SAVE ---
if __name__ == '__main__':
    
    print("--- 3.1 Extracting Features (Creating 24 Features/Epoch) ---")
    # Total features are now 3 channels * 8 features/channel = 24 features!
    X_features = extract_all_features(X_data, SFREQ, CH_NAMES, BAND_LIMITS)
    
    # Combine features (X), labels (Y)
    feature_df = pd.DataFrame(X_features)
    feature_df['target'] = Y_labels
    
    # Ensure subject_id column exists for classification script (even if only placeholder)
    if 'subject_id' not in feature_df.columns:
        feature_df['subject_id'] = np.arange(len(feature_df)) # Placeholder ID column

    # Save the final matrix to CSV for use in the classifier script
    feature_df.to_csv(SAVE_PATH, index=False)

    print(f"\nSUCCESS: Feature Matrix created with 24 enhanced features.")
    print(f"Final Matrix Shape: {feature_df.shape}")