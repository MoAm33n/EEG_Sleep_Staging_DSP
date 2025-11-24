import mne
import os
import numpy as np
import glob
import sys
import matplotlib.pyplot as plt # Included for utility but not used for plotting loop

# --- 1. CONFIGURATION ---
DATA_DIR = '/home/mohamed/Uni Stuff/Signals and Systems Project/EEG_Sleep_Staging_DSP/data' 
SAVE_PATH = os.path.join(DATA_DIR, 'processed_epochs_and_labels.npz') 

MAX_SUBJECTS = 20 # Processing limit for the initial training set

# Channels and DSP Parameters (4 Channels: 2 EEG, 1 EOG, 1 EMG)
CHANNELS_TO_LOAD = ['EEG Fpz-Cz', 'EEG Pz-Oz', 'EOG horizontal', 'EMG submental'] 
F_HIGH_PASS = 1.0  # CRITICAL FIX: Aggressive High-Pass to remove DC drift
F_LOW_PASS = 40.0 

# --- 2. MULTI-SUBJECT PIPELINE EXECUTION ---
if __name__ == '__main__':
    
    # ----------------------------------------
    # STEP 2.1: Setup and File Discovery
    # ----------------------------------------
    
    NESTED_DATA_DIR = os.path.join(DATA_DIR, 'sleep-cassette')
    search_pattern = os.path.join(NESTED_DATA_DIR, 'SC*E0-PSG.edf')
    all_psg_files = glob.glob(search_pattern) 
    all_psg_files.sort()
    
    if not all_psg_files:
        print(f"ERROR: No PSG files found matching {search_pattern}. Check your data folder content.")
        sys.exit(1)

    print(f"--- Found {len(all_psg_files)} total subjects. Processing first {MAX_SUBJECTS} subjects ---")
    
    all_X_data = [] 
    all_Y_labels = [] 
    all_subject_ids = [] 
    processed_count = 0
    
    for psg_file in all_psg_files: 
        
        if processed_count >= MAX_SUBJECTS:
            print(f"\nSTOPPING: Reached the limit of {MAX_SUBJECTS} successfully processed subjects.")
            break

        # Extract subject ID (e.g., SC4001) and derive Hypnogram file name
        subject_id = os.path.basename(psg_file).split('E0')[0]
        hypno_file_ec = os.path.join(NESTED_DATA_DIR, f'{subject_id}EC-Hypnogram.edf')
        hypno_file_ea = os.path.join(NESTED_DATA_DIR, f'{subject_id}EA-Hypnogram.edf')
        
        # Determine correct hypno file path
        if os.path.exists(hypno_file_ec):
            hypno_file = hypno_file_ec
        elif os.path.exists(hypno_file_ea):
            hypno_file = hypno_file_ea
        else:
            # Skip subject if no matching Hypnogram file is found
            continue
            
        try:
            print(f"\nProcessing Subject: {subject_id}...")
            
            # ----------------------------------------
            # STEP 2.2: Data Loading and Filtering
            # ----------------------------------------
            
            raw = mne.io.read_raw_edf(psg_file, preload=True, verbose='error')
            
            # B. Apply DSP Filtering (CRITICAL FIX: 1.0 Hz High-Pass)
            raw.filter(l_freq=F_HIGH_PASS, h_freq=F_LOW_PASS, fir_design='firwin', phase='zero', verbose='error')
            raw.pick_channels(CHANNELS_TO_LOAD)
            
            # C. Load annotations and align
            annotations = mne.read_annotations(hypno_file)
            raw.set_annotations(annotations, emit_warning=False)
            
            # ----------------------------------------
            # STEP 2.3: Epoching and Data Quality Check
            # ----------------------------------------
            
            # D. Epoching (30-second segments)
            event_mapping = {'Sleep stage W': 1, 'Sleep stage 1': 2, 'Sleep stage 2': 3, 'Sleep stage 3': 4, 
                             'Sleep stage 4': 4, 'Sleep stage R': 5, 'Sleep stage M': -1, '?': -1}
            
            events, event_id = mne.events_from_annotations(raw, event_id=event_mapping, chunk_duration=30.0)
            epochs = mne.Epochs(raw=raw, events=events, event_id=event_id, tmin=0.0, tmax=29.99, baseline=None, preload=True)
            epochs.drop_bad() 

            if len(epochs) < 100:
                raise ValueError(f"Too few valid epochs ({len(epochs)}) for reliable feature extraction.")

            # E. Extract data and labels
            X_subject_data = epochs.get_data() 
            Y_subject_labels = epochs.events[:, 2] 
            
            # F. Create the subject ID vector for this batch of epochs
            Subject_ID_Vector = np.array([subject_id] * len(X_subject_data))

            all_X_data.append(X_subject_data)
            all_Y_labels.append(Y_subject_labels)
            all_subject_ids.append(Subject_ID_Vector)
            
            processed_count += 1
            print(f"-> Subject {subject_id} Epochs Processed: {len(epochs)}")

        except Exception as e:
            # Skip any subject file that causes a reading or processing error
            print(f"-> SKIPPING Subject {subject_id} due to error: {type(e).__name__}: {e}")
            continue

    # --- 3. COMBINE AND SAVE FINAL DATA ---
    if all_X_data:
        X_final = np.concatenate(all_X_data, axis=0)
        Y_final = np.concatenate(all_Y_labels, axis=0)
        Subject_ID_Vector_Final = np.concatenate(all_subject_ids, axis=0)
        
        final_ch_names = raw.ch_names
        final_sfreq = raw.info['sfreq'] 

        # Save the final matrix in a compressed NumPy format for the next script
        np.savez(
            SAVE_PATH, 
            X_data=X_final, Y_labels=Y_final, Subject_IDs=Subject_ID_Vector_Final, 
            sfreq=final_sfreq, ch_names=final_ch_names
        )
        print("\n=======================================================")
        print(f"SUCCESS: Pipeline 01 Complete. Processed {processed_count} subjects.")
        print(f"Total Epochs for Feature Extraction: {X_final.shape[0]}")
        print(f"Saved to: {SAVE_PATH}")
        print("=======================================================")
    else:
        print("\nFAILURE: No subjects processed successfully.")