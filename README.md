# ADHD-classification
This repository contains all code, scripts, and documentation for our pipeline to classify ADHD vs. control EEG and identify relevant neurophysiological biomarkers using a rigorously validated deep learning workflow.

# Quick-Start
- Python 3.8+

- PyTorch

- NumPy

- scikit-learn

- mne

- matplotlib

- geneticalgorithm, pyswarm, seaborn

#Setting up

- use the command below to install the dependencies
  ```pip install torch numpy scikit-learn mne matplotlib seaborn geneticalgorithm pyswarm```

- The EEG dataset used in this project is publicly available at:
  [IEEE DataPort: EEG data from ADHD and control children](https://ieee-dataport.org/open-access/eeg-data-adhd-control-children)
  Due to licensing and ethical restrictions, preprocessed files are not shared here.
  Please download the raw dataset from the above link and follow the provided preprocessing steps to generate the required format.
  - 1. Loaded raw EEG files (.mat format) using MNE.
  - 2. Applied a 1–45 Hz bandpass FIR filter.
  - 3. Re-referenced to the average of all channels.
  - 4. Independent component analysis(ICA) of signals.
  - 5. Segmented data into 2-second non-overlapping epochs.
  - 6. Removed epochs with peak-to-peak amplitude > 150 µV (artifact rejection).
  - 7. Normalized each epoch (z-score per channel).
  - 8. Saved the resulting data in `.set` format using EEGLAB.

- Train the supervised 1D autoencoder and save model weights.
  ```python 1D_autoencoder.py```

- Feature selection and classification
  ```python classification_pipeline.py```

- Biomarker identification
  ```python biomarker_identification.py```
  ```python single_channel_importance.py```

#Outputs:

Preprocessing would result in .set files, which would be utilized by ```classification_pipeline.py``` as well as ```1D_autoencoder.py```. Store the preprocessed .set files in "./ADHD_cleaned/" and "./Control_cleaned/".
```classification_pipeline.py``` will result in 10 feature indices for all four of the algorithms, use ```combined_feature_indices.py```(change ```base_filename``` as per convenience) to create a single indice for each Feature selection algorithm.
```biomarker_identification.py```(change ```INDICES_PATH``` as per convinience) would result in normalized channel_importance vectors for all four Feature Selection algorithms; use ```single_channel_importance.py``` to combine all four channel_importance vectors.
