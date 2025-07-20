import numpy as np
import matplotlib.pyplot as plt
import mne
from scipy.stats import gmean
 
def plot_topomap(importance, title, threshold=0.01):
    ch_names = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
                'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz']

    importance = np.asarray(importance)
    assert importance.shape[0] == len(ch_names), "Importance length mismatch"

    info = mne.create_info(ch_names=ch_names, sfreq=128, ch_types='eeg')
    montage = mne.channels.make_standard_montage('standard_1020')
    info.set_montage(montage)

    data = np.expand_dims(importance, axis=1)
    evoked = mne.EvokedArray(data, info, tmin=0)

    fig = evoked.plot_topomap(
        times=[0], time_format='',
        cmap='Reds', contours=5, show_names=True,
        sphere=(0, 0.009, 0, 0.11),
        scalings=1, show=True,
        vlim=(0,1)
    )

    print(f"\nTop channels (|importance| ≥ {threshold}) for: {title}")
    for i, imp in enumerate(importance):
        if abs(imp) >= threshold:
            print(f"{ch_names[i]} (Index {i}): Importance = {imp:.4f}")



rsa = np.load('rsa_importance_norm.npy')  
ga = np.load('ga_importance_norm.npy')
pso = np.load('pso_importance_norm.npy')
rfe = np.load('rfe_importance_norm.npy')


stacked = np.vstack([rsa, ga, pso, rfe])


stacked_norm = (stacked - stacked.min(axis=1, keepdims=True)) / \
               (stacked.max(axis=1, keepdims=True) - stacked.min(axis=1, keepdims=True) + 1e-8)



geo_mean_importance = gmean(stacked_norm, axis=0)
geo_mean_thresh = np.where(geo_mean_importance >= 0.5, geo_mean_importance, 0)
np.save('importance_geomean_thresholded.npy', geo_mean_thresh)
plot_topomap(geo_mean_thresh, title="Geometric Mean EEG Channel Map (ADHD vs Control)")



intersection = np.prod(stacked_norm, axis=0)
intersection_clipped = np.clip(intersection, 0, 1)
intersection_thresh = np.where(intersection_clipped >= 0.1, intersection_clipped, 0)
np.save('importance_intersection_thresholded.npy', intersection_thresh)
plot_topomap(intersection_thresh, title="Intersection Boosted EEG Channel Map (ADHD vs Control)")


import matplotlib.pyplot as plt
plt.show()
