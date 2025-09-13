import mne
import matplotlib
matplotlib.use('TkAgg')  # Ensure plotting works
from mne.preprocessing import ICA
from mne.time_frequency import psd_array_welch
# from mne.decoding import CSP  # Comment out CSP for now
# from sklearn.svm import SVC  # Comment out SVM for now
import numpy as np

# Load sample data for testing (simulate 2-4 channels later)
data_path = mne.datasets.sample.data_path()
raw = mne.io.read_raw_fif(data_path / 'MEG' / 'sample' / 'sample_audvis_raw.fif', preload=True)
raw.crop(tmax=60)  # Limit to 60 seconds to reduce lag
raw.pick_types(eeg=True)  # Select only EEG channels for testing
raw.filter(8, 30, fir_design='firwin')  # 8-30 Hz band-pass for mu/beta rhythms
raw.plot()  # Visualize filtered data
input("Press Enter to continue...")  # Pause to keep windows open

# ICA for artifact removal
ica = ICA(n_components=4, random_state=42)
ica.fit(raw)
raw = ica.apply(raw)  # Apply ICA to remove artifacts
raw.plot()  # Visualize after ICA
input("Press Enter to continue...")  # Pause again

# Event detection (manual stim channel for sample data)
stim_channel = 'STI 014'  # Try the sample data's known stim channel
try:
    events = mne.find_events(raw, stim_channel=stim_channel)
    if len(events) == 0:
        print(f"No events found on {stim_channel}. Skipping epoching.")
    else:
        epochs = mne.Epochs(raw, events, tmin=-0.2, tmax=1.0, baseline=None)
except ValueError:
    print(f"No {stim_channel} channel found. Skipping epoching and using raw data.")
    events = None

# FFT for power spectral density (use raw data with adjusted parameters)
psds, freqs = psd_array_welch(raw.get_data(), sfreq=raw.info['sfreq'], fmin=8, fmax=30, n_fft=int(raw.info['sfreq'] * 1))  # 1-second window
print("PSD calculated for frequencies:", freqs)

# CSP and SVM (commented out until real data with multiple classes is available)
# csp = CSP(n_components=4, reg='empirical', log=True)
# features = csp.fit_transform(raw.get_data().T[:4, :], y=np.zeros(4))  # Dummy labels
# print("CSP features shape:", features.shape)
#
# svm = SVC(kernel='rbf', C=1.0)
# # svm.fit(features, labels)  # Uncomment after adding real labels
# # print("SVM trained")  # Uncomment after adding labels