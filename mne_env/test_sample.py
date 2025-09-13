import mne
import matplotlib
matplotlib.use('TkAgg')  # Force Tkinter backend
data_path = mne.datasets.sample.data_path()
raw = mne.io.read_raw_fif(data_path / 'MEG' / 'sample' / 'sample_audvis_raw.fif')
fig = raw.plot()  # Store the figure object
input("Press Enter to close the plot...")  # Pause the script until Enter is pressed