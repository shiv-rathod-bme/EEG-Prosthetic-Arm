import mne
import serial
from joblib import load
import numpy as np

model = load('models/svm_eeg.joblib')
raw = mne.io.read_raw_openbci('demo/test_data.eeg', preload=True)

def extract_features(raw):
    raw.filter(8., 30., fir_design='firwin')
    data = raw.get_data()
    return np.mean(data, axis=1).reshape(1, -1)

features = extract_features(raw)
command = model.predict(features)[0]
ser = serial.Serial('COM3', 9600, timeout=1)
ser.write(command.encode())
ser.close()
print(f"Sent command: {command}")