import sounddevice as sd
import numpy as np
import scipy.signal as sig
from scipy.io import wavfile
import time

# Setup (must match other script)
frame_size = 128
builtin_delay = 0.05

# Setup (general)
duration = 60  # seconds to run

# Load system data
fs, _ = wavfile.read('input_recording.wav')
data = np.load('system_analysis.npz')
precalculated_filter_coeffs = data['filter_coeffs']
L = len(precalculated_filter_coeffs)  # Filter Order from previous script

# Calculate delay from settings
audio_data = np.zeros((duration + 2)*fs)
builtin_delay_samples = int(builtin_delay * fs) + frame_size
audio_idx = builtin_delay_samples

# FxLMS algorithm (not quite correct but *should* work as far as I know, just convergence would be slow)
def fxLMS(mic_input, adaptive_filter_coeffs, filter_state, mu=0.1):
    x_filtered, filter_state = sig.lfilter(precalculated_filter_coeffs, 1, mic_input, zi=filter_state)
    for i in range(L, len(x_filtered)):
        for j in range(L):
            adaptive_filter_coeffs[j] = adaptive_filter_coeffs[j] - mu * mic_input[i] * x_filtered[i - j]
    return adaptive_filter_coeffs, x_filtered, filter_state

# Audio callback
def callback(indata, outdata, frames, time_info, status):
    global adaptive_filter_coeffs, adaptive_filter_state, model_filter_state, audio_data, audio_idx

    # Grab input
    mic_in = indata[:, 0]
    audio_data[audio_idx:audio_idx+frames] = mic_in

    # Filter input with adaptive filter
    output_signal, adaptive_filter_state = sig.lfilter(adaptive_filter_coeffs, 1, mic_in, zi=adaptive_filter_state)

    # Run fxlms to update adaptive filter
    adaptive_filter_coeffs, _, model_filter_state = fxLMS(audio_data[audio_idx-builtin_delay_samples:audio_idx-builtin_delay_samples+frames], adaptive_filter_coeffs, model_filter_state)

    outdata[:, 0] = output_signal
    outdata[:, 1:] = 0  # Mute other channels

    audio_idx += frames

# Device selector
def select_device():
    print("Available audio devices:")
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        print(f"{i}: {dev['name']} {dev['max_input_channels']}x{dev['max_output_channels']} {dev['default_samplerate']}Hz {dev['hostapi']}")
    in_index = int(input("Select input device index: "))
    out_index = int(input("Select output device index: "))
    return in_index, out_index
audio_dev_index_in, audio_dev_index_out = select_device()

# Initialize state
adaptive_filter_coeffs = np.zeros(L)
adaptive_filter_state = np.zeros(L - 1)
model_filter_state = np.zeros(L - 1)

# Stream audio
with sd.Stream(callback=callback, samplerate=fs, blocksize=frame_size, channels=(1, 1), dtype='float32', device=(audio_dev_index_in, audio_dev_index_out), latency="low") as stream:
    def perform_noise_cancellation():
        stream.start()
        time.sleep(duration)
        stream.stop()
    perform_noise_cancellation()
