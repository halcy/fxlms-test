import sounddevice as sd
import numpy as np
import scipy.signal as sig
import time
from scipy.io import wavfile

# Setup (must match other script)
frame_size = 128
builtin_delay = 0.05

# Setup (general)
fs = 44100
duration = 10
L = 127  # taps for filter

# Frequency Sweep Parameters
start_freq = 1     # Start frequency of the sweep
end_freq = 20000    # End frequency of the sweep

# DC offset remover
normal_cutoff = 50 / (0.5 * fs)
b, a = sig.butter(6, normal_cutoff, btype='high', analog=False)

# Sweep-with-noise and state
current_index = 0
input_recording = np.zeros((duration)*fs)
sweep_recording = np.zeros((duration)*fs)
t = np.linspace(0.0, 1.0, fs, endpoint=False)
pause = list(np.zeros(fs))
sweep = np.array((list(sig.chirp(t, f0=start_freq, f1=end_freq/2.0, t1=1.0, method='linear')) + pause))
sweep[0:len(t) + int(len(pause) * 0.25)] += np.random.normal(0, 0.2, len(t) + int(len(pause) * 0.25))
sweep = np.array(list(sweep) * (duration + 1), dtype='float32') * 0.8

# Calculate delay
builtin_delay = 0.05
builtin_delay_samples = int(builtin_delay * fs) + frame_size

# Audio callback
def playingCallback(indata, outdata, frames, time_info, status):
    global input_recording, sweep_recording
    global current_index

    mic_in = indata[:, 0]
    mic_in = sig.lfilter(b, a, mic_in)  # Apply HPF
    input_recording[current_index:current_index+frames] = mic_in

    # Generate Sweep
    sweep_recording[current_index:current_index+frames] = sweep[current_index:current_index+frames]
    
    outdata[:, 0] = sweep[current_index:current_index+frames] * 10000.0
    outdata[:, 1:] = 0  # Mute other channels
    current_index += frames

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

# Setup Audio Stream
with sd.Stream(callback=playingCallback, samplerate=fs, blocksize=frame_size, channels=(1, 1), dtype='float32', device=(audio_dev_index_in, audio_dev_index_out), latency="low") as stream:
    def measure_system():
        global sweep_recording, input_recording

        # Run Measurement
        print('Girls are now sweeping, please wait silently...')
        stream.start()
        time.sleep(duration)
        stream.stop()

        # Save data as wav files
        wavfile.write('input_recording_raw.wav', fs, input_recording.astype(np.float32))
        wavfile.write('sweep_recording_raw.wav', fs, sweep_recording.astype(np.float32))

        # Calculate mean to try to reduce noise
        sweep_recording = sweep_recording.reshape(-1, len(t) + len(pause))[:, :len(sweep) + int(len(pause) * 0.5)].mean(axis=0)
        input_recording = input_recording.reshape(-1, len(t) + len(pause))[:, :len(sweep) + int(len(pause) * 0.5)].mean(axis=0)
        
        input_recording = input_recording[builtin_delay_samples:]
        sweep_recording = sweep_recording[:-builtin_delay_samples]

        # Save data as wav files
        wavfile.write('input_recording.wav', fs, input_recording.astype(np.float32))
        wavfile.write('sweep_recording.wav', fs, sweep_recording.astype(np.float32))

        # Calculate impulse response and filter
        print('Calculating transfer function and impulse response')
        f, H = sig.freqz(input_recording, sweep_recording, fs=fs, worN=2**14)
        impulse_response = np.fft.irfft(H)
        print('Calculating FIR filter')
        filter_coeffs = sig.firls(L, f, np.abs(H), fs=fs)
        print(filter_coeffs)

        wavfile.write('system_response.wav', fs, impulse_response.astype(np.float32))
        np.savez('system_analysis.npz', impulse_response=impulse_response, filter_coeffs=filter_coeffs)
        
        return impulse_response, filter_coeffs
    
    impulse_response, filter_coeffs = measure_system()
