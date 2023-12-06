# fxlms-test
basic python fxlms anc test using sounddevice

system_id.py - plays a frequency sweep and tries to estimate the secondary-path impulse response from your speaker to your mic
feedback_anc.py - tries to ues the frequency response to estimate an adaptive filter to perform ANC with

there's some settings in the scripts, most importantly, a fixed delay so as to avoid like 5ms of taps that are basically 0
