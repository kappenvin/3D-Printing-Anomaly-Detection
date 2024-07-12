from scipy.signal import butter, filtfilt
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq

def butter_lowpass_filter(data, cutoff, fs, order=5):
    
    # Calculate the Nyquist frequency
    nyquist = 0.5 * fs
    # Calculate the normal cutoff frequency needed for the filter , if normal_cutoff= 1 the nyquist frequency will be taken.
    normal_cutoff = cutoff / nyquist
    # Design the Butterworth filter
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    # Apply the Butterworth filter to the data
    y = filtfilt(b, a, data)
    return y


def apply_fft_normalized(data, fs):
    # Step 1: Calculate the number of samples in the data
    n = len(data)
    
    # Step 2: Compute the FFT of the data
    fft_data = np.fft.fft(data)
    
    # Step 3: Normalize the amplitude of the FFT output
    fft_amplitude = np.abs(fft_data) / n
    
    # Step 4: Compute the frequency bins corresponding to the FFT output
    freq = np.fft.fftfreq(n, 1/fs)
    
    # Step 5: Prepare the single-sided amplitude spectrum
    single_sided_amplitude = fft_amplitude[:n//2]
    single_sided_amplitude[1:] = 2 * single_sided_amplitude[1:]
    
    # Step 6: Prepare the corresponding frequency bins for the single-sided spectrum
    single_sided_freq = freq[:n//2]
    
    # Step 7: Return the single-sided amplitude and frequency arrays
    return single_sided_amplitude, single_sided_freq


def get_last_freq(fft, fft_freq,target_value):
    """
    Find the last frequency in fft_freq where target_value occurs in fft.

    Parameters:
    fft (np.ndarray): FFT magnitude array
    fft_freq (np.ndarray): Corresponding frequency array
    target_value (float): Value to search for in fft

    Returns:
    float or None: Last matching frequency or None if not found
    """
        
    # Find indices where fft matches the target value
    matching_indices = np.where(fft == target_value)[0]

    # If no matches found, return None or raise an exception
    if len(matching_indices) == 0:
        return None  # or raise ValueError("Target value not found in fft")

    # Get the last matching index
    last_index = matching_indices[-1]

    # Return the corresponding frequency
    return fft_freq[last_index]