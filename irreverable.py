import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import soundfile as sf
import os as os
from scipy import signal
from scipy.interpolate import interp1d

# Configuration
number_of_irs = 48
min_duration = 0.12
target_duration = 1.9 
max_duration = 10.3 
min_predelay = 0.01 
target_predelay = 0.1
max_predelay = 0.3
tau0min = 0.020 
tau0max = 0.070
tau0_target = 0.050
sample_rate = 48000
output_dir = "/Users/luna/Documents/Doc/Université/University-Research/Codes/ir_gen/IRs_2"

##################################################################################################################
## Function starts here : 

# Generate impulse response parameters
durations = np.sort(np.random.triangular(min_duration*2, target_duration*2, max_duration*2, number_of_irs))
predelays = np.random.triangular(min_predelay, target_predelay, max_predelay, number_of_irs)
etas = np.random.rand(number_of_irs)*0.04 + 0.07
rho0s = np.random.rand(number_of_irs) * 1e2 + 1e5
tau0s = np.sort(np.random.triangular(tau0min, tau0_target, tau0max, number_of_irs))
variability_low = np.random.triangular(0.06, 0.07, 0.08, number_of_irs)
variability_high = np.random.triangular(0.04, 0.044, 0.055, number_of_irs)
noise_rate_low = np.random.triangular(2, 3, 5, number_of_irs)
noise_rate_high = np.random.triangular(30, 50, 60, number_of_irs)

#print(durations)
#print(predelays)
#print(etas)
#print(rho0s)
#print(tau0s)
#print(variability_low)
#print(variability_high)
#print(noise_rate_low)
#print(noise_rate_high)

freq_bands = [20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 
              630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 
              10000, 12500, 16000, 20000]

os.makedirs(output_dir, exist_ok=True)

for i in range(number_of_irs):
    
    # Time vector
    t = np.linspace(0, durations[i], int(sample_rate * durations[i]))
    ns = len(t)

    # GENERATING ECHO DENSITY CURVE
    ###############################
    growth = np.log(0.05 / 0.95) / (tau0s[i] - etas[i])  # Growth rate set to reach 95% of max echo density at time t=eta
    base_curve = rho0s[i] / (1 + np.exp(growth * (tau0s[i] - t)))  # Logistic growth function
    # Generate low-frequency noise
    num_noise_low = max(4, int(noise_rate_low[i] * (t[-1] - t[0])))  # Ensure at least 4 points
    t_noise_low = np.linspace(t[0], t[-1], num_noise_low)  # Use num_noise_low instead of recomputing
    # Generate noise values
    noise_low = (np.random.rand(len(t_noise_low)) - 0.5) * 2 * variability_low[i] * rho0s[i]
    # Ensure unique time points (avoid duplicate interpolation points)
    t_noise_low, unique_indices = np.unique(t_noise_low, return_index=True)
    noise_low = noise_low[unique_indices]
    # Ensure we have at least 4 points for cubic interpolation, otherwise use linear
    if len(t_noise_low) < 4:
        interp_type = 'linear'  # Fallback to linear interpolation
    else:
        interp_type = 'cubic'
    # If no valid points exist after deduplication, set noise to zero
    if len(t_noise_low) == 0:
        noise_low_interp = np.zeros_like(t)  # Prevent crashes
    else:
        noise_low_interp = interp1d(t_noise_low, noise_low, kind=interp_type, fill_value="extrapolate")(t)  # Interpolated slow noise

    # Generate high-frequency noise
    t_noise_high = np.linspace(t[0], t[-1], int(noise_rate_high[i] * (t[-1] - t[0])))  # Coarse time vector for fast noise
    noise_high = (np.random.rand(len(t_noise_high)) - 0.5) * 2 * variability_high[i] * rho0s[i]
    noise_high_interp = interp1d(t_noise_high, noise_high, kind='cubic', fill_value="extrapolate")(t)  # Interpolated fast noise
    ###############################
    echo_density_curve = base_curve + noise_low_interp + noise_high_interp # combine growth function and noise

    # GENERATE ECHOES FROM ED CURVE
    ###############################
    index = np.round(np.interp(np.linspace(0, 1, ns), np.cumsum(echo_density_curve) / np.sum(echo_density_curve), np.arange(ns))).astype(int)
    min_density = 1e-3  # Avoid near-zero divisions
    safe_echo_density = np.clip(echo_density_curve[index], min_density, None)
    deltas = np.cumsum(np.abs(np.random.normal(scale=0.1, size=len(index)) + 1) * (3 * sample_rate / safe_echo_density))
    alphas = np.random.randn(len(index)) * np.sqrt(np.minimum(echo_density_curve[index], 1) / echo_density_curve[index])
    echoes = np.zeros(ns)
    m = 8
    for k in range(len(index)):
        ti = int(np.floor(deltas[k]))
        tf = deltas[k] - ti
        sinc_range = np.arange(ti - m, ti + m + 1)
        valid_range = (sinc_range >= 0) & (sinc_range < ns)
        echoes[sinc_range[valid_range]] += alphas[k] * np.sinc(tf - np.arange(-m, m + 1))[valid_range]
    
    echoes_lenght = ns + 2*sample_rate
    echoes = echoes[:echoes_lenght]

    # FREQUENCY DEPENDENT DECAY 
    ###########################
    num_bands = len(freq_bands) - 1
    max_rt60 = 0.99 * durations[i] # Max RT60 cannot exceed IR duration
    min_rt60 = max(0.1, max_rt60 * 0.1)  # Ensure a minimum RT60
    random_values = np.sort(np.random.rand(num_bands))[::-1]# Generate decreasing random values
    rt60_values = min_rt60 + (max_rt60 - min_rt60) * random_values # Scale to [min_rt60, max_rt60]
    decay_factors = np.exp(- 6.91 * t[:, None] / rt60_values[None, :]) # decay of 6.91 dB per RT60
    filtered_signal = np.zeros_like(echoes)

    for j in range(len(freq_bands) - 1):
        f_low = freq_bands[j]
        f_high = freq_bands[j + 1]

        # Create a bandpass filter for this frequency range
        b, a = signal.butter(2, [f_low / (sample_rate / 2), f_high / (sample_rate / 2)], btype='band')
        filtered_band = signal.filtfilt(b, a, echoes)

        # Apply frequency-dependent exponential decay using t
        filtered_band *= decay_factors[:, j]

        # Sum across all frequency bands
        filtered_signal += filtered_band
    

    # ADD PRE-DELAY
    ###############################
    predelay_samples = int(predelays[i]*sample_rate)
    if np.max(np.abs(filtered_signal)) != 0:
        signal_with_predelay = np.concatenate((np.zeros(predelay_samples), filtered_signal / np.max(np.abs(filtered_signal)) * 32768 / 32767))
    else:
        signal_with_predelay = np.concatenate((np.zeros(predelay_samples), filtered_signal))

    #SAVE IR
    ###############################
    
    output_file = os.path.join(output_dir, f'IR_length{round(np.mean(durations[i]/2),2)}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.wav')
    sf.write(output_file, signal_with_predelay, sample_rate, format='wav', subtype='PCM_32')
    print(f'IR {i+1}/{number_of_irs}: {output_file} DONE!')