from brian2 import *
import matplotlib.pyplot as plt
import numpy as np

# Simulation parameters
duration = 100 * ms
defaultclock.dt = 0.1 * ms

# Neuron parameters (optimized for 3nA input)
V_rest = -70 * mV
V_reset = -75 * mV  # Balanced reset potential
V_threshold = -50 * mV
Rm = 100 * Mohm
tau_m_prime = 4 * ms
k = 1.2 # Reduced current efficacy
refractory = 4 * ms  # Shorter refractory for high current
lambda_myelin = 1.3 * Hz  # Moderate myelination effect

# Neuron model with noise
eqs_myelinated = '''
dV/dt = (-(V - V_rest) + k*Rm*I_ext) / tau_m_prime + lambda_myelin*(V_prev - V) + sigma*xi*tau_m_prime**-0.5 : volt (unless refractory)
I_ext : amp               
V_prev : volt
lambda_myelin : Hz        
energy_used : joule
sigma : volt (constant)   # Noise amplitude
'''

# Neuron group
neuron = NeuronGroup(
    10000,
    model=eqs_myelinated,
    method='euler',
    threshold='V > V_threshold',
    reset='V = V_reset',
    refractory=refractory
)

# Initialize with variability
neuron.V = 'V_rest + rand()*5*mV'  # Random initial conditions
neuron.V_prev = V_rest
neuron.I_ext = '0.7*nA + randn()*0.3*nA'  # Noisy current input
neuron.lambda_myelin = lambda_myelin
neuron.sigma = 0.5 * mV  # Noise level
neuron.energy_used = 0 * joule


@network_operation(dt=defaultclock.dt)
def update_prev():
    neuron.V_prev = neuron.V


@network_operation(dt=defaultclock.dt)
def update_energy():
    neuron.energy_used += (neuron.I_ext ** 2 * Rm) * defaultclock.dt * 0.1  # 10% efficiency


# Monitors
spike_monitor = SpikeMonitor(neuron)
state_monitor = StateMonitor(neuron, 'V', record=range(100))  # Track first 100 neurons
population_rate = PopulationRateMonitor(neuron)

run(duration)

# Basic metrics
total_spikes = spike_monitor.num_spikes
total_energy = sum(neuron.energy_used)
energy_per_spike = total_energy / total_spikes if total_spikes > 0 else 0 * joule

# Advanced spike analysis
if total_spikes > 0:
    # Firing rate calculation
    firing_rates = np.bincount(spike_monitor.i, minlength=10000) / (duration / second)
    avg_firing_rate = np.mean(firing_rates[firing_rates > 0])

    # Spike timing precision
    first_spike_times = []
    for neuron_idx in range(10000):
        spike_times = spike_monitor.t[spike_monitor.i == neuron_idx]
        if len(spike_times) > 0:
            first_spike_times.append(spike_times[0] / ms)
    spike_timing_precision = np.std(first_spike_times) if len(first_spike_times) > 1 else 0

    # Latency to first spike
    avg_first_spike_latency = np.mean(first_spike_times) if first_spike_times else 0

    # Inter-spike interval analysis
    isi_values = []
    for neuron_idx in range(10000):
        spike_times = spike_monitor.t[spike_monitor.i == neuron_idx] / second
        if len(spike_times) > 1:
            isi_values.extend(np.diff(spike_times) * 1000)  # Convert to ms
    avg_isi = np.mean(isi_values) if isi_values else 0
    refractory_violations = np.sum(np.array(isi_values) < (refractory / ms)) if isi_values else 0

    # Threshold crossing analysis (fixed)
    threshold_crossings = []
    for neuron_idx in range(100):  # Check first 100 neurons
        v = state_monitor.V[neuron_idx] / mV
        crossings = np.sum((v[:-1] < V_threshold / mV) & (v[1:] >= V_threshold / mV))
        threshold_crossings.append(crossings)
    avg_threshold_crossings = np.mean(threshold_crossings)
else:
    avg_firing_rate = spike_timing_precision = avg_first_spike_latency = 0
    avg_isi = refractory_violations = avg_threshold_crossings = 0

# Print results
print("\n" + "=" * 50)
print("BASIC METRICS:")
print(f"Total spikes: {total_spikes:,}")
print(f"Total energy used: {total_energy:.3e} J")
print(f"Energy per spike: {energy_per_spike:.3e} J/spike")

print("\n" + "=" * 50)
print("SPIKE TIMING ANALYSIS:")
print(f"Average firing rate: {avg_firing_rate:.2f} Hz")
print(f"Spike timing precision (std of first spikes): {spike_timing_precision:.2f} ms")
print(f"Average latency to first spike: {avg_first_spike_latency:.2f} ms")
print(f"Average inter-spike interval (ISI): {avg_isi:.2f} ms")
print(f"Refractory period violations: {refractory_violations}")

print("\n" + "=" * 50)
print("THRESHOLD CROSSING ANALYSIS:")
print(f"Detected threshold crossings (per neuron): {avg_threshold_crossings:.1f}")
print(f"Spike-to-crossing ratio: {total_spikes / max(1, 100 * avg_threshold_crossings):.2f}:1")

# Plotting
plt.figure(figsize=(14, 10))
plt.subplot(2, 2, 1)
plt.plot(spike_monitor.t / ms, spike_monitor.i, '.k', markersize=1, alpha=0.3)
plt.xlabel('Time (ms)')
plt.ylabel('Neuron index')
plt.title(f'Spike Raster Plot\n(Total Spikes = {total_spikes:,})')

plt.subplot(2, 2, 2)
for neuron_idx in [0, 1, 2]:  # Plot first 3 neurons
    plt.plot(state_monitor.t / ms, state_monitor.V[neuron_idx] / mV, label=f'Neuron {neuron_idx}')
plt.axhline(V_threshold / mV, linestyle='--', color='r', label='Threshold')
plt.xlabel('Time (ms)')
plt.ylabel('Membrane Potential (mV)')
plt.legend()
plt.title('Sample Voltage Traces')

plt.subplot(2, 2, 3)
if isi_values:
    plt.hist(isi_values, bins=50, range=(0, 100))
    plt.axvline(refractory / ms, color='r', linestyle='--', label='Refractory period')
    plt.xlabel('Inter-spike interval (ms)')
    plt.ylabel('Count')
    plt.legend()
    plt.title('ISI Distribution')

plt.subplot(2, 2, 4)
plt.plot(population_rate.t / ms, population_rate.smooth_rate(width=5 * ms) / Hz)
plt.xlabel('Time (ms)')
plt.ylabel('Population rate (Hz)')
plt.title('Population Firing Rate')

plt.tight_layout()
plt.show()



"""
==================================================
BASIC METRICS:
Total spikes: 179,904
Total energy used: 5.823e-09 J
Energy per spike: 3.236e-14 J/spike

==================================================
SPIKE TIMING ANALYSIS:
Average firing rate: 186.72 Hz
Spike timing precision (std of first spikes): 2.04 ms
Average latency to first spike: 1.30 ms
Average inter-spike interval (ISI): 5.42 ms
Refractory period violations: 0

==================================================
THRESHOLD CROSSING ANALYSIS:
Detected threshold crossings (per neuron): 0.0
Spike-to-crossing ratio: 179904.00:1

"""
