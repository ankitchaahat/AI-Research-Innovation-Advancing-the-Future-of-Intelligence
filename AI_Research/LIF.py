from brian2 import *
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(3)
# Simulation parameters
duration = 100 * ms
defaultclock.dt = 0.1 * ms

# Standard LIF parameters (no myelination)
V_rest = -70 * mV
V_reset = -75 * mV
V_threshold = -50 * mV
Rm = 100 * Mohm
tau_m = 4 * ms
refractory = 4 * ms  # Standard refractory

# LIF equation (no myelination term)
eqs_lif = '''
dV/dt = (-(V - V_rest) + Rm*I_ext) / tau_m : volt (unless refractory)
I_ext : amp
energy_used : joule
'''

# Neuron group
neurons = NeuronGroup(
    10000,
    model=eqs_lif,
    threshold='V > V_threshold',
    reset='V = V_reset',
    refractory=refractory,
    method='euler'
)

# Initialize with variability
neurons.V = 'V_rest + rand()*5*mV'
neurons.I_ext = '0.7*nA + randn()*0.015*nA'  # Lower current for comparable firing rates
neurons.energy_used = 0 * joule


# Energy calculation
@network_operation(dt=defaultclock.dt)
def update_energy():
    neurons.energy_used += (neurons.I_ext ** 2 * Rm) * defaultclock.dt


# Monitors
spike_mon = SpikeMonitor(neurons)
state_mon = StateMonitor(neurons, 'V', record=range(100))
pop_rate = PopulationRateMonitor(neurons)

run(duration)

# Analysis (identical to your myelinated model)
total_spikes = spike_mon.num_spikes
total_energy = sum(neurons.energy_used)
energy_per_spike = total_energy / total_spikes if total_spikes > 0 else 0 * joule

# Spike timing analysis
if total_spikes > 0:
    firing_rates = np.bincount(spike_mon.i, minlength=10000) / (duration / second)
    avg_rate = np.mean(firing_rates[firing_rates > 0])

    first_spikes = [spike_mon.t[spike_mon.i == i][0] / ms for i in range(10000) if
                    len(spike_mon.t[spike_mon.i == i]) > 0]
    timing_precision = np.std(first_spikes) if len(first_spikes) > 1 else 0
    avg_latency = np.mean(first_spikes) if first_spikes else 0

    isi_values = []
    for i in range(10000):
        spikes = spike_mon.t[spike_mon.i == i] / second
        if len(spikes) > 1:
            isi_values.extend(np.diff(spikes) * 1000)
    avg_isi = np.mean(isi_values) if isi_values else 0
    ref_violations = sum(isi < (refractory / ms) for isi in isi_values) if isi_values else 0

    # Fixed threshold crossing detection
    crossings = []
    for i in range(100):
        v = state_mon.V[i] / mV
        crossings.append(np.sum((v[:-1] < V_threshold / mV) & (v[1:] >= V_threshold / mV)))
    avg_crossings = np.mean(crossings)
else:
    avg_rate = timing_precision = avg_latency = avg_isi = ref_violations = avg_crossings = 0

# Print results
print("\n" + "=" * 50)
print("STANDARD LIF NEURONS (10,000)")
print("=" * 50)
print(f"Total spikes: {total_spikes:,}")
print(f"Total energy: {total_energy:.3e} J")
print(f"Energy/spike: {energy_per_spike:.3e} J")
print(f"\nFiring rate: {avg_rate:.2f} Hz")
print(f"Spike timing precision: {timing_precision:.2f} ms")
print(f"Latency to first spike: {avg_latency:.2f} ms")
print(f"Avg ISI: {avg_isi:.2f} ms")
print(f"Refractory violations: {ref_violations}")
print(f"\nThreshold crossings: {avg_crossings:.1f}")
print(f"Spike-to-crossing ratio: {total_spikes / max(1, 100 * avg_crossings):.2f}:1")

# Plotting
plt.figure(figsize=(14, 10))
plt.subplot(2, 2, 1)
plt.plot(spike_mon.t / ms, spike_mon.i, '.k', markersize=1, alpha=0.3)
plt.title(f"Spike Raster (n={total_spikes:,})")

plt.subplot(2, 2, 2)
for i in [0, 1, 2]:
    plt.plot(state_mon.t / ms, state_mon.V[i] / mV, label=f'Neuron {i}')
plt.axhline(V_threshold / mV, ls='--', c='r', label='Threshold')
plt.legend()

plt.subplot(2, 2, 3)
if isi_values:
    plt.hist(isi_values, bins=50, range=(0, 100))
    plt.axvline(refractory / ms, c='r', ls='--')

plt.subplot(2, 2, 4)
plt.plot(pop_rate.t / ms, pop_rate.smooth_rate(width=5 * ms) / Hz)
plt.tight_layout()
plt.show()

"""

==================================================
STANDARD LIF NEURONS (10,000)
==================================================
Total spikes: 180,045
Total energy: 4.901e-08 J
Energy/spike: 2.722e-13 J

Firing rate: 180.04 Hz
Spike timing precision: 0.09 ms
Latency to first spike: 1.14 ms
Avg ISI: 5.55 ms
Refractory violations: 0

Threshold crossings: 0.0
Spike-to-crossing ratio: 180045.00:1
"""
