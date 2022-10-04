from numpy.fft import fft, ifft
import numpy as np
from matplotlib import pyplot as plt


def find_closest(freq_list, freq_element):
    prev = freq_list[0]
    for i, f in enumerate(freq_list[1:]):
        if f > freq_element: break
        if f < prev: break
        prev = f
    return i + 1


def max_slider(signal, local_max, local_range_min, local_range_max, period) :
    half_period = int(period / 4)
    if local_max == local_range_min and local_range_min > 0 :
        i = local_range_min-1
        while signal[i] > signal[local_max] and half_period > 0 :
            local_max = i
            if i == 0 :
                break
            i -= 1
            half_period -= 1
    elif local_max == local_range_max-1 and local_range_max < signal.shape[0]-1 :
        i = local_range_max+1
        while signal[i] > signal[local_max] and half_period > 0 :
            local_max = i
            if i == signal.shape[0] - 1 :
                break
            i += 1
            half_period -= 1
    return local_max


def true_max_measure(signal, local_max_list) :
    # on retire la premier car on sait qu'il est bon
    global_max = max(local_max_list)

    local_max_list.sort()
    true_maxes = []
    for prev, curr, next in zip(local_max_list[:-2], local_max_list[1:-1], local_max_list[2:]) :
        if curr == global_max : continue
        lower_boundary = int((prev + curr) / 2)
        upper_boundary = int((next + curr) / 2)
        local_range = signal[lower_boundary:upper_boundary]
        true_max = lower_boundary + np.argmax(local_range)
        true_maxes.append(1 if abs(true_max - curr) <= 1 else 0)

    # first max management :
    if local_max_list[0] != global_max :
        lower_boundary = 0
        upper_boundary = int((local_max_list[0] + local_max_list[1]) / 2)
        local_range = signal[lower_boundary:upper_boundary]
        true_max = lower_boundary + np.argmax(local_range)
        true_maxes.append(1 if abs(true_max - local_max_list[0]) <= 1 else 0)

    # last max management :
    if local_max_list[-1] != global_max:
        lower_boundary = int((local_max_list[-2] + local_max_list[-1]) / 2)
        upper_boundary = signal.shape[0]
        local_range = signal[lower_boundary:upper_boundary]
        true_max = lower_boundary + np.argmax(local_range)
        true_maxes.append(1 if abs(true_max - local_max_list[-1]) <= 1 else 0)

    if len(true_maxes) == 0 : return 0
    return sum(true_maxes) / len(true_maxes)


def extract_N_frequency(signal, N = 5, debug=False) : #############################################################"
    min = 1.9 / signal.shape[0]
    max = 0.5  # shannon

    specter = fft(signal)#.real
    specter = np.abs(specter)
    freq = np.fft.fftfreq(specter.shape[-1])

    fmin = find_closest(freq, min)
    specter[:fmin] = 0
    specter[-fmin:] = 0
    fmax = find_closest(freq, max)
    specter = specter[:fmax]
    freq = freq[:fmax]

    specter_copy = np.copy(specter)

    freqs = []
    # freq_global_max = np.max(specter_copy)
    while len(freqs) < N :
    # while True :
        # freq_max = np.max(specter_copy)
        # if np.log(freq_global_max)*0.8 > np.log(freq_max) : break

        max_index = np.argmax(specter_copy)
        specter_copy[max_index] = 0
        f = freq[max_index]

        double_detection = True
        # for f2 in freqs :
        #     if abs((f*2 - f2)/f2) < 0.1 :
        #         freqs.remove(f2)
        #         freqs.append(f)
        #         double_detection = False
        #         break
        #     elif abs((f/2 - f2)/f2) < 0.1 :
        #         double_detection = False
        #         break
        if double_detection :
            freqs.append(f)

    return freqs


def multi_max_jumper(signal, debug=False, period_range=0.1, N=5, use_edge_refiner=True) :
    min_range = 1 - period_range
    max_range = 1 + period_range

    signal = np.squeeze(signal)
    freqs = extract_N_frequency(signal, N, debug=debug)
    ratios = []
    local_max_list_list = []

    for freq in freqs :
        period = int(1 / freq)

        local_max_list = []
        signal_max_index = np.argmax(signal)
        local_max_list.append(signal_max_index)

        local_max = signal_max_index
        while local_max - period > 0 : # to index 0
            local_range_min = max(0, local_max - int(period * max_range) - 2)
            local_range_max = local_max - int(period * min_range)
            local_range = signal[local_range_min : local_range_max]
            new_local_max = local_range_min + np.argmax(local_range)
            period = local_max - new_local_max
            # new_local_max = max_slider(signal, new_local_max, local_range_min, local_range_max, period)
            local_max = new_local_max
            local_max_list.append(local_max)

        local_max = signal_max_index
        period = int(1 / freq)
        while local_max + period < signal.shape[0] : # to index max
            local_range_min = local_max + int(period * min_range) + 1
            local_range_max = min(signal.shape[0], local_max + int(period * max_range) + 2)
            local_range = signal[local_range_min : local_range_max]
            new_local_max = local_range_min + np.argmax(local_range)
            period = new_local_max - local_max
            # new_local_max = max_slider(signal, new_local_max, local_range_min, local_range_max, period)
            local_max = new_local_max
            local_max_list.append(local_max)

        ratio = true_max_measure(signal, local_max_list)
        # ratio = half_period_max_ratio(signal, local_max_list)
        # print(freq, ratio)
        ratios.append(ratio)
        local_max_list_list.append(local_max_list)

    best_ratio = ratios.index(max(ratios))
    local_max_list = local_max_list_list[best_ratio]

    if use_edge_refiner :
        local_max_list.sort()
        periods = [a2-a1 for a1, a2 in zip(local_max_list[:-1], local_max_list[1:])]
        period_mean = sum(periods) / len(periods)
        edge_action_time = local_max_list[0] + (signal.shape[0]-local_max_list[-1])
        if edge_action_time < period_mean*0.5 : local_max_list = local_max_list[:-1]

    if debug :
        time = range(signal.shape[0])
        max_amplitude = np.max(signal)
        min_amplitude = np.min(signal)
        repetitions_max = [0 if t not in local_max_list else max_amplitude for t in time]
        repetitions_min = [0 if t not in local_max_list else min_amplitude for t in time]
        plt.plot(time, repetitions_min, 'r', time, repetitions_max, 'r', time, signal)
        plt.title(str(ratio))
        plt.savefig("beats.png")
        plt.show()

    return len(local_max_list), max(ratios)


def count_repetitions(signal_1d, period_range=0.10, N=4) :
    count, ratio = multi_max_jumper(signal_1d, debug=True, period_range=period_range, N=N, use_edge_refiner=True)
    return count
