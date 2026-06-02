import json


def save_normalization_stats_apwmamba(amp_stats, phase_stats, path):
    """Save per-channel-per-subcarrier stats for APWMamba (db4 DWT).

    amp_stats:   {'mean': list[27][F2], 'std': list[27][F2]}
    phase_stats: {'mean': list[18][F2], 'std': list[18][F2]}
    """
    all_stats = {
        'amplitude_per_channel': amp_stats,
        'phase_per_channel':     phase_stats,
    }
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(all_stats, f, indent=2)


def save_normalization_stats_tfmamba(amp_xh_stats, amp_xv_stats,
                                      phase_xh_stats, phase_xv_stats, path):
    """Save per-channel stats for TF-Mamba XRF55 baselines (Haar DWT).

    amp_xh_stats:   {'mean': list[135], 'std': list[135]}
    amp_xv_stats:   {'mean': list[135], 'std': list[135]}
    phase_xh_stats: {'mean': list[90],  'std': list[90]}
    phase_xv_stats: {'mean': list[90],  'std': list[90]}
    """
    all_stats = {
        'amp_xh_per_channel':   amp_xh_stats,
        'amp_xv_per_channel':   amp_xv_stats,
        'phase_xh_per_channel': phase_xh_stats,
        'phase_xv_per_channel': phase_xv_stats,
    }
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(all_stats, f, indent=2)


def load_normalization_stats(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)
