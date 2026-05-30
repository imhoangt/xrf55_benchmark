try:
    import csiread
    _CSIREAD_OK = True
except ImportError:
    _CSIREAD_OK = False

import numpy as np
import scipy.io
from pathlib import Path

ACTION_ID_TO_LABEL = {
    31: 0,  32: 1,  33: 2,  34: 3,  35: 4,
    36: 5,  37: 6,  38: 7,  39: 8,  40: 9,  41: 10,
}
ACTION_IDS_USED = sorted(ACTION_ID_TO_LABEL.keys())   # [31..41]
ACTION_NAMES = [
    'Waving', 'Clap Hands', 'Fall on the Floor', 'Jumping', 'Running',
    'Sitting Down', 'Standing Up', 'Turning', 'Walking',
    'Stretch Oneself', 'Pat on Shoulder',
]                                                       # label-ordered (0..10)


def _apply_perm_correction(csi, perm, fpath):
    """Apply antenna permutation correction. csi/perm as from Intel tool."""
    p_min, p_max = int(perm.min()), int(perm.max())
    if p_min == 1 and p_max == 3:
        perm = perm - 1
    elif p_min == 0 and p_max == 2:
        pass
    else:
        raise ValueError(f"Unexpected perm range [{p_min}, {p_max}] in {fpath}")

    # CRITICAL: perm[:, None, :] shape (1000, 1, 3) != csi shape (1000, 30, 3).
    # np.take_along_axis behavior with mismatched non-axis dims is version-dependent
    # (may return shape (1000, 1, 3) → silent data corruption of all 30 subcarriers).
    # Use broadcast_to to make indexing shape EXPLICIT before calling take_along_axis.
    idx      = np.broadcast_to(perm[:, None, :], csi.shape)
    csi_phys = np.take_along_axis(csi, idx, axis=-1)
    assert csi_phys.shape == (1000, 30, 3), \
        f"perm correction returned wrong shape {csi_phys.shape}"
    return csi_phys


def parse_xrf55_dat(fpath):
    """Parse one .dat file with antenna permutation correction.
    Returns: csi_phys (1000, 30, 3) complex128
    """
    if not _CSIREAD_OK:
        raise ImportError('csiread not installed. Run: pip install csiread')
    cd = csiread.Intel(str(fpath), nrxnum=3, ntxnum=1, pl_size=0, if_report=False)
    cd.read()

    csi  = cd.csi[:, :, :, 0]
    perm = cd.perm

    assert csi.shape == (1000, 30, 3)
    assert np.isfinite(np.abs(csi)).all()

    return _apply_perm_correction(csi, perm, fpath).astype(np.complex128)


def parse_xrf55_mat(fpath):
    """Parse one .mat file (volunteers 04 & 12) with antenna permutation correction.

    .mat format: scipy.io.loadmat → {'data': (1000,1) object array of MATLAB structs}
    Each struct has 'csi' (1,3,30) complex128 and 'perm' (1,3) per packet.
    Perm is constant across packets; uses same correction as parse_xrf55_dat.

    Returns: csi_phys (1000, 30, 3) complex128
    """
    mat  = scipy.io.loadmat(str(fpath))
    data = mat['data']   # (1000, 1) object array
    assert data.shape[0] == 1000, f"Expected 1000 packets, got {data.shape[0]} in {fpath}"

    csi_list = []
    for t in range(1000):
        pkt = data[t, 0][0, 0]
        csi_list.append(pkt['csi'][0].T)   # (1,3,30)[0] → (3,30) → .T → (30,3)

    csi  = np.stack(csi_list, axis=0).astype(np.complex128)   # (1000, 30, 3)
    perm = data[0, 0][0, 0]['perm'][0].reshape(1, 3).repeat(1000, axis=0)  # (1000, 3)

    assert csi.shape == (1000, 30, 3)
    assert np.isfinite(np.abs(csi)).all()

    return _apply_perm_correction(csi, perm, fpath)


def load_xrf55_sample(raw_scene_dir, vol_id, action_id, rep_id):
    """Stack 3 RX devices → (1000, 30, M=3, A=3) complex128.

    Args:
        raw_scene_dir: Path to dataset/xrf55/raw/scene_01/
        vol_id: int 1-30
        action_id: int from ACTION_IDS_USED (31-41, NOT label!)
        rep_id: int 1-20

    NOTE: action_id is the filename ID (31-41), NOT the label (0-10).
    To get label: label = ACTION_ID_TO_LABEL[action_id]
    """
    assert action_id in ACTION_IDS_USED, f"Invalid action_id {action_id}"
    stem = f"{vol_id:02d}_{action_id:02d}_{rep_id:02d}"
    csi_list = []
    for rx in [1, 2, 3]:
        rx_dir = Path(raw_scene_dir) / f"rx_{rx:02d}" / f"{vol_id:02d}"
        dat_path = rx_dir / f"{stem}.dat"
        mat_path = rx_dir / f"{stem}.mat"
        if dat_path.exists():
            csi_list.append(parse_xrf55_dat(dat_path))
        elif mat_path.exists():
            csi_list.append(parse_xrf55_mat(mat_path))
        else:
            raise FileNotFoundError(
                f"No .dat or .mat for vol={vol_id} action={action_id} rep={rep_id} rx={rx}"
            )
    return np.stack(csi_list, axis=2)
