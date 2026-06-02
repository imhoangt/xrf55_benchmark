from pathlib import Path
import re


def scan_action_ids(scene_dir: Path, rx="rx_01"):
    """Scan filenames to discover (vol, action_id, rep) sets.
    Then verify discovered action_ids match expected ACTION_IDS_USED.
    """
    scene_dir = Path(scene_dir)
    pat = re.compile(r"(\d+)_(\d+)_(\d+)\.(dat|mat)")
    vol, act, rep = set(), set(), set()
    for f in (scene_dir / rx).rglob("*.[dm][aa][tt]"):
        m = pat.match(f.name)
        if m:
            vol.add(int(m.group(1)))
            act.add(int(m.group(2)))
            rep.add(int(m.group(3)))
    vol, act, rep = sorted(vol), sorted(act), sorted(rep)

    from src.data.preprocessing.parser import ACTION_IDS_USED
    assert vol == list(range(1, 31)), f"Expected vol 1-30, got {vol}"
    assert act == ACTION_IDS_USED,     f"Expected actions {ACTION_IDS_USED}, got {act}"
    assert rep == list(range(1, 21)),  f"Expected reps 1-20, got {rep}"

    print(f"Verified: 30 volunteers, 11 actions ({act[0]}-{act[-1]}), 20 reps")
    return vol, act, rep
