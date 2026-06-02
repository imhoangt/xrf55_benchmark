TRAINVAL_REPS = list(range(1, 15))   # reps 01–14  →  coarse train+val partition (4620 samples)
VAL_REPS   = list(range(13, 15))  # reps 13–14  →  val split  (  660 samples)
TEST_REPS  = list(range(15, 21))  # reps 15–20  →  test split (1980 samples)

# Fine-grained train/val/test split (after removing val from coarse train):
TRAIN_SAMPLES = 3960   # 30×11×12
VAL_SAMPLES   =  660   # 30×11×2
TEST_SAMPLES  = 1980   # 30×11×6
