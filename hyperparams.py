REPLAY_CAPACITY = int(1e6)
BATCH_SIZE = 64
# Chess terminal rewards arrive only at the end of the game, so no discounting
# within the K-step window is necessary.
GAMMA = 1.0
UNROLL_STEPS = 5
TD_STEPS = 5

NUM_SIMS_IN_SEARCH = 16
NUM_SAMPLED_ACTIONS = 8

SELF_PLAY_NET_UPDATE_INTERVAL = 100
TARGET_NET_UPDATE_INTERVAL = 400

REWARD_LOSS_COEFF = 1.0
POLICY_LOSS_COEFF = 1.0
VALUE_LOSS_COEFF = 0.25
CONSISTENCY_LOSS_COEFF = 2.0
POLICY_ENTROPY_LOSS_COEFF = 5e-3

# Optimizer settings (SGD for discrete / vision tasks per Table 3)
LR = 0.2
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4

# EfficientZero V2 mixed value target thresholds.
# Before T1 training steps, the value network is too inaccurate to rely on
# MCTS values, so the TD target is always used regardless of game recency.
T1 = 4e4
# The T2 most recently added games always use the TD target: their MCTS was
# run with an older network snapshot and is therefore less trustworthy.
T2 = 2e4
