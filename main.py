import copy
import time

import torch

from model import build_networks
from learner_worker import Learner
from batch_worker import provide_batch_transitions, set_target_networks
from player import play_game
from replay_buffer import get_num_episodes

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GAMES_PER_ITER = 4
TARGET_UPDATE_INTERVAL = 400
MAX_STEPS = 100_000


def main():
    nets = build_networks(device=DEVICE)
    learner = Learner(
        nets.representation,
        nets.dynamics,
        nets.policy,
        nets.value,
        lr=1e-4,
    )
    set_target_networks(copy.deepcopy(nets))

    step = 0
    while step < MAX_STEPS:
        # Self-play: collect games with the current (live) network
        nets.eval()
        with torch.no_grad():
            for _ in range(GAMES_PER_ITER):
                play_game(nets)

        # Training
        nets.train()
        from batch_worker import BATCH_SIZE
        if get_num_episodes() >= BATCH_SIZE:
            batch = provide_batch_transitions(step, nets)
            if batch is not None:
                learner.update_step(batch, step)
                step += 1

        # Periodic target network update for stable reanalysis
        if step % TARGET_UPDATE_INTERVAL == 0 and step > 0:
            set_target_networks(copy.deepcopy(nets))
            print(f"Step {step}: target network updated")

        if step % 100 == 0:
            print(f"Step {step}, episodes in buffer: {get_num_episodes()}")

    print("Training finished.")


if __name__ == "__main__":
    main()
