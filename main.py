import torch

from batch_worker import BATCH_SIZE, provide_batch_transitions
from learner_worker import Learner
from model import build_networks
from player import SelfPlayWorker
from replay_buffer import get_num_episodes

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)
MAX_STEPS = 10000
GAMES_PER_ITER = 4


def main():
    nets = build_networks(device=DEVICE)
    learner = Learner(
        nets.representation, nets.dynamics, nets.policy, nets.value, lr=3e-4
    )
    learner.update_target_networks()
    self_play_worker = SelfPlayWorker(nets)

    step = 0
    while step < MAX_STEPS:
        # self play worker: collect games with live networks
        # later, make this its own thread and get rid of GAMES_PER_ITER in favor of constant running
        self_play_worker.maybe_update(nets, step)
        with torch.no_grad():
            for _ in range(GAMES_PER_ITER):
                self_play_worker.play_game()

        # batch worker
        if get_num_episodes() >= BATCH_SIZE:
            batch = provide_batch_transitions(step, nets, learner.target_nets)
            if batch is not None:
                # learner worker
                nets.train()
                learner.update_step(batch, step)
                step += 1

        if step % 100 == 0:
            print(f"Step {step}, episodes in buffer: {get_num_episodes()}")

    print("Training finished.")


if __name__ == "__main__":
    main()
