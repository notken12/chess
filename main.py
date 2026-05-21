import argparse
import os

import torch

from batch_worker import BATCH_SIZE, provide_batch_transitions
from checkpoint import latest_checkpoint, load_checkpoint, save_checkpoint
from learner_worker import Learner
from model import build_networks
from player import SelfPlayWorker
from replay_buffer import get_num_episodes, load_replay_buffer, save_replay_buffer
from training_logger import TrainingLogger

torch.mps.synchronize()

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)
DEVICE = torch.device("cpu")

# How often to write a checkpoint (in train steps). Tune to balance disk + safety.
CHECKPOINT_EVERY = 500
# How often (in train steps) to print/log a row.
LOG_EVERY = 1
CHECKPOINT_DIR = "checkpoints"
LOG_DIR = "logs"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from latest checkpoint if present.",
    )
    parser.add_argument("--max-steps", type=int, default=10_000)
    parser.add_argument("--games-per-iter", type=int, default=1)
    args = parser.parse_args()

    nets = build_networks(device=DEVICE)
    learner = Learner(nets.representation, nets.dynamics, nets.policy, nets.value)
    self_play_worker = SelfPlayWorker(nets)
    logger = TrainingLogger(log_dir=LOG_DIR)

    step = 0
    if args.resume:
        ckpt = latest_checkpoint(CHECKPOINT_DIR)
        if ckpt is not None:
            step = load_checkpoint(ckpt, learner)
            print(f"Resumed from {ckpt} at step {step}", flush=True)
        else:
            print("No checkpoint found; starting fresh.", flush=True)

        if load_replay_buffer():
            print(f"Loaded replay buffer: {get_num_episodes()} episodes", flush=True)
        else:
            print("No replay buffer found; starting fresh.", flush=True)

    print(f"Device: {DEVICE}", flush=True)
    print(
        f"Need {BATCH_SIZE} episodes in buffer before training starts. "
        f"Currently: {get_num_episodes()}",
        flush=True,
    )

    try:
        while step < args.max_steps:
            # Self-play with the (periodically refreshed) snapshot.
            self_play_worker.maybe_update(nets, step)
            with torch.no_grad():
                for _ in range(args.games_per_iter):
                    history, winner, elapsed = self_play_worker.play_game()
                    logger.record_game(len(history), winner)
                    n = get_num_episodes()
                    result = (
                        "draw" if winner is None else ("white" if winner else "black")
                    )
                    print(
                        f"  game done: plies={len(history):>3d} result={result:<5s} "
                        f"time={elapsed:.1f}s buffer={n}/{BATCH_SIZE}",
                        flush=True,
                    )

            # Train step once buffer is full enough.
            if get_num_episodes() >= BATCH_SIZE:
                batch = provide_batch_transitions(step, nets, learner.target_nets)
                if batch is not None:
                    nets.train()
                    losses = learner.update_step(batch, step)
                    step += 1

                    if step % LOG_EVERY == 0:
                        logger.log_step(step, losses, get_num_episodes())

                    if step % CHECKPOINT_EVERY == 0:
                        ckpt_path = os.path.join(CHECKPOINT_DIR, f"ckpt_{step}.pt")
                        save_checkpoint(ckpt_path, learner, step)
                        print(f"Saved checkpoint: {ckpt_path}", flush=True)

    except KeyboardInterrupt:
        print("\nInterrupted. Saving final checkpoint...", flush=True)
    finally:
        final_path = os.path.join(CHECKPOINT_DIR, f"ckpt_{step}.pt")
        save_checkpoint(final_path, learner, step)
        print(f"Final checkpoint: {final_path}", flush=True)
        save_replay_buffer()
        print("Saved replay buffer.", flush=True)
        logger.close()


if __name__ == "__main__":
    main()
