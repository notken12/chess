import time
from collections import deque
import chess
import torch

from model import Networks, create_mask
from observation import NUM_SNAPSHOTS, board_to_observation
from tree import get_target_policy
from player import action_to_chess_move
from model import build_networks


def profile_play(nets: Networks, max_plies: int = 30):
    device = next(nets.representation.parameters()).device
    board = chess.Board()
    board_history: deque[chess.Board] = deque(maxlen=NUM_SNAPSHOTS)

    times = {
        "copy_mirror": 0.0,
        "observation": 0.0,
        "mask_policy": 0.0,
        "representation": 0.0,
        "policy_logits": 0.0,
        "mcts": 0.0,
        "push_move": 0.0,
    }
    counts = {k: 0 for k in times}

    for ply in range(max_plies):
        if board.is_game_over():
            break

        t0 = time.perf_counter()

        real_board = board.copy()
        board_history.appendleft(real_board)
        if board.turn == chess.BLACK:
            obs_board = board.copy().mirror()
            obs_history = [b.copy().mirror() for b in board_history]
        else:
            obs_board = real_board
            obs_history = list(board_history)
        t1 = time.perf_counter()

        observation = board_to_observation(obs_board, history=list(obs_history)).to(device)
        t2 = time.perf_counter()

        mask = create_mask(obs_board, device=device).view(-1).unsqueeze(0)
        t3 = time.perf_counter()

        with torch.no_grad():
            latent = nets.representation(observation.permute(2, 0, 1).unsqueeze(0))
            t4 = time.perf_counter()

            policy_logits = nets.policy(latent, mask)
            t5 = time.perf_counter()

            target_action, _target_policy, _target_value = get_target_policy(
                latent, policy_logits.squeeze(0), nets
            )
            t6 = time.perf_counter()

        board.push(action_to_chess_move(target_action, board, board.turn == chess.BLACK))
        t7 = time.perf_counter()

        times["copy_mirror"] += t1 - t0
        times["observation"] += t2 - t1
        times["mask_policy"] += t3 - t2
        times["representation"] += t4 - t3
        times["policy_logits"] += t5 - t4
        times["mcts"] += t6 - t5
        times["push_move"] += t7 - t6
        for k in counts:
            counts[k] += 1

    total = sum(times.values())
    print(f"\nProfile over {counts['mcts']} plies (total {total:.2f}s, {total/counts['mcts']*1000:.1f} ms/ply)\n")
    print(f"{'Component':<20} {'Total (s)':<12} {'Per-ply (ms)':<14} {'%':<8}")
    print("-" * 60)
    for name in ["copy_mirror", "observation", "mask_policy", "representation", "policy_logits", "mcts", "push_move"]:
        t = times[name]
        print(f"{name:<20} {t:<12.3f} {t/counts[name]*1000:<14.1f} {t/total*100:<7.1f}%")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sims", type=int, default=8)
    parser.add_argument("--plies", type=int, default=30)
    parser.add_argument("--sampled", type=int, default=8)
    args = parser.parse_args()

    import hyperparams
    hyperparams.NUM_SIMS_IN_SEARCH = args.sims
    hyperparams.NUM_SAMPLED_ACTIONS = args.sampled

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    nets = build_networks(device=device)
    nets.eval()
    profile_play(nets, max_plies=args.plies)
