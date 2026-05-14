from collections import deque
from datetime import datetime
from typing import List, Optional, Tuple
import copy
import os
import time

import chess
import chess.pgn

from model import Networks, create_mask
from observation import NUM_SNAPSHOTS, board_to_observation
from tree import get_target_policy
from replay_buffer import Step, save_to_replay_buffer
from hyperparams import SELF_PLAY_NET_UPDATE_INTERVAL
import torch


GameResult = Tuple[List[Step], Optional[bool], float]  # (history, winner, elapsed_s)


class SelfPlayWorker:
    """Owns a network snapshot for self-play, refreshed from the learner periodically."""

    def __init__(self, nets: Networks) -> None:
        self.nets = copy.deepcopy(nets)
        self.nets.eval()
        for module in (
            self.nets.representation,
            self.nets.dynamics,
            self.nets.policy,
            self.nets.value,
        ):
            for p in module.parameters():
                p.requires_grad = False
        self._last_update_step = -1

    def maybe_update(self, latest_nets: Networks, training_step: int) -> bool:
        """Refresh the local snapshot if the update interval has passed."""
        if training_step - self._last_update_step >= SELF_PLAY_NET_UPDATE_INTERVAL:
            self.nets = copy.deepcopy(latest_nets)
            self.nets.eval()
            for module in (
                self.nets.representation,
                self.nets.dynamics,
                self.nets.policy,
                self.nets.value,
            ):
                for p in module.parameters():
                    p.requires_grad = False
            self._last_update_step = training_step
            return True
        return False

    def play_game(self) -> GameResult:
        return _play_game(self.nets)


def action_to_chess_move(
    action_idx: int, board: chess.Board, mirrored: bool
) -> chess.Move:
    z = action_idx // 64
    x = (action_idx % 64) // 8
    y = action_idx % 8
    from_sq = chess.square(x, y)

    if z < 56:
        direction = z // 7
        distance = (z % 7) + 1
        dirs = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]
        dx, dy = dirs[direction]
        to_sq = chess.square(x + dx * distance, y + dy * distance)
        promotion = None
    elif z < 64:
        knight_offsets = [
            (1, 2),
            (2, 1),
            (2, -1),
            (1, -2),
            (-1, -2),
            (-2, -1),
            (-2, 1),
            (-1, 2),
        ]
        idx = z - 56
        dx, dy = knight_offsets[idx]
        to_sq = chess.square(x + dx, y + dy)
        promotion = None
    else:
        promo_idx = z - 64
        direction_idx = promo_idx // 3
        piece_idx = promo_idx % 3
        promo_dirs = [(-1, 1), (0, 1), (1, 1)]
        dx, dy = promo_dirs[direction_idx]
        to_sq = chess.square(x + dx, y + dy)
        promo_pieces = [chess.KNIGHT, chess.BISHOP, chess.ROOK]
        promotion = promo_pieces[piece_idx]

    if mirrored:
        from_sq = chess.square_mirror(from_sq)
        to_sq = chess.square_mirror(to_sq)

    move = chess.Move(from_sq, to_sq, promotion)
    # Auto-fix queen promotion if we decoded a pawn move to the back rank without promotion
    if move not in board.legal_moves and promotion is None:
        piece = board.piece_at(from_sq)
        if piece is not None and piece.piece_type == chess.PAWN:
            to_rank = chess.square_rank(to_sq)
            if to_rank == 0 or to_rank == 7:
                move = chess.Move(from_sq, to_sq, chess.QUEEN)

    return move


def _play_game(nets: Networks) -> GameResult:
    start = time.perf_counter()
    history: List[Step] = []
    board = chess.Board()
    # Rolling window of board snapshots always encoded from white's perspective,
    # ordered most-recent first. Maintained externally because board.mirror()
    # produces a board with no move stack, so board_to_observation cannot
    # reconstruct history from it on its own.
    board_history: deque[chess.Board] = deque(maxlen=NUM_SNAPSHOTS)
    while not board.is_game_over():
        # Always encode from white's perspective: mirror the board when it's
        # black's turn.
        real_board = board.copy()
        board_history.appendleft(real_board)
        if board.turn == chess.BLACK:
            obs_board = board.copy().mirror()
            obs_history = [b.copy().mirror() for b in board_history]
        else:
            obs_board = real_board
            obs_history = list(board_history)

        device = next(nets.representation.parameters()).device
        observation = board_to_observation(obs_board, history=list(obs_history)).to(
            device
        )
        latent = nets.representation(observation.permute(2, 0, 1).unsqueeze(0))
        policy_logits = nets.policy(
            latent, create_mask(obs_board, device=device).view(-1).unsqueeze(0)
        )
        target_action, _target_policy, _target_value = get_target_policy(
            latent, policy_logits.squeeze(0), nets
        )
        mask = create_mask(obs_board, device=torch.device("cpu")).view(-1)
        board.push(
            action_to_chess_move(target_action, board, board.turn == chess.BLACK)
        )
        outcome = board.outcome()
        if outcome and outcome.winner is not None:
            # board.turn is the player to move next; the player who just moved is not board.turn.
            reward = 1 if outcome.winner == (not board.turn) else -1
        else:
            reward = 0
        history.append((observation, target_action, reward, mask))
    save_to_replay_buffer(history)
    final_outcome = board.outcome()
    winner = final_outcome.winner if final_outcome else None

    os.makedirs("games", exist_ok=True)
    game = chess.pgn.Game.from_board(board)
    game.headers["Result"] = board.result()
    game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    pgn_path = f"games/game_{timestamp}.pgn"
    with open(pgn_path, "w") as f:
        f.write(str(game) + "\n")

    elapsed = time.perf_counter() - start
    return history, winner, elapsed
