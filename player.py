from collections import deque
from typing import List, Tuple

import chess

from observation import NUM_SNAPSHOTS, board_to_observation
from tree import get_target_policy
from replay_buffer import Step, save_to_replay_buffer


def action_to_chess_move(action: int, mirrored: bool) -> chess.Move:
    # unimplemented
    # flip the action's coordinates around if it's black's turn
    # since the model flips the board
    # to always play from white's perspective
    pass


def play_game() -> List[Step]:
    history: List[Step] = []
    board = chess.Board()
    # Rolling window of board snapshots always encoded from white's perspective,
    # ordered most-recent first. Maintained externally because board.mirror()
    # produces a board with no move stack, so board_to_observation cannot
    # reconstruct history from it on its own.
    board_history: deque[chess.Board] = deque(maxlen=NUM_SNAPSHOTS)
    while not board.is_game_over():
        # Always encode from white's perspective: mirror the board when it's
        # black's turn. Use copy(stack=False) for white to avoid storing a
        # live reference that mutates after board.push().
        obs_board = board.mirror() if board.turn == chess.BLACK else board.copy(stack=False)
        board_history.appendleft(obs_board)
        observation = board_to_observation(obs_board, history=list(board_history))
        latent = nets.representation(observation.permute(2, 0, 1).unsqueeze(0))
        policy_logits = nets.policy(latent, boards=None)
        target_action, _target_policy, _target_value = get_target_policy(
            latent, policy_logits, nets
        )
        board.push(action_to_chess_move(target_action, board.turn == chess.BLACK))
        outcome = board.outcome()
        # if the game isn't over or it's a draw, we give a reward of 0
        # if our last move made us win, we give a reward of 1
        # there is no way to lose from making a move.
        reward = 1 if outcome and outcome.winner is not None else 0
        history.append((observation, target_action, reward))
    save_to_replay_buffer(history)
    return history
