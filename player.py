from collections import deque
from typing import List

import chess

from model import Networks
from observation import NUM_SNAPSHOTS, board_to_observation
from tree import get_target_policy
from replay_buffer import Step, save_to_replay_buffer


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


def play_game(nets: Networks) -> List[Step]:
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
        real_board = board.copy(stack=False)
        board_history.appendleft(real_board)
        if board.turn == chess.BLACK:
            obs_board = board.mirror()
            obs_history = [b.mirror() for b in board_history]
        else:
            obs_board = real_board
            obs_history = list(board_history)

        observation = board_to_observation(obs_board, history=list(obs_history))
        latent = nets.representation(observation.permute(2, 0, 1).unsqueeze(0)).squeeze(
            0
        )
        policy_logits = nets.policy(latent, boards=None).squeeze(0)
        target_action, _target_policy, _target_value = get_target_policy(
            latent, policy_logits, nets
        )
        board.push(
            action_to_chess_move(target_action, board, board.turn == chess.BLACK)
        )
        outcome = board.outcome()
        # if the game isn't over or it's a draw, we give a reward of 0
        # if our last move made us win, we give a reward of 1
        # there is no way to lose from making a move.
        reward = 1 if outcome and outcome.winner is not None else 0
        history.append((observation, target_action, reward))
    save_to_replay_buffer(history)
    return history
