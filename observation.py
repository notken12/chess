import math
import chess
import torch

PIECE_TYPES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
COLORS = [chess.WHITE, chess.BLACK]

NUM_SNAPSHOTS = 8
CHANNELS_PER_SNAPSHOT = 14  # 12 piece planes + 2 repetition planes
GLOBAL_CHANNELS = 7          # color, 4x castling, no-progress, move count
TOTAL_CHANNELS = NUM_SNAPSHOTS * CHANNELS_PER_SNAPSHOT + GLOBAL_CHANNELS  # 119


def _encode_snapshot(board: chess.Board, tensor: torch.Tensor, channel_offset: int) -> None:
    """Encode a single board snapshot into 14 channels starting at channel_offset."""
    for color_idx, color in enumerate(COLORS):
        for piece_idx, piece_type in enumerate(PIECE_TYPES):
            channel = channel_offset + color_idx * len(PIECE_TYPES) + piece_idx
            for sq in board.pieces(piece_type, color):
                rank, file = sq // 8, sq % 8
                tensor[rank, file, channel] = 1.0

    rep1_channel = channel_offset + 12
    rep2_channel = channel_offset + 13
    if board.is_repetition(2):
        tensor[:, :, rep1_channel] = 1.0
    if board.is_repetition(3):
        tensor[:, :, rep2_channel] = 1.0


def board_to_observation(board: chess.Board) -> torch.Tensor:
    """
    Convert a python-chess Board into an (8, 8, 119) float32 observation tensor.

    The 119 channels are:
      - 112 channels: 8 historical snapshots x 14 channels each
          - 12 piece planes (2 colors x 6 piece types, binary)
          - 2 repetition planes (all-1s if repeated once / twice)
      - 7 global channels derived from the current board:
          - color to move, white/black castling rights (x2 each),
            no-progress count, total move count
    """
    tensor = torch.zeros(8, 8, TOTAL_CHANNELS, dtype=torch.float32)

    # Reconstruct historical snapshots via the move stack
    copy = board.copy(stack=True)
    for snapshot_idx in range(NUM_SNAPSHOTS):
        if snapshot_idx > 0 and not copy.move_stack:
            break
        if snapshot_idx > 0:
            copy.pop()
        channel_offset = snapshot_idx * CHANNELS_PER_SNAPSHOT
        _encode_snapshot(copy, tensor, channel_offset)

    # Global state channels (from the current board)
    base = NUM_SNAPSHOTS * CHANNELS_PER_SNAPSHOT  # 112

    tensor[:, :, base + 0] = 1.0 if board.turn == chess.WHITE else 0.0
    tensor[:, :, base + 1] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
    tensor[:, :, base + 2] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
    tensor[:, :, base + 3] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
    tensor[:, :, base + 4] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0
    tensor[:, :, base + 5] = board.halfmove_clock / 100.0
    tensor[:, :, base + 6] = math.tanh(board.fullmove_number / 100.0)

    return tensor
