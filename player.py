import chess

from observation import board_to_observation
from tree import get_target_policy


def play_game():
    history = []
    board = chess.Board()
    # at each step, we:
    # record the observation (current board state)
    # get the target action using Gumbel tree search
    # execute the action in the environment
    # record the action and reward
    # repeat until the game is over
    while not board.is_game_over():
        observation = board_to_observation(board)
        latent = representation_function(observation)
        policy_logits = policy_function(latent)
        target_action, target_policy = get_target_policy(latent, policy_logits)
        board.push(target_action)
        history.append((observation, target_action, board.result()))
    return history
