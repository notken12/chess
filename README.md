
## RL Environment
python-chess

## Action Space

Actions are encoded by the square of the piece being moved and the movement being made.
8 x 8 x 73

Types of movements:
A: Queen-like sliding moves:
8 possible directions * 7 distances = 56 movements
B: Knight jumps: 8 movements
C: Underpromotions
When a pawn moves to the end of the board using a Category A move, it's assumed to promote to a queen. To represent an underpromotion, we encode the movement (either up and to the left, up, or up and to the right) and the piece being promoted to (bishop, knight, rook), for a total of 3 x 3 = 9 underpromotion movements.
Total: 73 movements

## Observation Space
8 x 8 x 119 channels
The 8 x 8 represents the board.
For each player and each type of piece, it has a channel of 0s and 1s that represents if that player's piece type is present on a given square. 2 players * 6 piece types = 12 channels
Repetitions: a global channel that's set to all 1s if the current board state has repeated once and another channel for if the board state has repeated twice. 2 channels.
We will provide 8 historical snapshots so the model can see the recent board history:
14 channels * 8 historical snapshots = 112 channels.

Global state and rules:
Color to Move: all 1s if it's white's turn, else 0s.
White's castling rights: all 1s if true, else 0s
Black's castling rights: same logic
No-progress count: number of halfmoves since the last capture or pawn move divided by 100 so that it scales to 0.0 to 1.0.
Total move count: encoded as a number from 0.0 to 1.0, either by dividing by a maximum expected move count like 200 or using asymptotic scaling (tanh).

## Network Architecture

### Representation Function
H: s_t = H(o_t)
Outputs the latent state given an observation. 

### Dynamic Function
G: s_t+1, r_t = G(s_t, a_t)
Predicts the next state and the reward given the current state and an action.

### Policy Function
P: p_t = P(s_t)
Outputs a distribution of moves weighted by predicted Q-value given the current state.

### Value Function
V: v_t = V(s_t)
Predicts the "value" of a state (how "good" the model thinks it is).

