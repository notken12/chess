
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

## Action Space (probably better)
8 x 8 x 5
The first two channels represent the start position and end position of the piece. The last 3 channels represent the underpromotion type, and are either filled with ones if it underpromoted to that piece or 0s otherwise.
We can easily tack these 5 channels onto the latent state and provide to the dynamics function.


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

Takes our 8x8x119 observation and outputs latent state with our chosen shape 8x8x256. This shape allows us to use the first two dimensions to represent the 8x8 board and the third dimension is channels that represent concepts for each cell in the board, such as white's bishops' level of control over that cell.

First:
Apply a 3x3x119x256 conv layer to convert 8x8x119 observation into the shape of our latent state (8x8x256).
Then, stack a bunch of ResNet modules on top of each other to keep iterating on the values while preserving our 8x8x256 shape until we get to our final output.

ResNet:
Input: 8x8x256
1. Apply the first 3×3×256×256 convolution.
2. Apply Batch Normalization and ReLU.
3. Apply a second 3×3×256×256 convolution.
4. Apply Batch Normalization.
5. The Skip Connection: Take the original 8×8×256 tensor that entered the block and mathematically add it to the output of step 4.
6. Apply a final ReLU.
Output: 8x8x256


### Dynamic Function
G: s_t+1, r_t = G(s_t, a_t)

Predicts the next state and the reward given the current state and an action.

We add the action into the input by putting it side by side with the state. Our state is 8x8x256, and our action is 8x8x5, so we just add those 5 extra channels to get 8x8x261.

Apply a 3x3x261x256 conv layer to convert the input into the shape of the latent state.
Then use the same ResNet stack architecture as above to get the predicted next latent state, s^_t+1.

However, we should not compare this latent state output directly with the latent state output of our observation function, since this would lead to representation collapse where the model gets lazy and outputs zeros for everything since that would result in perfect consistency. We instead pass the predicted latent state through P1 and P2, which are both MLPs, to get P2(P1(s^_t+1)). We pass the latent state given by the representation function, s_t+1, through P1, giving P1(s_t+1). We take the cosine similarity loss between the two and add a stop-gradient on P1(s_t+1) to prevent the representation function from adjusting its weights. This makes it quite hard for the model to cheat because it has to ensure consistency even after this extra projection step.

Note on reward: Chess has no intermediate rewards. You can attach a tiny MLP to predict the step reward, but it will essentially learn to constantly output 0 until a terminal state is reached. But the predicted rewards are still needed for the math of the Multi-Step TD Target to work out, so we'll just use a really simple network.
- Take the output of the dynamics network (the predicted next state)
- Flatten or pool the tensor
- Pass it through a single dense layer with 3 output neurons. Those neurons will represent the probabilities of the 3 possible rewards: -1 for a loss, 0 for draw, and 1 for a win. We follow EfficientZero's approach of outputting a probability distribution instead of the expected value of the reward. See "Value Function" for details.

### Policy Function
P: p_t = P(s_t)

Outputs a distribution of moves weighted by predicted Q-value given the current state. This gives us a fast "instinctual" guess of the next best move.

First, we again convert the dimensions of the input to that of the output. We employ a (1x1 or 3x3)x256x5 channel to get an output of 8x8x5, the dimensions of our action space. 

Optional: we can stack some ResNet layers to iterate.

Flatten, apply a legal move mask (force illegal moves to -infinity), then apply a softmax to turn raw logits into a probability distribution.

### Value Function
V: v_t = V(s_t)

Predicts the "value" of a state (how "good" the model thinks it is).

- Channel reduction: use a 1x1 convolution to drastically reduce the channel count from 256 to 32 or even 1.
- Flatten into a 1D vector
- One or two fully-connected layers (e.g. 256 neurons) with ReLU activation
- Categorical output: in standard RL, we would just output the expected value of v_t, but EfficientZero V2 generates the discrete probability distribution of v_t. It splits the range of possible v_t values into equally sized bins, and outputs a probability for each bin. E.g., it can output 10% chance of -1.0 to 0.0 and 90% chance of 0.0 to 1.0. We can decide how many bins we want and the number of output neurons would be the number of bins.

