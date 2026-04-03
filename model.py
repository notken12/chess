import torch
import torch.nn as nn
import torch.nn.functional as F
import chess
import math

class ResBlock(nn.Module):
    def __init__(self, convChannels): #channels is 256 because we should be feeding the 256 x 8 x 8 into the resBlocks
        super().__init__()
        # We use padding=1 to keep the 8x8 spatial dimension the same, kernel size 3x3 conv filter
        self.conv1 = nn.Conv2d(convChannels, convChannels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(convChannels)
        self.conv2 = nn.Conv2d(convChannels, convChannels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(convChannels)

    def forward(self, x):
        identity = x 
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity  # The Skip Connection logic
        return F.relu(out)
    

class Representation(nn.Module):
    def __init__(self, obsChannels, convChannels): #obsChannels = 119, convChannels = 256
        super().__init__()

        self.conv1 = nn.Conv2d(obsChannels, convChannels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(convChannels)
        self.relu = nn.ReLU()
        self.resblocks = nn.Sequential(*[ResBlock(convChannels) for _ in range(10)])

    def forward(self, x): #x is the initial observation block
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.resblocks(x)
        return x
    
class Dynamics(nn.Module):
    def __init__(self, actionChannels, stateChannels, convChannels): #actions is now 73 x 8 x 8
        super().__init__()

        self.conv1 = nn.Conv2d(actionChannels + stateChannels, convChannels, kernel_size=3, padding=1, bias=False) #convert 329 x 8 x 8 to 256 x 8 x 8
        self.bn1 = nn.BatchNorm2d(convChannels)
        self.relu = nn.ReLU()
        self.resblocks = nn.Sequential(*[ResBlock(convChannels) for _ in range(10)])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(256, 3)

    def forward(self, state, action):       
        combined = torch.cat((state, action), dim=1)  
        nextState = self.relu(self.bn1(self.conv1(combined)))
        nextState = self.resblocks(nextState)
        rewardProbs = self.reward(nextState)
        return nextState, rewardProbs
    
    def reward(self, result):
        pooled = self.pool(result)
        flatten = torch.flatten(pooled, 1)
        rewardProbs = self.linear(flatten)
        return rewardProbs



def generateMoveLookup():
        files = 'abcdefgh'
        ranks = '12345678'
        letterMapping = {v : i for i,v in enumerate(files)}
        queenMove = {"N":0, "NE":1, "E":2, "SE":3, "S":4, "SW":5, "W":6, "NW":7}
        knightMove = {(1,2):56, (2,1):57, (2,-1):58, (1,-2):59, (-1,-2):60, (-2,-1):61, (-2,1):62, (-1,2):63}
        underPromoMapping = {"n":0, "b":1, "r":2}
        
        lookup = {}

        for f1 in range(8):
            for r1 in range(8):
                for f2 in range(8):
                    for r2 in range(8):
                        # Skip if start and end are the same
                        if f1 == f2 and r1 == r2: continue
                        
                        # Generate base UCI (e.g., "a1a2")
                        base_uci = f"{files[f1]}{ranks[r1]}{files[f2]}{ranks[r2]}"
                        
                        dx = f2 - f1
                        dy = r2 - r1
                        
                        distance = (dx, dy)
                        channel = None
                        
                        if distance[0] == 0: #vertical move
                            if distance[1] > 0 : #N
                                channel = queenMove["N"] * 7 + abs(distance[1]) - 1
                            else: #S
                                channel = queenMove["S"] * 7 + abs(distance[1]) - 1
                        elif distance[1] == 0: #horizontal move
                            if distance[0] > 0: #E
                                channel = queenMove["E"] * 7 + abs(distance[0]) - 1
                            else: #W
                                channel = queenMove["W"] * 7 + abs(distance[0]) - 1
                        elif abs(distance[0]) == abs(distance[1]): #diagonal move
                            if distance[0] == distance[1]: 
                                if distance[0] > 0: #NE
                                    channel = queenMove["NE"] * 7 + abs(distance[1]) - 1
                                else: #SW
                                    channel = queenMove["SW"] * 7 + abs(distance[1]) -1

                            elif distance[0] == -1 * distance[1]: 
                                if distance[0] > 0: #SE
                                    channel = queenMove["SE"] * 7 + abs(distance[1]) -1
                                else: #NW
                                    channel = queenMove["NW"] * 7 + abs(distance[1]) - 1
                        else: #knight move
                            channel = knightMove[distance]
                        
                        # Store standard move
                        if channel is not None and 0 <= channel < 64:
                            lookup[base_uci] = (channel, f1, r1)
                            # If this is a promotion rank, also add the 'q' version
                            if r2 == 7 or r2 == 0:
                                lookup[base_uci + 'q'] = (channel, f1, r1)

                        # 3. Underpromotions (Only possible if moving to last/first rank)
                        if r2 == 7 or r2 == 0:
                            # Only pawns promote, so dx must be -1, 0, or 1
                            if abs(dx) <= 1 and abs(dy) == 1:
                                for p_char, p_idx in underPromoMapping.items():
                                    promo_uci = base_uci + p_char
                                    # 64 + 3 directions * 3 pieces
                                    # We normalize dx to 0, 1, 2
                                    promo_channel = 64 + 3 * (dx + 1) + p_idx
                                    lookup[promo_uci] = (promo_channel, f1, r1)
        return lookup
MOVE_LOOKUP = generateMoveLookup()

class Policy(nn.Module):

    def __init__(self, actionChannels, convChannels):
        super().__init__()

        self.conv1 = nn.Conv2d(convChannels, actionChannels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(actionChannels)

    

    def createMask(board):
        mask = torch.full((73, 8, 8), -1e9)
    
        # Iterate through legal moves
        for move in board.legal_moves:
            uci = move.uci()
            if uci in MOVE_LOOKUP:
                z, x, y = MOVE_LOOKUP[uci]
                mask[z, x, y] = 0
            else:
                print(f"Warning: Move {uci} not found in lookup table.")
                
        return mask

    def forward(self, x, boards):
        logits = self.conv1(x)
        logits = self.bn1(logits)

        batch_size = logits.size(0)
        logits = logits.view(batch_size, -1)

        masks = []
        for b in boards:
            m = self.createMask(b).view(-1)
            masks.append(m)

        mask_stack = torch.stack(masks).to(logits.device)
        masked_logits = logits + mask_stack
        probs = torch.softmax(masked_logits, dim=1)

        return probs

class Value(nn.Module):
    def __init__(self, convChannels, numBins): # convChannels=256, numBins
        super().__init__()
        
        # Channel Reduction
        self.conv1 = nn.Conv2d(convChannels, 32, kernel_size=1) 
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        
        # Fully Connected Layers
        self.flatten = nn.Flatten()
        # 32 channels * 8x8 grid = 2048 inputs
        self.linear1 = nn.Linear(32 * 8 * 8, 256)
        self.linear2 = nn.Linear(256, numBins) 

    def forward(self, x):
        # x is the Latent State (Batch, 256, 8, 8)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.flatten(x)
        x = self.relu(self.linear1(x))
        
        # Output raw logits for the bins
        # During search, you'll Softmax these to get probabilities
        value_logits = self.linear2(x)
        return value_logits

                
        