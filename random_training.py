import numpy as np
import random


ROWS = 6
COLS = 7
NUM_ACTIONS = 7

q_table = np.zeros((ROWS, COLS, NUM_ACTIONS))

epsilon = 1.0                 # Exploration rate
max_epsilon = 1.0             # Exploration probability at start
min_epsilon = 0.01            # Minimum exploration probability 
decay_rate = 0.01 

def update_q_table(state, action, reward, next_state):
    learning_rate = 0.1
    discount_factor = 0.9  

    q_table[state[0]][state[1]][action] += learning_rate * (reward + discount_factor * np.max(q_table[next_state[0]][next_state[1]]) - q_table[state[0]][state[1]][action])


def check_win(board, player):
    for row in range(ROWS):
        for col in range(COLS - 3):
            if board[row][col] == player and board[row][col+1] == player and board[row][col+2] == player and board[row][col+3] == player:
                return True

    for row in range(ROWS - 3):
        for col in range(COLS):
            if board[row][col] == player and board[row+1][col] == player and board[row+2][col] == player and board[row+3][col] == player:
                return True

    return False

def get_valid_columns(board):
    valid = []
    for i in range(COLS):
        if board[0][i] == 0:
            valid.append(i)
    return valid

def check_draw(board):
    for col in range(COLS):
        if board[0][col] == 0:
            return False
    return True

def get_empty_row(board, column):
    for i in range(ROWS - 1, -1, -1):
        if board[i][column] == 0:
            return i
    return -1

# epsilon-greedy strategy
def choose_action(state, epsilon):
    if random.uniform(0, 1) < epsilon:
        action = random.randint(0, NUM_ACTIONS - 1)
    else:
        row = state[0]
        col = state[1]
        action = np.argmax(q_table[row, col, :])
    return action

def play_game(epsilon):
    board = np.zeros((ROWS, COLS))
    game_over = False
    max_index = np.unravel_index(np.argmax(q_table[5]), q_table[5].shape)
    current_cell = [5, max_index[0]]  
    current_player = 1
    board[current_cell[0]][current_cell[1]] = current_player
    current_player = 2

    while not game_over:
        # Player turn
        if current_player == 1:
            action = choose_action(current_cell, epsilon) 
            col = action

            if board[0][col] != 0:
                continue

            row = get_empty_row(board, col)
            board[row][col] = current_player

            if check_win(board, current_player):
                reward = 1  
                game_over = True
            elif check_draw(board):
                reward = 0.5  
                game_over = True
            else:
                reward = 0 

            next_cell = [row, col]

            update_q_table(current_cell, action, reward, next_cell)

            current_cell = next_cell

            current_player = 2

        # AI's turn (player 2)
        else:
            valid_columns = get_valid_columns(board)
            if len(valid_columns) == 0:
                reward = 0.5  # Draw
                game_over = True
                current_player = 1
                continue
            else:
                col = random.choice(valid_columns)

            row = get_empty_row(board, col)
            board[row][col] = current_player

            if check_win(board, current_player):
                reward = -1
                game_over = True
            elif check_draw(board):
                reward = 0.5 
                game_over = True
            else:
                reward = 0 

            current_player = 1


    return reward

# Training loop
num_episodes = 10000
win = 0
lose = 0

for episode in range(num_episodes):
    reward = play_game(epsilon)
    epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode) 
    if reward == 1:
        win+=1
    elif reward == -1:
        lose+=1
print(f"while training: wining percent of trained agent:{win*100/num_episodes}")
print(f"while training: losing percent of trained agent:{lose*100/num_episodes}")


np.save('Q_table_random.npy', q_table)


