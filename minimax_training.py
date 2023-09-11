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

    q_table[state[0]][state[1]][action] += learning_rate * (reward + discount_factor * np.max(q_table[next_state[0]][next_state[1]][:]) - q_table[state[0]][state[1]][action])


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


def evaluate(board):
    score = 0
    for row in range(ROWS):
        for col in range(COLS - 3):
            circles = [board[row][col], board[row][col+1], board[row][col+2], board[row][col+3]]
            score += count_score(circles, 2, 1)
            
    for row in range(ROWS - 3):
        for col in range(COLS):
            circles = [board[row][col], board[row+1][col], board[row+2][col], board[row+3][col]]
            score += count_score(circles, 2, 1)
    for row in range(ROWS):
        if board[row][3] == 2:  # 3 is middle of 7. 2 is for AI
            score += 6
        elif board[row][3] == 1:    # 1 is for trained-model
            score -= 4
    return score

def count_score(circles, player, opponent):
    score = 0
    if circles.count(player) == 4:
        score += 1000
    elif circles.count(opponent) == 4:
        score -= 1000
    elif circles.count(player) == 3 and circles.count(0) == 1:
        score += 30
    elif circles.count(opponent) == 3 and circles.count(0) == 1:
        score -= 28
    elif circles.count(player) == 2 and circles.count(0) == 2:
        score += 10
    elif circles.count(opponent) == 2 and circles.count(0) == 1:
        score -= 8

    return score

def minimax(board, depth, max_turn, max_depth):
    if check_win(board, 2):
        return None, 10000
    elif check_win(board, 1):
        return None,-10000
    elif check_draw(board):
        return None, 0
    if depth == max_depth:
        return None, evaluate(board)

    valid_columns = get_valid_columns(board)
    
    if max_turn == 2:   # AI is maximizing
        best_value = -10000
        column = random.choice(valid_columns)
        for col in valid_columns:
            row = get_empty_row(board, col)
            b_temp = np.copy(board)
            b_temp[row][col] = 2
            new_value = minimax(b_temp, depth+1, 1, max_depth)[1]
            if new_value > best_value:
                best_value = new_value
                column = col
        return column, best_value
    
    else:   # trained-model is minimizing
        best_value = 10000
        column = random.choice(valid_columns)
        for col in valid_columns:
            row = get_empty_row(board, col)
            b_temp = np.copy(board)
            b_temp[row][col] = 1
            new_value = minimax(b_temp, depth+1, 2, max_depth)[1]
            if new_value < best_value:
                best_value = new_value
                column = col
        return column, best_value

    
def play_minimax(board, max_depth):
    col = minimax(board,0, 2, max_depth)[0]
    row = get_empty_row(board, col)
    board[row][col] = 2

def play_game(epsilon):
    board = np.zeros((ROWS, COLS))
    current_player = 1
    game_over = False
    max_index = np.unravel_index(np.argmax(q_table[5]), q_table[5].shape)
    current_cell = [5, max_index[0]]    
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
                play_minimax(board, 1)

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


def choose_action(board, current_cell):
    row = current_cell[0]
    col = current_cell[1]
    max_action = float('-inf')
    valid = get_valid_columns(board)
    actions = q_table[row][col][:]
    action_index = float('-inf')
    for index, action in enumerate(actions):
        if action > max_action and index in valid:
            max_action = action
            action_index = index
    return action_index

def play_rl(board, current_cell):
    col = choose_action(board, current_cell)
    row = get_empty_row(board, col)
    board[row][col] = 1
    return [row, col]


def play_against(current_cell):
    board = np.zeros((ROWS, COLS))
    game_over = False
    current_player = 1
    board[current_cell[0]][current_cell[1]] = current_player
    max_depth = 1
    current_player = 2
    
    while not game_over:
        if current_player == 1:
            res = play_rl(board, current_cell)
            current_cell = res
            current_player = 2
        else:
            play_minimax(board, max_depth)
            current_player = 1
        if check_win(board, 1):
            return 1
        if check_win(board, 2):
            return 2
        if check_draw(board):
            return 0
        


# getting percentage of winning of trained model
trained_win = 0
AI_win = 0
max_index = np.unravel_index(np.argmax(q_table[5]), q_table[5].shape)
current_cell = [5, max_index[0]]    
for _ in range(num_episodes):
    result = play_against(current_cell)
    if result == 1:
        trained_win+=1
    elif result == 2:
        AI_win += 1

print(f"trained agent winning rate:{trained_win * 100 / num_episodes}")
print(f"trained agent losing rate:{AI_win * 100 / num_episodes}")

np.save('Q_table_minimax.npy', q_table)

