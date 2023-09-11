####### Fatemeh Mirzaei Kalani#######

import random
import pygame
import numpy as np
pygame.init()

pygame.font.init()
WIDTH, HEIGHT = 560, 700
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Connect4")

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
BLUE = (0, 0, 255)

font = pygame.font.SysFont('Comic Sans MS', 35)


ROWS = 6
COLS = 7

FPS = 60
board = np.full((ROWS, COLS), -1)

player = 0 # 0 for Yellow(Human) and 1 for Blue(AI)
q_table = np.load('Q_table_random.npy')    
# q_table = np.load('Q_table_minimax.npy')  # uncomment this if you wanna test it


def draw_circles(board):
    for row in range(ROWS):
        for col in range(COLS):
            if board[row][col] == -1:
                color = WHITE
            elif board[row][col] == 0:
                color = YELLOW
            else:
                color = BLUE
            pygame.draw.rect(WIN, RED, [col * 80, row * 80, 80, 80])
            pygame.draw.circle(WIN, color, [int((col + 0.5) * 80), int((row + 0.5) * 80)], int(80 * 0.4), 0)
    
def get_empty_row(board, column):
    for i in range(ROWS - 1, -1, -1):
        if board[i][column] == -1:
            return i
    return -1

def check_win(board):
    for i in range(ROWS):
        for j in range(COLS - 3):
            if board[i][j] != -1 and board[i][j] == board[i][j + 1] == board[i][j + 2] == board[i][j + 3]:
                return board[i][j]
    
    for i in range(ROWS - 3):
        for j in range(COLS):
            if board[i][j] != -1 and board[i][j] == board[i + 1][j] == board[i + 2][j] == board[i + 3][j]:
                return board[i][j]
    return -1

def check_draw(board):
    for col in range(COLS):
        if board[0][col] == -1:
            return False
    return True

def evaluate(board, player):
    score = 0
    for row in range(ROWS):
        for col in range(COLS - 3):
            circles = [board[row][col], board[row][col+1], board[row][col+2], board[row][col+3]]
            score += count_score(circles, player, abs(player - 1))
            
    for row in range(ROWS - 3):
        for col in range(COLS):
            circles = [board[row][col], board[row+1][col], board[row+2][col], board[row+3][col]]
            score += count_score(circles, player, abs(player - 1))
    for row in range(ROWS):
        if board[row][3] == 1:  # 3 is middle of 7
            score += 6
        elif board[row][3] == 0:
            score -= 4
    return score

def count_score(circles, player, opponent):
    score = 0
    if circles.count(player) == 4:
        score += 1000
    elif circles.count(opponent) == 4:
        score -= 1000
    elif circles.count(player) == 3 and circles.count(-1) == 1:
        score += 30
    elif circles.count(opponent) == 3 and circles.count(-1) == 1:
        score -= 28
    elif circles.count(player) == 2 and circles.count(-1) == 2:
        score += 10
    elif circles.count(opponent) == 2 and circles.count(-1) == 1:
        score -= 8

    return score

def get_valid_columns(board):
    valid = []
    for i in range(COLS):
        if board[0][i] == -1:
            valid.append(i)
    return valid

def minimax(board, depth, max_turn):
    winner = check_win(board)
    if winner == 1:
        return None, 10000
    elif winner == 0:
        return None,-10000
    elif check_draw(board):
        return None, 0
    if depth == 4:
        return None, evaluate(board, 1)

    valid_columns = get_valid_columns(board)
    
    if max_turn == 1:   # AI is maximizing
        best_value = -10000
        column = random.choice(valid_columns)
        for col in valid_columns:
            row = get_empty_row(board, col)
            b_temp = np.copy(board)
            b_temp[row][col] = 1
            new_value = minimax(b_temp, depth+1, 0)[1]
            if new_value > best_value:
                best_value = new_value
                column = col
        return column, best_value
    
    else:
        best_value = 10000
        column = random.choice(valid_columns)
        for col in valid_columns:
            row = get_empty_row(board, col)
            b_temp = np.copy(board)
            b_temp[row][col] = 0
            new_value = minimax(b_temp, depth+1, 1)[1]
            if new_value < best_value:
                best_value = new_value
                column = col
        return column, best_value

def play_minimax(board):
    col = minimax(board,0, 1)[0]
    row = get_empty_row(board, col)
    board[row][col] = 1

def alphabeta(board, alpha, beta, depth, max_turn):
    winner = check_win(board)
    if winner == 1:
        return None, 10000
    elif winner == 0:
        return None,-10000
    elif check_draw(board):
        return None, 0
    if depth == 6:
        return None, evaluate(board, 1)

    valid_columns = get_valid_columns(board)
    
    if max_turn == 1:   # AI is maximizing
        value = -10000
        column = random.choice(valid_columns)
        for col in valid_columns:
            row = get_empty_row(board, col)
            b_temp = np.copy(board)
            b_temp[row][col] = 1
            new_score = alphabeta(b_temp, alpha, beta, depth+1, 0)[1]
            if new_score > value:
                value = new_score
                column = col
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return column, value
    
    else:
        value = 10000
        column = random.choice(valid_columns)
        for col in valid_columns:
            row = get_empty_row(board, col)
            b_temp = np.copy(board)
            b_temp[row][col] = 0
            new_score = alphabeta(b_temp, alpha, beta, depth+1, 1)[1]
            if new_score < value:
                value = new_score
                column = col
            beta = min(beta, value)
            if alpha >= beta:
                break
        return column, value


def play_alphabeta(board):
    col = alphabeta(board,-10_000, 10_000, 0 , 1)[0]
    row = get_empty_row(board, col)
    board[row][col] = 1

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

def main_page(board, player): 
    logo = pygame.image.load("logo1.png")
    button_image1 = pygame.image.load("alpha.png")
    button_image2 = pygame.image.load("minimax.png")
    button_image3 = pygame.image.load("rl.png")

    button_rect = logo.get_rect()
    button_rect1 = button_image1.get_rect()
    button_rect2 = button_image2.get_rect()
    button_rect3 = button_image3.get_rect()

    button_rect.centerx = WIN.get_width() // 2
    button_rect.bottom = WIN.get_height() - 500

    button_rect1.centerx = WIN.get_width() // 2
    button_rect1.bottom = WIN.get_height() - 400

    button_rect2.centerx = WIN.get_width() // 2
    button_rect2.bottom = WIN.get_height() - 300

    button_rect3.centerx = WIN.get_width() // 2
    button_rect3.bottom = WIN.get_height() - 200

    WIN.fill(WHITE)
    
    WIN.blit(logo, button_rect)
    WIN.blit(button_image1, button_rect1)
    WIN.blit(button_image2, button_rect2)
    WIN.blit(button_image3, button_rect3)

    pygame.display.update()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
            if event.type == pygame.MOUSEBUTTONDOWN and button_rect1.collidepoint(event.pos):
                main(board, player, "AlphaBeta")
            elif event.type == pygame.MOUSEBUTTONDOWN and button_rect2.collidepoint(event.pos):
                main(board, player, "MiniMax")
            elif event.type == pygame.MOUSEBUTTONDOWN and button_rect3.collidepoint(event.pos):
                main(board, player, "RL")
        

def main(board, player, game_type):
    
    current_cell = [5, random.choice(list(range(7)))]
    if game_type == "RL":
        board[current_cell[0]][current_cell[1]] = 1
    clock = pygame.time.Clock()
    run = True
    while run:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if reset_rect.collidepoint(event.pos):
                    board = np.full((ROWS, COLS), -1)
                    if game_type == "RL":
                        player = 1
                    else:
                        player = 0
                    break
                if player == 0: # 0 for human
                    column = int(pygame.mouse.get_pos()[0] / 80)
                    row = get_empty_row(board, column)
                    if row != -1:
                        board[row][column] = 0
                        player = 1
                
               
        WIN.fill(WHITE)
        draw_circles(board)
        ## reset Button
        reset = pygame.image.load("reset.png")
        reset_rect = reset.get_rect()
        reset_rect.centerx = WIN.get_width() // 2
        reset_rect.bottom = WIN.get_height() 
        WIN.blit(reset, reset_rect)
        ##
        winner = check_win(board)
        if winner == 0:
            run = False
            text = font.render(f"You won!", True, BLACK)
            text_rect = text.get_rect()
            text_rect.centerx = WIN.get_width() // 2
            text_rect.bottom = WIN.get_height() - 80
            WIN.blit(text, text_rect)
            pygame.display.update()
            pygame.time.delay(3000)
            continue

        elif check_draw(board):
            run = False
            text = font.render(f"Draw!", True, BLACK)
            text_rect = text.get_rect()
            text_rect.centerx = WIN.get_width() // 2
            text_rect.bottom = WIN.get_height() - 80
            WIN.blit(text, text_rect)
            pygame.display.update()
            pygame.time.delay(3000)
            continue


        pygame.display.update()

        if player == 1:
            if game_type == "AlphaBeta":
                play_alphabeta(board)
            elif game_type == "MiniMax":
                play_minimax(board)
            elif game_type == "RL":
                res = play_rl(board, current_cell)
                current_cell = res
            player = 0
            WIN.fill(WHITE)
            draw_circles(board)
            ## reset button
            reset = pygame.image.load("reset.png")
            reset_rect = reset.get_rect()
            reset_rect.centerx = WIN.get_width() // 2
            reset_rect.bottom = WIN.get_height() 
            WIN.blit(reset, reset_rect)
            ##
            winner = check_win(board)
            if winner == 1:
                run = False
                text = font.render(f"AI won!", True, BLACK)
                text_rect = text.get_rect()
                text_rect.centerx = WIN.get_width() // 2
                text_rect.bottom = WIN.get_height() - 80
                WIN.blit(text, text_rect)
                pygame.display.update()
                pygame.time.delay(3000)

            elif check_draw(board):
                run = False
                text = font.render(f"Draw!", True, BLACK)
                text_rect = text.get_rect()
                text_rect.centerx = WIN.get_width() // 2
                text_rect.bottom = WIN.get_height() - 80
                WIN.blit(text, text_rect)
                pygame.display.update()
                pygame.time.delay(3000)

            pygame.display.update()


    pygame.quit()


main_page(board, player)
