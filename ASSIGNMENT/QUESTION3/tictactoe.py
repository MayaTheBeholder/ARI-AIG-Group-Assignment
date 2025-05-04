import math
import copy

EMPTY = None

def initial_state():
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]

def player(board):
    x = sum(row.count('X') for row in board)
    o = sum(row.count('O') for row in board)
    return 'X' if x == o else 'O'

def actions(board):
    return [(i,j) for i in range(3) for j in range(3) if board[i][j] == EMPTY]

def result(board, action):
    if board[action[0]][action[1]] != EMPTY:
        raise Exception("Invalid action")
    
    new_board = copy.deepcopy(board)
    new_board[action[0]][action[1]] = player(board)
    return new_board

def winner(board):
    # check rows and columns
    for i in range(3):
        if board[i][0] == board[i][1] == board[i][2] != EMPTY:
            return board[i][0]
        if board[0][i] == board[1][i] == board[2][i] != EMPTY:
            return board[0][i]
    
    #check diagonal
    if board[0][0] == board[1][1] == board[2][2] != EMPTY:
        return board[0][0]
    if board[0][2] == board[1][1] == board[2][0] != EMPTY:
        return board[0][2]
    
    return None

def terminal(board):
    return winner(board) is not None or all(cell is not EMPTY for row in board for cell in row)

def utility(board):
    w = winner(board)
    return 1 if w == 'X' else -1 if w == 'O' else 0

def minimax(board):
    if terminal(board):
        return None
    
    if player(board) == 'X':
        return max_value(board)[1]
    else:
        return min_value(board)[1]

def max_value(board):
    if terminal(board):
        return (utility(board), None)
    
    v = -math.inf
    best_action = None
    
    for action in actions(board):
        new_v = min_value(result(board, action))[0]
        if new_v > v:
            v = new_v
            best_action = action
            if v == 1:  #early end if winning move found
                break
    
    return (v, best_action)

def min_value(board):
    if terminal(board):
        return (utility(board), None)
    
    v = math.inf
    best_action = None
    
    for action in actions(board):
        new_v = max_value(result(board, action))[0]
        if new_v < v:
            v = new_v
            best_action = action
            if v == -1:  # early end if winning move found
                break
    
    return (v, best_action)