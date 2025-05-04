import copy
import math

EMPTY = None

def initial_state():
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]

def player(board):
    x_count = sum(row.count('X') for row in board)
    o_count = sum(row.count('O') for row in board)
    return 'X' if x_count == o_count else 'O'

def actions(board):
    possible_actions = set()
    for i in range(3):
        for j in range(3):
            if board[i][j] == EMPTY:
                possible_actions.add((i, j))
    return possible_actions

def result(board, action):
    if action not in actions(board):
        raise Exception("Invalid action")
    
    new_board = copy.deepcopy(board)
    new_board[action[0]][action[1]] = player(board)
    return new_board

def winner(board):
    # check row
    for row in board:
        if row.count('X') == 3:
            return 'X'
        if row.count('O') == 3:
            return 'O'
    
    #check column
    for col in range(3):
        if board[0][col] == board[1][col] == board[2][col] != EMPTY:
            return board[0][col]
    
    # check diagonal
    if board[0][0] == board[1][1] == board[2][2] != EMPTY:
        return board[0][0]
    if board[0][2] == board[1][1] == board[2][0] != EMPTY:
        return board[0][2]
    
    return None

def terminal(board):
    if winner(board) is not None:
        return True
    return all(cell is not EMPTY for row in board for cell in row)

def utility(board):
    win = winner(board)
    if win == 'X':
        return 1
    elif win == 'O':
        return -1
    else:
        return 0

def minimax(board):
    if terminal(board):
        return None
    
    current_player = player(board)
    
    if current_player == 'X':
        value, move = max_value(board)
    else:
        value, move = min_value(board)
    
    return move

def max_value(board):
    if terminal(board):
        return utility(board), None
    
    v = -math.inf
    best_move = None
    
    for action in actions(board):
        new_value, _ = min_value(result(board, action))
        if new_value > v:
            v = new_value
            best_move = action
            if v == 1:  #early end if winning move found
                break
                
    return v, best_move

def min_value(board):
    if terminal(board):
        return utility(board), None
    
    v = math.inf
    best_move = None
    
    for action in actions(board):
        new_value, _ = max_value(result(board, action))
        if new_value < v:
            v = new_value
            best_move = action
            if v == -1:  # early end if winning move found
                break
                
    return v, best_move

# alphabeta pruning vers 
def minimax_ab(board, alpha=-math.inf, beta=math.inf):
    if terminal(board):
        return None
    
    current_player = player(board)
    
    if current_player == 'X':
        value, move = max_value_ab(board, alpha, beta)
    else:
        value, move = min_value_ab(board, alpha, beta)
    
    return move

def max_value_ab(board, alpha, beta):
    if terminal(board):
        return utility(board), None
    
    v = -math.inf
    best_move = None
    
    for action in actions(board):
        new_value, _ = min_value_ab(result(board, action), alpha, beta)
        if new_value > v:
            v = new_value
            best_move = action
            alpha = max(alpha, v)
            if alpha >= beta:
                break
                
    return v, best_move

def min_value_ab(board, alpha, beta):
    if terminal(board):
        return utility(board), None
    
    v = math.inf
    best_move = None
    
    for action in actions(board):
        new_value, _ = max_value_ab(result(board, action), alpha, beta)
        if new_value < v:
            v = new_value
            best_move = action
            beta = min(beta, v)
            if alpha >= beta:
                break
                
    return v, best_move