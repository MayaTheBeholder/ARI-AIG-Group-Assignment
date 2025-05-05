# ### Constants and Initial Setup

import math  # Used for infinity values in minimax
import copy  # To duplicate the board for simulation purposes

# **Define constant for an empty cell**
EMPTY = None

# ### Function: Initial State
def initial_state():
    """
    **Creates the initial board state.**
    - Returns a 3x3 grid initialized with `EMPTY` values.
    - Represents an empty Tic-Tac-Toe board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]

# ### Function: Player
def player(board):
    """
    **Determines the player to move next.**
    - Counts the occurrences of 'X' and 'O' on the board.
    - If equal, it's 'X's turn; otherwise, it's 'O's turn.

    **Parameters:**
    - `board`: The current board state.

    **Returns:**
    - 'X' or 'O' (the next player).
    """
    x = sum(row.count('X') for row in board)
    o = sum(row.count('O') for row in board)
    return 'X' if x == o else 'O'

# ### Function: Actions
def actions(board):
    """
    **Lists all possible actions (valid moves).**
    - Scans the board for empty cells.

    **Parameters:**
    - `board`: The current board state.

    **Returns:**
    - A list of tuples representing row and column indices of valid moves.
    """
    return [(i, j) for i in range(3) for j in range(3) if board[i][j] == EMPTY]

# ### Function: Result
def result(board, action):
    """
    **Applies an action to the board and returns the new board state.**

    **Parameters:**
    - `board`: The current board state.
    - `action`: A tuple `(row, col)` representing the desired move.

    **Returns:**
    - A new board state with the action applied.

    **Raises:**
    - Exception if the action is invalid (e.g., cell is already occupied).
    """
    if board[action[0]][action[1]] != EMPTY:
        raise Exception("Invalid action")

    new_board = copy.deepcopy(board)  # Create a copy of the board
    new_board[action[0]][action[1]] = player(board)  # Place the player's symbol
    return new_board

# ### Function: Winner
def winner(board):
    """
    **Checks if there is a winner.**
    - Scans rows, columns, and diagonals for three identical symbols in a line.

    **Parameters:**
    - `board`: The current board state.

    **Returns:**
    - 'X' or 'O' if there's a winner, otherwise `None`.
    """
    # Check rows and columns
    for i in range(3):
        if board[i][0] == board[i][1] == board[i][2] != EMPTY:
            return board[i][0]
        if board[0][i] == board[1][i] == board[2][i] != EMPTY:
            return board[0][i]

    # Check diagonals
    if board[0][0] == board[1][1] == board[2][2] != EMPTY:
        return board[0][0]
    if board[0][2] == board[1][1] == board[2][0] != EMPTY:
        return board[0][2]

    return None  # No winner

# ### Function: Terminal
def terminal(board):
    """
    **Checks if the game is over.**
    - The game ends if there is a winner or the board is completely filled.

    **Parameters:**
    - `board`: The current board state.

    **Returns:**
    - `True` if the game is over, otherwise `False`.
    """
    return winner(board) is not None or all(cell is not EMPTY for row in board for cell in row)

# ### Function: Utility
def utility(board):
    """
    **Assigns a utility value to the board state.**
    - +1 for an 'X' win, -1 for an 'O' win, 0 otherwise.

    **Parameters:**
    - `board`: The current board state.

    **Returns:**
    - Utility value: 1, -1, or 0.
    """
    w = winner(board)
    return 1 if w == 'X' else -1 if w == 'O' else 0

# ### Function: Minimax
def minimax(board):
    """
    **Implements the Minimax algorithm.**
    - Calculates the optimal move for the current player.

    **Parameters:**
    - `board`: The current board state.

    **Returns:**
    - The optimal action (row, col) for the current player.
    """
    if terminal(board):
        return None  # No moves if the game is over

    if player(board) == 'X':  # Maximizing player
        return max_value(board)[1]
    else:  # Minimizing player
        return min_value(board)[1]

# ### Function: Max Value
def max_value(board):
    """
    **Calculates the maximum utility value achievable for 'X'.**
    - Recursively evaluates all possible moves.

    **Parameters:**
    - `board`: The current board state.

    **Returns:**
    - A tuple `(utility, action)`, where `action` is the optimal move for 'X'.
    """
    if terminal(board):
        return (utility(board), None)

    v = -math.inf
    best_action = None

    for action in actions(board):
        new_v = min_value(result(board, action))[0]
        if new_v > v:
            v = new_v
            best_action = action
            if v == 1:  # Early exit if a winning move is found
                break

    return (v, best_action)

# ### Function: Min Value
def min_value(board):
    """
    **Calculates the minimum utility value achievable for 'O'.**
    - Recursively evaluates all possible moves.

    **Parameters:**
    - `board`: The current board state.

    **Returns:**
    - A tuple `(utility, action)`, where `action` is the optimal move for 'O'.
    """
    if terminal(board):
        return (utility(board), None)

    v = math.inf
    best_action = None

    for action in actions(board):
        new_v = max_value(result(board, action))[0]
        if new_v < v:
            v = new_v
            best_action = action
            if v == -1:  # Early exit if a winning move is found
                break

    return (v, best_action)