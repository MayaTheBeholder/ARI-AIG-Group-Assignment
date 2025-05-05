# Importing necessary functions from the tictactoe module
from tictactoe import (
    initial_state,  # Creates the initial empty board
    player,       # Determines which player's turn it is
    actions,        # Lists all possible moves from the current board
    result,      # Returns the board state after a given move
    winner,        # Determines the winner of the game
    terminal,      # Checks if the game is over
    minimax         # Implements the AI to calculate the optimal move
)

# ### Function to Print the Board
def print_board(board):
    """
    **Prints the current state of the Tic-Tac-Toe board.**
    - Displays the board with row and column indices for ease of interaction.
    - Empty cells are displayed as '-'.

    **Parameters:**
    - `board`: The 2D list representing the current state of the game board.
    """
    print("  0 1 2")  # Column indices
    for i, row in enumerate(board):  # Iterate through each row
        print(i, end=" ")  # Row index
        for cell in row:  # Iterate through each cell
            print(cell if cell is not None else '-', end=" ")  # Print the cell or '-' if empty
        print()  # Newline for the next row

# ### Main Game Loop
def play_game():
    """
    **Controls the flow of a Tic-Tac-Toe game.**
    - Allows a human player to choose X or O.
    - Alternates turns between the human player and an AI opponent.
    - Uses the minimax algorithm for the AI's optimal moves.
    - Ends when the game reaches a terminal state (win, loss, or draw).
    """
    # Initialize the board
    board = initial_state()

    # Ask the human player to choose X or O
    human = input("Choose X or O: ").upper()
    while human not in ['X', 'O']:  # Validate the input
        human = input("Invalid choice. Choose X or O: ").upper()

    ai = 'O' if human == 'X' else 'X'  # Assign the other symbol to the AI

    # Main game loop: continues until the game is over
    while not terminal(board):
        print_board(board)  # Display the current board
        current = player(board)  # Determine whose turn it is

        if current == human:  # Human's turn
            print("Your turn!")
            while True:  # Loop until the human makes a valid move
                try:
                    row = int(input("Row (0-2): "))  # Input for row
                    col = int(input("Col (0-2): "))  # Input for column
                    if (row, col) in actions(board):  # Check if the move is valid
                        board = result(board, (row, col))  # Update the board with the move
                        break
                    else:
                        print("Invalid move!")
                except ValueError:  # Handle invalid inputs (e.g., non-numeric)
                    print("Enter numbers only!")
        else:  # AI's turn
            print("AI's turn...")
            move = minimax(board)  # Calculate the optimal move using minimax
            board = result(board, move)  # Update the board with the AI's move

    # End of the game
    print_board(board)  # Display the final board
    w = winner(board)  # Determine the winner
    print("Game over!", f"{w} wins!" if w else "It's a tie!")  # Announce the result

# ### Run the Game
if __name__ == "__main__":
    """
    Entry point for the program.
    - Starts the Tic-Tac-Toe game.
    """
    play_game()