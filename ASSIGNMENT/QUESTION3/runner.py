from tictactoe import initial_state, player, actions, result, winner, terminal, minimax

def print_board(board):
    print("  0 1 2")
    for i, row in enumerate(board):
        print(i, end=" ")
        for cell in row:
            print(cell if cell is not None else '-', end=" ")
        print()

def play_game():
    board = initial_state()
    human = input("Choose X or O: ").upper()
    while human not in ['X', 'O']:
        human = input("Invalid choice. Choose X or O: ").upper()
    
    ai = 'O' if human == 'X' else 'X'
    
    while not terminal(board):
        print_board(board)
        current = player(board)
        
        if current == human:
            print("Your turn!")
            while True:
                try:
                    row = int(input("Row (0-2): "))
                    col = int(input("Col (0-2): "))
                    if (row, col) in actions(board):
                        board = result(board, (row, col))
                        break
                    else:
                        print("Invalid move!")
                except ValueError:
                    print("Enter numbers only!")
        else:
            print("AI's turn...")
            move = minimax(board)
            board = result(board, move)
    
    print_board(board)
    w = winner(board)
    print("Game over!", f"{w} wins!" if w else "It's a tie!")

if __name__ == "__main__":
    play_game()