import unittest
from sudoku_solver import solve_sudoku, is_valid_sudoku

sample_puzzle = [
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9],
]

solved_puzzle = [
    [5,3,4,6,7,8,9,1,2],
    [6,7,2,1,9,5,3,4,8],
    [1,9,8,3,4,2,5,6,7],
    [8,5,9,7,6,1,4,2,3],
    [4,2,6,8,5,3,7,9,1],
    [7,1,3,9,2,4,8,5,6],
    [9,6,1,5,3,7,2,8,4],
    [2,8,7,4,1,9,6,3,5],
    [3,4,5,2,8,6,1,7,9],
]

class TestSudokuSolver(unittest.TestCase):
    def test_solver_completes_puzzle(self):
        result = solve_sudoku(sample_puzzle)
        # Ensure there are no zeros left
        self.assertTrue(all(0 not in row for row in result))
        # And that the solution is valid
        self.assertTrue(is_valid_sudoku(result))
    
    def test_validity_of_correct_solution(self):
        self.assertTrue(is_valid_sudoku(solved_puzzle))
    
    def test_solver_on_complete_puzzle(self):
        result = solve_sudoku(solved_puzzle)
        self.assertEqual(result, solved_puzzle)

if __name__ == '__main__':
    unittest.main()
