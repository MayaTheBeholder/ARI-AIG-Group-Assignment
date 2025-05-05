import unittest
from degrees import shortest_path, neighbors_for_person

# Dummy data injection can be achieved by temporarily overriding the data structures
# For demonstration purposes, assume your code allows injection via global variables,
# or you have a function to load dummy data.

# Dummy data definitions:
dummy_scientists = {
    'A': {'name': 'Alice', 'papers': {'p1'}},
    'B': {'name': 'Bob', 'papers': {'p1', 'p2'}},
    'C': {'name': 'Carol', 'papers': {'p2'}},
}
dummy_papers = {
    'p1': {'title': 'Paper 1', 'year': 2025, 'authors': {'A', 'B'}},
    'p2': {'title': 'Paper 2', 'year': 2025, 'authors': {'B', 'C'}},
}

class TestDegreesSeparation(unittest.TestCase):
    def setUp(self):
       
        import degrees  
        degrees.scientists = dummy_scientists
        degrees.papers = dummy_papers

    def test_neighbors_for_person(self):
        expected = {('p1', 'A'), ('p2', 'C')}
        result = neighbors_for_person('B')
        self.assertEqual(result, expected)

    def test_shortest_path_success(self):
        expected_path = [('p1', 'B'), ('p2', 'C')]
        result = shortest_path('A', 'C')
        self.assertEqual(result, expected_path)

    def test_shortest_path_no_connection(self):
        result = shortest_path('A', 'NonExistent')
        self.assertIsNone(result)

if __name__ == '__main__':
    unittest.main()
