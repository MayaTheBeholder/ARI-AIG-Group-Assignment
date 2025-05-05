import unittest
from degrees_impl import shortest_path, neighbors_for_person


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
        
        self.scientists = dummy_scientists
        self.papers = dummy_papers

    def test_neighbors_for_person(self):
        expected = {('p1', 'A'), ('p2', 'C')}
    
        result = neighbors_for_person('B', self.scientists, self.papers)
        self.assertEqual(result, expected)

    def test_shortest_path_success(self):
        expected_path = [('p1', 'B'), ('p2', 'C')]
        result = shortest_path('A', 'C', self.scientists, self.papers)
        self.assertEqual(result, expected_path)

    def test_shortest_path_no_connection(self):
        result = shortest_path('A', 'NonExistent', self.scientists, self.papers)
        self.assertIsNone(result)

if __name__ == '__main__':
    unittest.main()
