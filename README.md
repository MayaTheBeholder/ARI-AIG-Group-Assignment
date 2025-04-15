--------------------------------------------------------------------------------------------------------------------------------
**PROJECT PART 1: DEGREES OF SEPARATION**

*$ python degrees.py small
Loading data...
Data loaded.
Name: Jun Zhang
Name: Balázs Győrffy
3 degrees of separation.
1: Jun Zhang and Elı́
as Campo co-authored "Advanced design strategies for
Fe-based metal–organic frameworks"
2: Elı́ as Campo and David S. Liebeskind co-authored "Hungarian
Statistical Review"
3: David S. Liebeskind and Balázs Győrffy co-authored "Laser Excitation
of the Th-229 Nucleus"*


Write a Python program to find how many degrees of separation (shortest co-authorship path) exist between two scientists.

- **Inspired by**: The Six Degrees of Kevin Bacon game.  
- **Goal**: Find the shortest connection path between two scientists via co-authored papers.  
- **Example**: Jun Zhang → Elías Campo → David S. Liebeskind → Balázs GyőrMy (3 steps).  
- **Method**:  
  - **States** = Scientists  
  - **Actions** = Co-authored papers (connect scientists)  
  - **Search strategy** = Breadth-First Search (BFS)  
- **Purpose**: Model academic co-authorship as a search problem.

**Files:**
- scientists.csv: Maps scientist_id to names.
- papers.csv: Maps paper_id to titles and years.
- authors.csv: Links scientist_id with paper_id.
  
**Data Structures to Use:**
- Name to IDs:
  name_to_ids = {name: set of scientist_ids}
  
  - Scientist Info:
  scientists = {scientist_id: {'name': name, 'papers': set of paper_ids}}

- Paper Info:
  papers = {paper_id: {'title': title, 'year': year, 'authors': set of scientist_ids}}

Function: shortest_path(source, target)
Returns the shortest path from one scientist to another.
Output: List of (paper_id, scientist_id) pairs (as strings).
If no path exists, return None.
Path Example:
[(1, 2), (3, 4)] → source wrote paper 1 with scientist 2, who wrote paper 3 with scientist 4 (the target).
Function: neighbors_for_person(sci_id)
Returns all (paper_id, coauthor_id) pairs for a given scientist.
We can add other functions and use standard Python libraries.

**HINTS** 
Early Goal Check:
Check if the target scientist is a neighbor before adding a node to the frontier.
→ If yes, return the path immediately — faster and uses less memory.
Use Lecture Code:
You can reuse and modify:
Node (to track states and paths),
QueueFrontier (for BFS),
StackFrontier (if needed, but BFS needs a queue).


--------------------------------------------------------------------------------------------------------------------------------
**PROJECT PART 2: SUDOKU AI SOLVER**

*$ python sudoku_solver.py data/sudoku_easy.txt
5 3 4 | 6 7 8 | 9 1 2
6 7 2 | 1 9 5 | 3 4 8
1 9 8 | 3 4 2 | 5 6 7
------+-------+------
8 5 9 | 7 6 1 | 4 2 3
4 2 6 | 8 5 3 | 7 9 1
7 1 3 | 9 2 4 | 8 5 6
------+-------+------
9 6 1 | 5 3 7 | 2 8 4
2 8 7 | 4 1 9 | 6 3 5
3 4 5 | 2 8 6 | 1 7 9*

Treat as CSP (Constraint Satisfaction Problem)
  Variables = 81 cells (each cell at (row, col)).
  Domains = Digits 1–9.
  Constraints = No repeat in any row, column, or 3×3 box.

Main Components to Implement (in Sudoku_AI_solver class):
  enforce_node_consistency()
  → Remove impossible values from each cell based on the initial puzzle.
  revise(x, y)
  → If x and y share a constraint, remove values from x’s domain that can’t match any in y.
  ac3()
  → Apply the AC-3 algorithm to reduce the domains before solving (makes puzzle arc-consistent).
  assignment_complete(assignment)
  → Returns True if all cells are filled (no 0s).
  consistent(assignment)
  → Checks that current values don’t break Sudoku rules.
  order_domain_values(var, assignment)
  → Pick values for a cell that rule out the fewest options for neighbors (Least Constraining Value heuristic).
  select_unassigned_variable(assignment)
  → Pick the next cell using MRV (Minimum Remaining Values) and Degree heuristics.
  backtrack(assignment)
  → Use backtracking to try values, back up if needed, and solve the board.
  
Hints & Tips:
  Use a dict like {(row, col): set of possible values} for the board.
  Write helper functions for:
  Getting the 3×3 box a cell belongs to.
  Finding all neighbors of a cell.
  Start small: Get backtracking working first, then add AC-3 and heuristics.
  Optional: Build a GUI or save the solved board as text/image.

Testing:
  Download three Sudoku puzzles:
    sudoku_easy.txt
    sudoku_medium.txt
    sudoku_hard.txt
 Each file contains a 9x9 Sudoku grid with numbers separated by spaces. 0 represents an empty cell.
 Test the AI solver using these puzzles.

--------------------------------------------------------------------------------------------------------------------------------

**PROJECT PART 3: TRAFFIC SIGN RECOGNITION**

*$ python traffic_signs.py gtsrb model.h5
Loading data...
Data loaded.
Training model...
Model saved to model.h5
Evaluating model
Model accuracy: 0.9535*


















