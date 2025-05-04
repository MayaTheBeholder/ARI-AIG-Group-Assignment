## Contents

1. [Degrees of Separation (Search Algorithms)](#1-degrees-of-separation-search-algorithms)
2. [Sudoku Solver (Constraint Satisfaction)](#2-sudoku-solver-constraint-satisfaction)
3. [Traffic Sign Recognition (Machine Learning)](#3-traffic-sign-recognition-machine-learning)
4. [Submission Instructions](#submission-instructions)
5. [References and Resources](#references-and-resources)

---

## 1. Degrees of Separation (Search Algorithms)

**Marks:** 30

### Objective

Build a Python program to determine how many "degrees of separation" exist between two scientists based on academic co-authorship networks.

### Approach

- Model the problem as a graph search where:
  - Nodes are scientists
  - Edges are papers co-authored
- Implement **Breadth-First Search (BFS)** to find the shortest path.
- Input data is provided as CSV files:
  - `scientists.csv` – scientist ID and name
  - `papers.csv` – paper ID, title, year
  - `authors.csv` – links between scientists and papers

### Required Functions

- `shortest_path(source, target)`:
  - Returns a list of (paper_id, scientist_id) tuples.
- `neighbors_for_person(scientist_id)`:
  - Returns a set of co-authors and shared papers.

### Example Command

```bash
$ python degrees.py small
```

### Expected Output

```
3 degrees of separation.
1: Scientist A and Scientist B co-authored "Paper 1"
2: ...
```

---

## 2. Sudoku Solver (Constraint Satisfaction)

**Marks:** 40

### Objective

Create an AI to solve 9×9 Sudoku puzzles using constraint satisfaction principles.

### Approach

Model each cell as a variable with a domain of [1–9] and apply:
- Node consistency
- Arc consistency (AC-3)
- Backtracking search with inference

### Required Class and Functions

`Sudoku_AI_solver` class with:

- `enforce_node_consistency()`
- `revise(x, y)`
- `ac3()`
- `assignment_complete(assignment)`
- `consistent(assignment)`
- `order_domain_values(var, assignment)`
- `select_unassigned_variable(assignment)`
- `backtrack(assignment)`

### Example Command

```bash
$ python sudoku_solver.py data/sudoku_easy.txt
```

### Expected Output

A completed, valid Sudoku grid printed to the terminal.

---

## 3. Traffic Sign Recognition (Machine Learning)

**Marks:** 30

### Objective

Train a deep learning model using TensorFlow to classify German traffic signs.

### Dataset

Use the **GTSRB (German Traffic Sign Recognition Benchmark)**:
- 43 categories
- Over 50,000 images

### Workflow

1. Preprocess images:
   - Resize to 30×30 pixels
   - Normalise pixel values
2. Split data into training/testing sets
3. Build and train a Convolutional Neural Network (CNN)
4. Evaluate model and save it (`model.h5`)

### Example Command

```bash
$ python traffic_signs.py gtsrb model.h5
```

### Required Output

- Model accuracy (e.g., `Model accuracy: 0.9535`)
- Saved model file
- Evaluation metrics (e.g., confusion matrix)

---

## Submission Instructions

1. Upload your final Python notebook (with Markdown explanations and test cases).
2. Ensure your code is hosted on a public GitHub repository with:
   - This `README.md`
   - A valid copyright
   - All group members contributing
3. Add `naftalindeapo` as a repository contributor.
4. Submit links to:
   - GitHub repository
   - PDF version of the final notebook

---

## References and Resources

- [Best Practices for GitHub Repositories](https://docs.github.com/en/repositories/creating-and-managing-repositories/best-practices-for-repositories)
- [Markdown Cheat Sheet](https://www.markdownguide.org/cheat-sheet/)
- [OpenAlex Database](https://openalex.org/)
- [GTSRB Dataset](https://benchmark.ini.rub.de/?section=gtsrb&subsection=news)
- [TensorFlow](https://www.tensorflow.org/)
- [CS50 AI Projects](https://cs50.harvard.edu/ai/2024/projects/5/traffic/)
