Article Dataset Search Algorithms (Python)

---

This project evaluates the performance of various search algorithms on a dataset of research articles. It assesses the impact of data structures and algorithmic strategies on search efficiency.

Outcomes
	- Evaluate algorithm efficiency through empirical testing.
	- Assess algorithm suitability for different data structures.

---

Dataset Structure

The dataset should be a CSV file named `Article.csv` with the following columns:

ID, TITLE, ABSTRACT, Computer Science, Physics, Mathematics, Statistics, Quantitative Biology, Quantitative Finance

Each row represents one research article.

---

Features

- Implements eight search algorithm variants:
  - Linear Search
  - Binary Search
  - Jump Search
  - Interpolation Search
  - Each applied to both array (list) and linked list structures

- Automatically:
  - Sorts the data where needed
  - Randomly selects keys (both valid and invalid) for testing
  - Measures execution time for 30 runs per algorithm
  - Computes best, average, and worst-case performance
  - Visualizes results using a bar chart

---

Output

Generates a performance comparison chart showing search time (in microseconds) for each algorithm and data structure combination.

---

How to Run

1. Ensure Python 3 is installed.
2. Install required libraries:
   - pip install matplotlib numpy
3. Place `Article.csv` in the same directory as the script.
4. Run the script:
   - python search_article_dataset.py

---





