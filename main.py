import csv
import time
import random
import math
import matplotlib.pyplot as plt
import numpy as np

# ============================
# Article Class Definition
# ============================

class Article:
    """
    A class to represent a research article.
    Each article has an ID, title, abstract, and a dictionary of category flags.
    """
    def __init__(self, id, title, abstract, categories):
        self.id = id
        self.title = title
        self.abstract = abstract
        self.categories = categories  # Dictionary of fields: CS, Math, etc.

    def __eq__(self, other):
        return self.id == other.id  # Equality based on ID

    def __lt__(self, other):
        return self.id < other.id  # Needed for sorting

    def __str__(self):
        return f"{self.id}: {self.title}"


# ============================
# Data Loader
# ============================

def load_articles(csv_path, max_records=1000):
    """
    Loads articles from a CSV file with a header row.
    Returns a list of Article objects (up to max_records).
    """
    articles = []
    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for i, row in enumerate(reader):
            if i >= max_records:
                break
            try:
                id = int(row['ID'])
                title = row['TITLE']
                abstract = row['ABSTRACT']
                categories = {
                    "CS": int(row['Computer Science']),
                    "Physics": int(row['Physics']),
                    "Math": int(row['Mathematics']),
                    "Stats": int(row['Statistics']),
                    "Bio": int(row['Quantitative Biology']),
                    "Finance": int(row['Quantitative Finance']),
                }
                articles.append(Article(id, title, abstract, categories))
            except ValueError:
                continue  # Skip invalid records
    return articles


# ============================
# Linked List Implementation
# ============================

class Node:
    """Singly-linked list node for Article objects."""
    def __init__(self, article):
        self.value = article
        self.next = None

def array_to_linked_list(articles):
    """Converts a list of Article objects into a singly linked list."""
    if not articles:
        return None
    head = Node(articles[0])
    current = head
    for article in articles[1:]:
        current.next = Node(article)
        current = current.next
    return head


# ============================
# Search Algorithms
# ============================

def linear_search(data, key_id):
    """Linear search for Article ID in an array."""
    for i, article in enumerate(data):
        if article.id == key_id:
            return i
    return -1

def linear_search_linked(head, key_id):
    """Linear search for Article ID in a linked list."""
    current = head
    index = 0
    while current:
        if current.value.id == key_id:
            return index
        current = current.next
        index += 1
    return -1

def binary_search(data, key_id):
    """Binary search on a sorted array of Article objects."""
    left, right = 0, len(data) - 1
    while left <= right:
        mid = (left + right) // 2
        if data[mid].id == key_id:
            return mid
        elif data[mid].id < key_id:
            left = mid + 1
        else:
            right = mid - 1
    return -1

def jump_search(data, key_id):
    """Jump search for sorted array."""
    n = len(data)
    step = int(math.sqrt(n))
    prev = 0
    while prev < n and data[min(step, n) - 1].id < key_id:
        prev = step
        step += int(math.sqrt(n))
        if prev >= n:
            return -1
    for i in range(prev, min(step, n)):
        if data[i].id == key_id:
            return i
    return -1

def interpolation_search(data, key_id):
    """Interpolation search for uniformly distributed sorted array."""
    low, high = 0, len(data) - 1
    while low <= high and data[low].id <= key_id <= data[high].id:
        if data[high].id == data[low].id:
            break
        pos = low + ((key_id - data[low].id) * (high - low)) // (data[high].id - data[low].id)
        if pos >= len(data):
            break
        if data[pos].id == key_id:
            return pos
        elif data[pos].id < key_id:
            low = pos + 1
        else:
            high = pos - 1
    return -1


# ============================
# Execution + Timing
# ============================

def run_search(search_fn, key_id):
    """Measures the runtime of a single search in microseconds."""
    start = time.perf_counter()
    search_fn(key_id)
    end = time.perf_counter()
    return (end - start) * 1_000_000  # µs

def run_tests(articles):
    """Runs all 8 search algorithms on a dataset 30 times each, with random keys."""
    articles.sort(key=lambda x: x.id)  # Needed for binary/jump/interpolation
    linked_articles = array_to_linked_list(articles)
    article_ids = [a.id for a in articles]

    # Define all algorithm variations
    search_algorithms = [
        ("Linear (ArrayList)", lambda k: linear_search(articles, k)),
        ("Linear (LinkedList)", lambda k: linear_search_linked(linked_articles, k)),
        ("Binary (ArrayList)", lambda k: binary_search(articles, k)),
        ("Binary (LinkedList)", lambda k: binary_search(articles, k)),  # Not optimal
        ("Jump (ArrayList)", lambda k: jump_search(articles, k)),
        ("Jump (LinkedList)", lambda k: jump_search(articles, k)),      # Not optimal
        ("Interpolation (ArrayList)", lambda k: interpolation_search(articles, k)),
        ("Interpolation (LinkedList)", lambda k: interpolation_search(articles, k)),  # Not optimal
    ]

    results = {}

    for name, func in search_algorithms:
        times = []
        for _ in range(30):
            key = random.choice(article_ids) if random.random() < 0.5 else -1
            t = run_search(func, key)
            times.append(t)
        results[name] = {
            "best": min(times),
            "avg": sum(times) / len(times),
            "worst": max(times)
        }

    return results


# ============================
# Graph Plotting
# ============================

def plot_results(results):
    """Visualizes best, average, and worst-case runtimes."""
    labels = list(results.keys())
    best = [results[name]["best"] for name in labels]
    avg = [results[name]["avg"] for name in labels]
    worst = [results[name]["worst"] for name in labels]

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(15, 6))
    bars1 = ax.bar(x - width, best, width, label='Best')
    bars2 = ax.bar(x, avg, width, label='Average')
    bars3 = ax.bar(x + width, worst, width, label='Worst')

    ax.set_ylabel('Time (µs)')
    ax.set_title('Search Algorithm Performance on Article Dataset')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.4f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.show()


# ============================
# Main Entry Point
# ============================

if __name__ == "__main__":
    articles = load_articles("Article.csv")  # <-- Ensure this file is in the working directory
    results = run_tests(articles)
    plot_results(results)
