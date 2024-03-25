## Purpose

This code is designed to facilitate the selection of optimal SNOMED concepts for the prediction of class labels by leveraging techniques such as graph analysis and statistics to identify concepts that best fit specified criteria, such the difference in class prevelance at a given node.

## Usage

To use the `determine_best_concepts` class, follow these steps:

1. Instantiate the class with paths to input CSV files containing child-parent relationships, instance data, and group data.
2. Call the `select_concepts()` method with parameters specifying the minimum difference and number of concepts to select.
3. Access the selected concepts from the returned list for further analysis or processing.

```python
# Example Usage
from determine_best_concepts import determine_best_concepts

# Instantiate the class
concept_selector = determine_best_concepts(child_parents_path='child_parents.csv',
                                           instance_data_path='instance_data.csv',
                                           groups_path='groups.csv')

# Select concepts based on criteria
selected_concepts = concept_selector.select_concepts(min_diff=0.1, n_concepts=10)

# Print selected concepts
for concept in selected_concepts:
    print(concept)
