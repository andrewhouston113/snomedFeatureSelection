import pandas as pd
from tqdm import tqdm
import networkx as nx
import numpy as np

class determine_best_concepts:

    def __init__(self, child_parents_path, instance_data_path, groups_path):
        """
        Initialize the determine_best_concepts class.

        Parameters:
            child_parents_path (str): Path to the file containing child-parent relationships data.
            instance_data_path (str): Path to the file containing instance data.
            groups_path (str): Path to the file containing groups data.
        """

        # Create SNOMED-CT Graph
        self.snomed_ct_graph = self._create_snomed_graph(child_parents_path)

        # Assign Instances to Nodes
        instance_data = self._read_instance_data(instance_data_path)
        for node in self.snomed_ct_graph.nodes:
            if node not in instance_data:
                instance_data[node] = set()

        concept_depth_dict = nx.shortest_path_length(self.snomed_ct_graph,'138875005')
        sorted_shortest_paths = dict(sorted(concept_depth_dict.items(), key=lambda item: item[1], reverse=True))

        # Calculating Relative Depth
        self.relative_depth_dict = {}
        for node in tqdm(self.snomed_ct_graph.nodes(), desc="Calculating Relative Depth"):
            descendants = len(nx.descendants(self.snomed_ct_graph, node))
            ancestors = len(nx.ancestors(self.snomed_ct_graph, node))
            self.relative_depth_dict[node] = ancestors/(descendants+ancestors)

        # Subsume Concepts
        for node in tqdm(sorted_shortest_paths, desc="Subsuming Concepts"):
            if node in self.snomed_ct_graph:
                descendents = nx.descendants(self.snomed_ct_graph, node)
                children_lists = [instance_data[descendant] for descendant in descendents if descendant in instance_data]
                child_instances = set().union(*children_lists)
                instance_data[node] = instance_data[node].union(child_instances)

        # Load Groups
        self.group_instance_map = self._read_group_data(groups_path)
        self.groups = list(self.group_instance_map.keys())
        self.group_size = {group:len(self.group_instance_map[group]) for group in self.groups}
        
        # Assigning Node Contents to Groups
        self.grouped_node_attributes = {}
        for node in tqdm(instance_data, desc="Assigning Node Contents to Groups"):
            if node in self.snomed_ct_graph:
                self.grouped_node_attributes[node] = {group: [id for id in instance_data[node] if id in self.group_instance_map[group]] for group in self.groups}
        
        # Calculate Difference Score
        for node in tqdm(self.grouped_node_attributes, desc="Calculating Differences"):
            self.grouped_node_attributes[node]['precision'] = self._average_precision(node)
            self.grouped_node_attributes[node]['difference'] = self._calculate_difference(node)

    def select_concepts(self, min_diff, min_depth, n_concepts, metric):
        """
        Select concepts based on minimum difference and number of concepts required.

        Parameters:
            min_diff (float): The minimum difference score for selecting a concept.
            n_concepts (int): The maximum number of concepts to select.

        Returns:
            list: A list of selected concept nodes.
        """
        # Sort nodes according to their difference, breaking ties with the deeper concept
        sorted_node_attributes = dict(sorted(self.grouped_node_attributes.items(), key=lambda item: item[1][metric], reverse=True))

        # For each node, append the "best node" to the list, until the min_diff is not met or n_concepts is reached or there are no more features
        best_nodes = []
        lineage = []
        for node in sorted_node_attributes:
            if self.grouped_node_attributes[node][metric] < min_diff or len(best_nodes) >= n_concepts or len(best_nodes) == len(self.grouped_node_attributes):
                break

            if (node in lineage) | (self.relative_depth_dict[node] < min_depth):
                pass
            else:
                best_nodes.append(node)
                lineage += nx.descendants(self.snomed_ct_graph,node)
                lineage += nx.ancestors(self.snomed_ct_graph,node)
    
        return best_nodes

    def _create_snomed_graph(self, child_parents_path):
        knowledge_graph = nx.DiGraph()
        with open(child_parents_path, 'r') as file:
            # Skipping the header line
            next(file)

            for line in file:
                child, parent = line.strip().split(",")
                knowledge_graph.add_edge(parent, child)
        return knowledge_graph
    
    def _read_instance_data(self, instance_data_path):
        instance_graphs = {}
        with open(instance_data_path, 'r') as file:
            # Skipping the header line
            file.readline()

            for line in file:
                cols = line.strip().split(",")
                if len(cols) == 3:  # Ensure all columns are present
                    instance_id, _, code = cols
                    instance_graphs.setdefault(code, set()).add(instance_id)

        return instance_graphs
    
    def _read_group_data(self, cohort_path):
        with open(cohort_path, 'r') as file:
            next(file)
            cohort_instance_map = {}
            for line in file:
                instance_id, cohort = line.strip().split(",")
                cohort_instance_map.setdefault(cohort, set()).add(instance_id)

            return cohort_instance_map
        
    def _calculate_difference(self, node):
        """
        Calculate the difference in proportions of instances between groups for a given node.

        Parameters:
            node: The node for which the difference is calculated.

        Returns:
            float: The mean difference in proportions between groups.
        """
        num_groups = len(self.groups)
        
        differences = []
        for i in range(num_groups):
            for j in range(i+1, num_groups):
                group_a = self.groups[i]
                group_b = self.groups[j]
                
                group_a_size = self.group_size[group_a]
                group_b_size = self.group_size[group_b]
                
                group_a_instance_size = len(self.grouped_node_attributes[node][group_a])
                group_b_instance_size = len(self.grouped_node_attributes[node][group_b])
                
                p_a = group_a_instance_size / group_a_size if group_a_size != 0 else 0
                p_b = group_b_instance_size / group_b_size if group_b_size != 0 else 0
                
                differences.append(abs(p_a - p_b))
        
        return np.mean(differences)

    def _average_precision(self, node):
        num_groups = len(self.groups)

        average_precision = []
        for i in range(num_groups):
            group_a = self.groups[i]
            other_groups = [group for group in self.groups if group != group_a]
            instances_a_node = self.grouped_node_attributes[node][group_a]
            
            group_a_size = self.group_size[group_a]
            cohort_proportion = group_a_size/(group_a_size+sum(self.group_size[group] for group in other_groups))

            true_positives_a = len(instances_a_node)

            if true_positives_a > group_a_size*0.01:
                false_positives_a = sum(len(self.grouped_node_attributes[node][group]) for group in other_groups)
                average_precision.append(abs(self.precision(true_positives_a, false_positives_a)-cohort_proportion))
            else:
                average_precision.append(0)
        return np.mean(average_precision)

                
    def precision(self, true_positives, false_positives):
        if true_positives + false_positives == 0:
            return 0
        return true_positives / (true_positives + false_positives)