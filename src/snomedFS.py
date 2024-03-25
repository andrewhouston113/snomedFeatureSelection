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
        # Read Data
        self.child_parents_df = pd.read_csv(child_parents_path)
        self.instance_data_df = pd.read_csv(instance_data_path)
        self.instance_groups_df = pd.read_csv(groups_path)

        # Get Group Labels
        self.group_instance_map = {}
        for _, row in self.instance_groups_df.iterrows():
            self.group_instance_map.setdefault(row['label'], set()).add(row["id"])
        
        self.groups = list(self.group_instance_map.keys())
        self.group_size = {group:len(self.group_instance_map[group]) for group in self.groups}

        # Create SNOMED-CT Graph
        kg_directed = self._create_snomed_graph(self.child_parents_df, graph_type='directed')
        kg_undirected = kg_directed.to_undirected()

        # Assign Instances to Nodes
        node_attributes = {}
        for _, row in self.instance_data_df.iterrows():
            node_attributes.setdefault(row["code"], set()).add(row["id"])

        for node in kg_directed.nodes:
            if node not in node_attributes:
                node_attributes[node] = []

        nx.set_node_attributes(kg_directed, node_attributes, 'cohort')

        # Subsume Concepts
        for node in tqdm(node_attributes, desc="Subsuming Concepts"):
            descendents = nx.descendants(kg_directed, node)
            children_lists = [node_attributes[descendant] for descendant in descendents if descendant in node_attributes]
            node_attributes[node] = list(set().union(*children_lists, node_attributes[node]))

        # Assigning Node Contents to Groups
        self.grouped_node_attributes = {}
        for node in tqdm(node_attributes, desc="Assigning Node Contents to Groups"):
            self.grouped_node_attributes[node] = {group: [id for id in node_attributes[node] if id in self.group_instance_map[group]] for group in self.groups}
        
        # Calculate Difference Score and Depths for Each Node
        for node in tqdm(self.grouped_node_attributes, desc="Calculating Depth and Differences"):
            self.grouped_node_attributes[node]['difference'] = self._calculate_difference(node)
            self.grouped_node_attributes[node]['depth'] = self._get_concept_distance(kg_undirected, node, 138875005)

    def select_concepts(self, min_diff, n_concepts):
        """
        Select concepts based on minimum difference and number of concepts required.

        Parameters:
            min_diff (float): The minimum difference score for selecting a concept.
            n_concepts (int): The maximum number of concepts to select.

        Returns:
            list: A list of selected concept nodes.
        """
        # Sort nodes according to their difference, breaking ties with the deeper concept
        sorted_node_attributes = dict(sorted(self.grouped_node_attributes.items(), key=self._custom_sort_key))

        # For each node, append the "best node" to the list, until the min_diff is not met or n_concepts is reached or there are no more features
        best_nodes = []
        for node in sorted_node_attributes:
            if self.grouped_node_attributes[node]['difference'] < min_diff or len(best_nodes) >= n_concepts or len(best_nodes) == len(self.grouped_node_attributes):
                break
            best_nodes.append(node)
        return best_nodes

    def _create_snomed_graph(self, relationships_df, graph_type='both'):
        """
        Create SNOMED CT graphs based on provided relationships.

        Parameters:
            relationships_df (pd.DataFrame): DataFrame containing SNOMED CT relationships.
                It should have 'sourceId' and 'destinationId' columns for edges.
            
            graph_type (str, optional): The type of graph to create.
                Possible values:
                    - 'directed': Create a directed graph.
                    - 'undirected': Create an undirected graph.
                    - 'both' (default): Create both directed and undirected graphs.

        Returns:
            nx.Graph or nx.DiGraph: A NetworkX graph object based on the specified graph_type.
        
        Raises:
            ValueError: If the 'graph_type' argument is not one of the allowed values.
            ValueError: If the 'relationships_df' is not a DataFrame or missing required columns.
        """
        
        # Check if 'graph_type' is valid
        if graph_type not in ['directed', 'undirected', 'both']:
            raise ValueError("Invalid 'graph_type'. Allowed values are 'directed', 'undirected', or 'both'.")
        
        # Validate the input DataFrame
        if not isinstance(relationships_df, pd.DataFrame):
            raise ValueError("Input 'relationships_df' is not a DataFrame.")
            
        required_columns = ['sourceId', 'destinationId']
        missing_columns = [col for col in required_columns if col not in relationships_df.columns]
        if missing_columns:
            raise ValueError(f"'relationships_df' must include the following columns: {', '.join(required_columns)}")

        # Create a directed graph
        G = nx.DiGraph()

        # Add nodes and edges from the provided DataFrame
        for _, row in tqdm(relationships_df.iterrows(), total=len(relationships_df), desc="Building Graph"):
            G.add_edge(row['destinationId'], row['sourceId'])

        if graph_type == 'directed':
            return G
        elif graph_type == 'undirected':
            return G.to_undirected()
        else:  # Default: 'both'
            snomed_graphs = {
                'directed': G,
                'undirected': G.to_undirected()
            }
            return snomed_graphs

    def _get_concept_distance(self, G, source_id, target_id):
        """
        Calculate the shortest path length between two nodes in an undirected graph, considering edge weights if specified.

        Parameters:
            G (networkx.Graph): The undirected graph in which to calculate the distance.
            source_id (int): The source node.
            target_id (int): The target node.
            weighted (bool or None, optional): Whether to consider the edge weights in the distance calculation.
                If None, it will default to True if the graph has edge weights, or False if it doesn't.

        Returns:
            int: The shortest path length between the source and target nodes. Returns -1 if no path exists.

        Raises:
            ValueError: If 'G' is not an undirected graph.
        """
        # Check if 'G' is an undirected graph
        if not isinstance(G, nx.Graph):
            raise ValueError("'G' must be a graph (networkx.Graph).")

        try:
            if G.is_directed():
                G_ = G.to_undirected()
                distance = nx.shortest_path_length(G_, source=source_id, target=target_id)
            else:
                distance = nx.shortest_path_length(G, source=source_id, target=target_id)
            return distance
        except nx.NetworkXNoPath:
            return -1  # No path exists
        
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
    
    def _custom_sort_key(self, item):
        """
        Custom sorting key for sorting nodes based on difference and depth.

        Parameters:
            item: The item to be sorted.

        Returns:
            tuple: A tuple containing difference and depth values for sorting.
        """
        # Sort by difference then depth, largest difference and depth first
        return (-item[1]["difference"], -item[1]["depth"])