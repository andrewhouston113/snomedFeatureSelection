import networkx as nx
from tqdm import tqdm
from collections import Counter

class SNOMEDGraphTool:
    """
    A tool for managing and analyzing a SNOMED CT hierarchy graph.
    """
    
    def __init__(self, relationships, descriptions, X=None, y_dict=None, 
                 code_column='snomedCode', id_column='patient_id', depth_method = 'absolute'):
        """
        Initializes the SNOMEDGraphTool with the provided data and builds the graph.
        
        Parameters:
        relationships (DataFrame): The DataFrame containing relationship data.
        descriptions (DataFrame): The DataFrame containing description data.
        X (DataFrame): The DataFrame containing patient data with SNOMED codes.
        y_dict (dict): A dictionary mapping patient IDs to labels.
        code_column (str, optional): The name of the column containing SNOMED codes in X. Defaults to 'snomedCode'.
        id_column (str, optional): The name of the column containing patient IDs in X. Defaults to 'patient_id'.
        depth_method (str, optional): The method used to calculate depth. Option include 'absolute' or 'relative'.
        """
        if depth_method not in  ['absolute', 'relative']:
            raise ValueError(f"Invalid depth_method: {depth_method}. Must be either 'absolute' or 'relative'.")

        self.G = nx.DiGraph()
        self.relationships = relationships
        self.code_column = code_column
        self.id_column = id_column
        self.concept_dict = dict(zip(descriptions['conceptId'], descriptions['term']))
        self.depth_method = depth_method

        self.build_graph()
        if (X is not None) & (y_dict is not None):
            self.all_codes_used = X[code_column].unique()
            self.X = X
            self.y_dict = y_dict
            self.filter_graph()
            self.add_node_attributes()
            self.update_all_nodes_with_descendant_ids()
            self.map_ids_to_labels()

    def build_graph(self):
        """
        Builds the directed graph using the relationships data.
        """
        active_relationships = self.relationships[
            (self.relationships['typeId'] == 116680003) &  # 'Is a' relationship
            (self.relationships['active'] == 1)
        ]
        for _, row in tqdm(active_relationships.iterrows(), total=active_relationships.shape[0], desc='Building Graph'):
            self.G.add_edge(row['destinationId'], row['sourceId'])

    def filter_graph(self):
        """
        Filters the graph to include only the nodes that are in all_codes_used and their ancestors.
        """
        nodes_to_keep = set(self.all_codes_used)

        for node in self.all_codes_used:
            if node in self.G:
                ancestors = nx.ancestors(self.G, node)
                nodes_to_keep.update(ancestors)

        filtered_graph = self.G.subgraph(nodes_to_keep).copy()
        self.G = filtered_graph

    def add_node_attributes(self):
        """
        Adds attributes (depth, label, ids) to the nodes in the graph.
        """
        root = [n for n, d in self.G.in_degree() if d == 0][0]
        depth = nx.single_source_shortest_path_length(self.G, root)

        for node in tqdm(self.G.nodes, total=len(self.G.nodes), desc='Assigning Attributes'):
            if self.depth_method == 'absolute':
                descendants_depth = [depth.get(descendant,0) for descendant in nx.descendants(self.G, node)]
                self.G.nodes[node]['depth'] = depth.get(node,0)/max(descendants_depth) if len(descendants_depth) > 0 else 1
            elif self.depth_method == 'relative':
                descendants = len(nx.descendants(self.G, node))
                ancestors = len(nx.ancestors(self.G, node))
                self.G.nodes[node]['depth'] =ancestors/(descendants+ancestors)

            self.G.nodes[node]['label'] = self.concept_dict[node]
            self.G.nodes[node]['ids'] = set(self.X.query(f'{self.code_column} == {node}')[self.id_column])

    def update_all_nodes_with_descendant_ids(self):
        """
        Updates each node with the IDs of all its descendant nodes.
        """
        for node in tqdm(self.G.nodes, total=len(self.G.nodes), desc='Updating Nodes'):
            ids = self.G.nodes[node].get('ids', set())
            descendants = nx.descendants(self.G, node)

            for descendant in descendants:
                ids.update(self.G.nodes[descendant].get('ids', set()))

            self.G.nodes[node]['ids'] = ids

    def map_ids_to_labels(self):
        """
        Maps patient IDs to labels for each node and counts the labels.
        """
        for node in tqdm(self.G.nodes, total=len(self.G.nodes), desc='Mapping Nodes'):
            self.G.nodes[node]['label_counts'] = dict(Counter([self.y_dict[id] for id in self.G.nodes.get(node)['ids']]))
    
    def assign_patients_to_nodes(self, X):
        """
        Reassigns patients to nodes and reperforms the subsumption process
        """
        self.X = X
        self.add_node_attributes()
        self.update_all_nodes_with_descendant_ids()
        self.map_ids_to_labels()


    def score_nodes(self, scorer):
        """
        Scores the nodes using the provided scoring function.
        
        Parameters:
        scorer (function): A function that takes the graph and a node, and returns a score.
        """
        for node in tqdm(self.G.nodes, total=len(self.G.nodes), desc='Scoring Nodes'):
            self.G.nodes[node]['score'] = scorer(self.G, node)

    def weight_scores(self, weight):
        """
        Weights the node scores by their depth.
        
        Parameters:
        weight (float): The weight to apply to the node depth.
        """
        for node in tqdm(self.G.nodes, total=len(self.G.nodes), desc='Weighting Node Scores'):
            self.G.nodes[node]['weighted_score'] = abs(self.G.nodes[node]['score']) * (1 + (self.G.nodes[node]['depth'] * weight))

    def get_eligible_nodes(self, scorer, total_patients, rarity_threshold=0.05, min_depth=0.1, weight=None):
        """
        Gets the eligible nodes based on their scores, avoiding ancestors and descendants of selected nodes.
        
        Parameters:
        scorer (function): A function that takes the graph and a node, and returns a score.
        weight (float, optional): The weight to apply to the node depth. Defaults to None.
        
        Returns:
        list: A list of eligible nodes sorted by their weighted scores.
        """
        self.score_nodes(scorer)
        self.weight_scores(weight if weight else 0)

        sorted_nodes = sorted(self.G.nodes(data=True), key=lambda x: x[1].get('weighted_score', 0), reverse=True)
        
        tabu = []
        candidate_nodes = []
        while len(sorted_nodes) > 0:
            if (sum(sorted_nodes[0][1]['label_counts'].values()) > (total_patients*rarity_threshold)) & (sorted_nodes[0][1]['depth'] > min_depth):
                if sorted_nodes[0][0] not in tabu:
                    candidate_nodes += [sorted_nodes[0]]
                    tabu += [sorted_nodes[0][0]]
                    tabu += [v for v in nx.ancestors(self.G, sorted_nodes[0][0])]
                    tabu += [v for v in nx.descendants(self.G, sorted_nodes[0][0])]

            sorted_nodes.pop(0)

        return candidate_nodes
