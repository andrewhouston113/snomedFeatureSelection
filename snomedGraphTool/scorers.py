import math
from scipy.stats import chi2_contingency

def agg_difference(g, node, label_totals):
    """
    Calculates the difference between the proportions of two labels for a given node.
    
    Parameters:
    g (networkx.Graph): The graph containing the node.
    node (int): The node for which the difference is calculated.
    label_totals (dict): A dictionary with labels as keys and their total counts as values.
    
    Returns:
    float: The difference between the proportions of the second label and the first label.
    """
    perc = []
    for i in label_totals.keys():
        if i in g.nodes[node]['label_counts']:
            perc.append(g.nodes[node]['label_counts'][i] / label_totals[i])
        else:
            perc.append(0)
    return perc[1] - perc[0]

def entropy(proportions):
    """
    Calculates the entropy given a list of proportions.
    
    Parameters:
    proportions (list): A list of proportions for different labels.
    
    Returns:
    float: The entropy value.
    """
    entropy_value = 0
    for p in proportions:
        if p > 0:
            entropy_value -= p * math.log(p, 2)
    return 1 - entropy_value

def agg_entropy(g, node, label_totals):
    """
    Calculates the entropy of label proportions for a given node.
    
    Parameters:
    g (networkx.Graph): The graph containing the node.
    node (int): The node for which the entropy is calculated.
    label_totals (dict): A dictionary with labels as keys and their total counts as values.
    
    Returns:
    float: The entropy value of the label proportions.
    """
    perc = []
    for i in label_totals.keys():
        if i in g.nodes[node]['label_counts']:
            perc.append(g.nodes[node]['label_counts'][i] / label_totals[i])
        else:
            perc.append(0)
    return entropy(perc)

def get_contingency_matrix(label_totals, label_counts, label):
    """
    Compute the contingency matrix for a given label.

    Parameters:
    label_totals (dict): A dictionary where keys are labels and values are the total counts of each label.
    label_counts (dict): A dictionary where keys are labels and values are the counts of each label within a subset.
    label (int or str): The specific label for which the contingency matrix is to be computed.

    Returns:
    tuple: A tuple containing four elements (TP, TN, FP, FN):
        TP (int): True Positives - count of correctly identified positive instances.
        TN (int): True Negatives - count of correctly identified negative instances.
        FP (int): False Positives - count of incorrectly identified positive instances.
        FN (int): False Negatives - count of incorrectly identified negative instances.
    """
    if label in label_counts:
        TP = label_counts.get(label, 0)
    else:
        TP = 0
    FN = label_totals[label] - TP
    FP = sum(label_counts.values()) - TP
    TN = sum(label_totals.values()) - TP - FP - FN
    return TP, TN, FP, FN

def agg_chi2(g, node, label_totals):
    """
    Aggregate the chi-squared statistic for a node in a graph based on label counts.

    Parameters:
    g (networkx.Graph): The graph containing the nodes with label counts.
    node (node): The specific node in the graph for which the chi-squared statistic is to be computed.
    label_totals (dict): A dictionary where keys are labels and values are the total counts of each label.

    Returns:
    float: The chi-squared statistic for the given node.
    """
    TP, TN, FP, FN = get_contingency_matrix(label_totals, g.nodes[node]['label_counts'], 1)
    contingency_table = [[TP, FP], [FN, TN]]
    try:
        chi2, _, _, _ = chi2_contingency(contingency_table)
    except:
        chi2 = 0
    return chi2

def agg_odds_ratio(g, node, label_totals):
    """
    Aggregate the chi-squared statistic for a node in a graph based on label counts.

    Parameters:
    g (networkx.Graph): The graph containing the nodes with label counts.
    node (node): The specific node in the graph for which the chi-squared statistic is to be computed.
    label_totals (dict): A dictionary where keys are labels and values are the total counts of each label.

    Returns:
    float: The chi-squared statistic for the given node.
    """
    TP, TN, FP, FN = get_contingency_matrix(label_totals, g.nodes[node]['label_counts'], 1)

    if FP == 0 or FN == 0:
        # Prevent division by zero
        odds_ratio = -1
    else:
        # Calculate odds ratio
        odds_ratio = (TP * TN) / (FP * FN)

    # If odds ratio is less than 1, invert it
    if odds_ratio < 1 and odds_ratio != 0:
        odds_ratio = 1 / odds_ratio

    return odds_ratio