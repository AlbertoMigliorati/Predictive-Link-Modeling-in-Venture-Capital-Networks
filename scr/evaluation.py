import numpy as np
from collections import defaultdict


def precision_at_k(predictions, positive_edges, k):
    """
    Compute Precision@k.

    Args:
        predictions: List of (edge, score) tuples
        positive_edges: Set of actual positive edges
        k: Number of top predictions to consider

    Returns:
        float: Precision@k score
    """
    # Sort by score descending
    sorted_preds = sorted(predictions, key=lambda x: x[1], reverse=True)

    # Get top-k predictions
    top_k = sorted_preds[:k]

    # Count true positives
    tp = 0
    for edge, score in top_k:
        # Check both directions for undirected graphs
        if edge in positive_edges or (edge[1], edge[0]) in positive_edges:
            tp += 1

    return tp / k if k > 0 else 0.0


def compute_pr_auc(predictions, positive_edges):
    """
    Compute Precision-Recall Area Under Curve.

    Args:
        predictions: List of (edge, score) tuples
        positive_edges: Set of actual positive edges

    Returns:
        float: PR-AUC score
    """
    from sklearn.metrics import precision_recall_curve, auc

    # Create labels (1 for positive, 0 for negative)
    y_true = []
    y_scores = []

    for edge, score in predictions:
        # Check both directions
        is_positive = edge in positive_edges or (edge[1], edge[0]) in positive_edges
        y_true.append(1 if is_positive else 0)
        y_scores.append(score)

    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    # Handle edge case where all predictions are same class
    if len(np.unique(y_true)) < 2:
        return 0.0

    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    return auc(recall, precision)


def generate_hard_negative_samples(graph, test_edges, ratio=5):
    """
    Generate HARD negative samples for realistic evaluation.

    Hard negatives are sampled from nodes with similar degree distribution
    to the positive test edges, making the task more challenging and realistic.

    Args:
        graph: Training graph
        test_edges: Positive test edges
        ratio: Ratio of negative to positive samples

    Returns:
        list: Negative test edges
    """
    import random

    # Compute degree statistics from positive test edges
    positive_degrees = []
    for u, v in test_edges:
        if u in graph and v in graph:
            positive_degrees.append(graph.degree(u))
            positive_degrees.append(graph.degree(v))

    if not positive_degrees:
        # Fallback to random sampling if no valid test edges
        return generate_random_negative_samples(graph, test_edges, ratio)

    # Get degree range of positive edges
    min_deg = max(1, np.percentile(positive_degrees, 10))
    max_deg = np.percentile(positive_degrees, 90)

    print(f"      Positive edges degree range: [{min_deg:.0f}, {max_deg:.0f}]")

    # Get nodes with similar degrees for hard negatives
    candidate_nodes = [n for n in graph.nodes()
                       if min_deg <= graph.degree(n) <= max_deg]

    print(f"      Candidate nodes for hard negatives: {len(candidate_nodes)}")

    # If not enough candidates, include more nodes
    if len(candidate_nodes) < 100:
        candidate_nodes = [n for n in graph.nodes() if graph.degree(n) >= min_deg / 2]

    existing_edges = set(graph.edges())
    test_edges_set = set(test_edges)

    # Add both directions
    all_edges = set()
    for u, v in existing_edges:
        all_edges.add((u, v))
        all_edges.add((v, u))
    for u, v in test_edges_set:
        all_edges.add((u, v))
        all_edges.add((v, u))

    num_negative = len(test_edges) * ratio
    negative_samples = []
    attempts = 0
    max_attempts = num_negative * 50

    while len(negative_samples) < num_negative and attempts < max_attempts:
        # Sample from candidate nodes (similar degree to positives)
        if len(candidate_nodes) >= 2:
            u = random.choice(candidate_nodes)
            v = random.choice(candidate_nodes)
        else:
            nodes = list(graph.nodes())
            u = random.choice(nodes)
            v = random.choice(nodes)

        if u != v and (u, v) not in all_edges:
            negative_samples.append((u, v))
            all_edges.add((u, v))
            all_edges.add((v, u))

        attempts += 1

    return negative_samples


def generate_random_negative_samples(graph, test_edges, ratio=5):
    """
    Generate random negative samples (fallback method).

    Args:
        graph: Training graph
        test_edges: Positive test edges
        ratio: Ratio of negative to positive samples

    Returns:
        list: Negative test edges
    """
    import random

    nodes = list(graph.nodes())
    existing_edges = set(graph.edges())
    test_edges_set = set(test_edges)

    # Add both directions
    all_edges = set()
    for u, v in existing_edges:
        all_edges.add((u, v))
        all_edges.add((v, u))
    for u, v in test_edges_set:
        all_edges.add((u, v))
        all_edges.add((v, u))

    num_negative = len(test_edges) * ratio
    negative_samples = []
    attempts = 0
    max_attempts = num_negative * 20

    while len(negative_samples) < num_negative and attempts < max_attempts:
        u = random.choice(nodes)
        v = random.choice(nodes)

        if u != v and (u, v) not in all_edges:
            negative_samples.append((u, v))
            all_edges.add((u, v))
            all_edges.add((v, u))

        attempts += 1

    return negative_samples


def analyze_test_edge_difficulty(test_edges, graph):
    """
    Analyze the difficulty of test edges based on node degrees.

    Categories:
    - Easy: Both nodes have degree > 10
    - Medium: Both nodes have degree 5-10
    - Hard: At least one node has degree < 5
    - Very Hard: At least one node is new (not in training graph)

    Args:
        test_edges: List of test edges
        graph: Training graph

    Returns:
        dict: Counts and percentages for each difficulty category
    """
    categories = defaultdict(list)

    for u, v in test_edges:
        u_in_graph = u in graph
        v_in_graph = v in graph

        if not u_in_graph or not v_in_graph:
            categories['very_hard'].append((u, v))
            continue

        deg_u = graph.degree(u)
        deg_v = graph.degree(v)
        min_deg = min(deg_u, deg_v)

        if min_deg > 10:
            categories['easy'].append((u, v))
        elif min_deg >= 5:
            categories['medium'].append((u, v))
        else:
            categories['hard'].append((u, v))

    total = len(test_edges)
    analysis = {}

    for cat in ['easy', 'medium', 'hard', 'very_hard']:
        count = len(categories[cat])
        analysis[cat] = {
            'count': count,
            'percentage': count / total * 100 if total > 0 else 0
        }

    return analysis


# Import heuristic and feature functions
from scr.models import (
    compute_common_neighbors,
    compute_jaccard,
    compute_preferential_attachment,
    extract_features
)


def evaluate_predictions(heuristic_scores, ml_predictions, test_edges, train_graph, k_values=[50, 100], ml_models=None):
    """
    Evaluate all prediction methods.

    Args:
        heuristic_scores: Dict of heuristic scores
        ml_predictions: Dict of ML model predictions
        test_edges: List of positive test edges
        train_graph: Training graph
        k_values: List of k values for Precision@k
        ml_models: Dict containing 'lr', 'rf', 'scaler' trained models

    Returns:
        dict: Evaluation results
    """
    print("   Analyzing test edge difficulty...")
    difficulty = analyze_test_edge_difficulty(test_edges, train_graph)

    for cat, info in difficulty.items():
        print(f"      {cat.replace('_', ' ').title()}: {info['count']} ({info['percentage']:.1f}%)")

    # Generate HARD negative test samples for realistic evaluation
    print(f"\n   Generating hard negative test samples...")
    negative_test_edges = generate_hard_negative_samples(
        train_graph,
        test_edges,
        ratio=5
    )
    print(f"      Generated {len(negative_test_edges)} hard negative samples (ratio 5:1)")

    # Positive edges set
    positive_edges = set(test_edges)

    # Compute heuristic scores for negative samples
    print(f"   Computing heuristic scores for negative samples...")

    for method_name, scores in heuristic_scores.items():
        negative_scores = []

        for edge in negative_test_edges:
            u, v = edge
            if method_name == 'common_neighbors':
                score = compute_common_neighbors(train_graph, u, v)
            elif method_name == 'jaccard':
                score = compute_jaccard(train_graph, u, v)
            elif method_name == 'preferential_attachment':
                score = compute_preferential_attachment(train_graph, u, v)
            else:
                score = 0

            negative_scores.append((edge, score))

        # Combine with existing positive scores
        heuristic_scores[method_name] = scores + negative_scores
        print(f"      {method_name}: ✓ Done")

    # Compute ML model scores for negative samples using REAL trained models
    print(f"   Computing ML scores for {len(negative_test_edges)} negative samples...")

    # Build feature matrix for negative samples (extended features)
    X_negative = []
    for edge in negative_test_edges:
        u, v = edge
        features = extract_features(train_graph, u, v)
        X_negative.append(features)

    X_negative = np.array(X_negative)

    if ml_models is not None:
        # Use real trained models
        lr_model = ml_models['lr']
        rf_model = ml_models['rf']
        scaler = ml_models['scaler']

        # Scale features
        X_negative_scaled = scaler.transform(X_negative)

        # Get predictions from real models
        lr_probs_neg = lr_model.predict_proba(X_negative_scaled)[:, 1]
        rf_probs_neg = rf_model.predict_proba(X_negative_scaled)[:, 1]

        # Logistic Regression scores for negative samples
        lr_negative_scores = [(negative_test_edges[i], lr_probs_neg[i])
                             for i in range(len(negative_test_edges))]
        ml_predictions['logistic_regression'] = ml_predictions['logistic_regression'] + lr_negative_scores
        print(f"      logistic_regression: ✓ Done")

        # Random Forest scores for negative samples
        rf_negative_scores = [(negative_test_edges[i], rf_probs_neg[i])
                             for i in range(len(negative_test_edges))]
        ml_predictions['random_forest'] = ml_predictions['random_forest'] + rf_negative_scores
        print(f"      random_forest: ✓ Done")

        # Voting: weighted average of heuristics + ML probabilities
        voting_negative_scores = []

        # Get heuristic scores for normalization
        all_cn = [s for _, s in heuristic_scores['common_neighbors']]
        all_pa = [s for _, s in heuristic_scores['preferential_attachment']]
        cn_min, cn_max = min(all_cn), max(all_cn)
        pa_min, pa_max = min(all_pa), max(all_pa)

        for i, edge in enumerate(negative_test_edges):
            cn = X_negative[i][0]  # CN is first feature
            jc = X_negative[i][1]  # Jaccard is second feature
            pa = X_negative[i][2]  # PA is third feature

            # Normalize heuristics
            cn_norm = (cn - cn_min) / (cn_max - cn_min + 1e-10)
            pa_norm = (pa - pa_min) / (pa_max - pa_min + 1e-10)

            # Weighted average (same weights as in training)
            vote_score = (cn_norm * 0.15 + jc * 0.10 + pa_norm * 0.15 +
                         lr_probs_neg[i] * 0.30 + rf_probs_neg[i] * 0.30)
            voting_negative_scores.append((edge, vote_score))

        ml_predictions['voting'] = ml_predictions['voting'] + voting_negative_scores
        print(f"      voting: ✓ Done")

    print(f"   ✓ Computed real scores for all {len(negative_test_edges)} negative samples")

    # Evaluate all methods
    print("\n   Evaluating all methods...")
    results = {
        'metrics': {},
        'difficulty_analysis': difficulty,
        'predictions': {}
    }

    all_methods = {}
    all_methods.update(heuristic_scores)
    all_methods.update(ml_predictions)

    # Add random baseline
    import random
    random.seed(42)
    all_edges = list(positive_edges) + negative_test_edges
    random_scores = [(edge, random.random()) for edge in all_edges]
    all_methods['random_baseline'] = random_scores

    for method_name, predictions in all_methods.items():
        metrics = {}

        # Precision@k
        for k in k_values:
            p_at_k = precision_at_k(predictions, positive_edges, k)
            metrics[f'precision@{k}'] = p_at_k

        # PR-AUC
        pr_auc = compute_pr_auc(predictions, positive_edges)
        metrics['pr_auc'] = pr_auc

        results['metrics'][method_name] = metrics
        results['predictions'][method_name] = predictions

        print(f"      {method_name}: P@100={metrics['precision@100']:.3f}, PR-AUC={pr_auc:.3f}")

    return results


def visualize_results(results, output_path, task):
    """
    Generate visualization plots for results.

    Args:
        results: Evaluation results dict
        output_path: Directory to save plots
        task: Task name for file naming
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os

    metrics = results['metrics']

    # Sort methods by PR-AUC for consistent ordering
    methods_sorted = sorted(metrics.keys(), key=lambda x: metrics[x].get('pr_auc', 0), reverse=True)

    # Color palette
    colors = sns.color_palette("husl", len(methods_sorted))

    # 1. Precision@100 bar chart
    fig, ax = plt.subplots(figsize=(10, 6))

    p100_values = [metrics[m].get('precision@100', 0) for m in methods_sorted]
    bars = ax.bar(range(len(methods_sorted)), p100_values, color=colors)

    ax.set_xticks(range(len(methods_sorted)))
    ax.set_xticklabels([m.replace('_', ' ').title() for m in methods_sorted], rotation=45, ha='right')
    ax.set_ylabel('Precision@100')
    ax.set_title(f'Link Prediction: Precision@100 ({task})')
    ax.set_ylim(0, 1.1)

    # Add value labels
    for bar, val in zip(bars, p100_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'precision_at_100.png'), dpi=150)
    plt.close()

    # 2. PR-AUC bar chart
    fig, ax = plt.subplots(figsize=(10, 6))

    prauc_values = [metrics[m].get('pr_auc', 0) for m in methods_sorted]
    bars = ax.bar(range(len(methods_sorted)), prauc_values, color=colors)

    ax.set_xticks(range(len(methods_sorted)))
    ax.set_xticklabels([m.replace('_', ' ').title() for m in methods_sorted], rotation=45, ha='right')
    ax.set_ylabel('PR-AUC')
    ax.set_title(f'Link Prediction: PR-AUC ({task})')
    ax.set_ylim(0, 1.1)

    # Add value labels
    for bar, val in zip(bars, prauc_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'pr_auc.png'), dpi=150)
    plt.close()

    # 3. Metrics heatmap
    fig, ax = plt.subplots(figsize=(12, 8))

    metric_names = ['precision@50', 'precision@100', 'pr_auc']
    data = []
    for m in methods_sorted:
        row = [metrics[m].get(metric, 0) for metric in metric_names]
        data.append(row)

    data = np.array(data)

    sns.heatmap(data, annot=True, fmt='.3f', cmap='YlGnBu',
                xticklabels=['P@50', 'P@100', 'PR-AUC'],
                yticklabels=[m.replace('_', ' ').title() for m in methods_sorted],
                ax=ax)

    ax.set_title(f'Link Prediction Metrics Heatmap ({task})')

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'metrics_heatmap.png'), dpi=150)
    plt.close()

    # 4. Save top predictions to text file
    best_method = methods_sorted[0]
    best_predictions = results['predictions'][best_method]
    sorted_preds = sorted(best_predictions, key=lambda x: x[1], reverse=True)[:50]

    with open(os.path.join(output_path, 'top_50_predictions.txt'), 'w') as f:
        f.write(f"Top 50 Predictions ({best_method})\n")
        f.write("=" * 60 + "\n\n")

        for i, (edge, score) in enumerate(sorted_preds, 1):
            f.write(f"{i:2d}. {edge[0]} <-> {edge[1]}\n")
            f.write(f"    Score: {score:.4f}\n\n")

    print(f"   ✓ Saved visualizations to {output_path}")
    print(f"      - precision_at_100.png")
    print(f"      - pr_auc.png")
    print(f"      - metrics_heatmap.png")
    print(f"      - top_50_predictions.txt")
