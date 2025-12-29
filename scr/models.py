import numpy as np
from collections import defaultdict




def compute_common_neighbors(graph, u, v):
    """
    Compute Common Neighbors score for a pair of nodes.

    CN(u, v) = |N(u) ∩ N(v)|

    Args:
        graph: NetworkX graph
        u, v: Node identifiers

    Returns:
        int: Number of common neighbors
    """
    if u not in graph or v not in graph:
        return 0
    neighbors_u = set(graph.neighbors(u))
    neighbors_v = set(graph.neighbors(v))
    return len(neighbors_u & neighbors_v)


def compute_jaccard(graph, u, v):
    """
    Compute Jaccard Coefficient for a pair of nodes.

    J(u, v) = |N(u) ∩ N(v)| / |N(u) ∪ N(v)|

    Args:
        graph: NetworkX graph
        u, v: Node identifiers

    Returns:
        float: Jaccard coefficient (0 to 1)
    """
    if u not in graph or v not in graph:
        return 0.0
    neighbors_u = set(graph.neighbors(u))
    neighbors_v = set(graph.neighbors(v))
    union = neighbors_u | neighbors_v
    if len(union) == 0:
        return 0.0
    return len(neighbors_u & neighbors_v) / len(union)


def compute_preferential_attachment(graph, u, v):
    """
    Compute Preferential Attachment score for a pair of nodes.

    PA(u, v) = |N(u)| × |N(v)|

    Args:
        graph: NetworkX graph
        u, v: Node identifiers

    Returns:
        int: Product of node degrees
    """
    if u not in graph or v not in graph:
        return 0
    deg_u = graph.degree(u)
    deg_v = graph.degree(v)
    return deg_u * deg_v


def compute_degree_sum(graph, u, v):
    """
    Compute sum of degrees for a pair of nodes.

    DS(u, v) = deg(u) + deg(v)

    Args:
        graph: NetworkX graph
        u, v: Node identifiers

    Returns:
        int: Sum of node degrees
    """
    if u not in graph or v not in graph:
        return 0
    return graph.degree(u) + graph.degree(v)


def compute_degree_diff(graph, u, v):
    """
    Compute absolute difference of degrees for a pair of nodes.

    DD(u, v) = |deg(u) - deg(v)|

    Args:
        graph: NetworkX graph
        u, v: Node identifiers

    Returns:
        int: Absolute difference of node degrees
    """
    if u not in graph or v not in graph:
        return 0
    return abs(graph.degree(u) - graph.degree(v))


def compute_heuristics(graph, test_edges):
    """
    Compute all heuristic scores for test edges.

    Args:
        graph: Training graph (NetworkX)
        test_edges: List of (u, v) tuples to predict

    Returns:
        dict: {method_name: [(edge, score), ...]}
    """
    heuristic_scores = {
        'common_neighbors': [],
        'jaccard': [],
        'preferential_attachment': []
    }

    print(f"   Computing heuristics for {len(test_edges)} test edges...")

    for i, (u, v) in enumerate(test_edges):
        cn = compute_common_neighbors(graph, u, v)
        jc = compute_jaccard(graph, u, v)
        pa = compute_preferential_attachment(graph, u, v)

        heuristic_scores['common_neighbors'].append(((u, v), cn))
        heuristic_scores['jaccard'].append(((u, v), jc))
        heuristic_scores['preferential_attachment'].append(((u, v), pa))

        if (i + 1) % 500 == 0:
            print(f"      Processed {i + 1}/{len(test_edges)} edges...")

    # Print statistics
    for method, scores in heuristic_scores.items():
        values = [s for _, s in scores]
        print(f"   {method}:")
        print(f"      Min: {min(values):.4f}, Max: {max(values):.4f}, "
              f"Avg: {np.mean(values):.4f}, Non-zero: {sum(1 for v in values if v > 0)/len(values)*100:.1f}%")

    return heuristic_scores


# FEATURE EXTRACTION

def extract_features(graph, u, v):
    """
    Extract all features for a node pair.

    Features:
    1. Common Neighbors
    2. Jaccard Coefficient
    3. Preferential Attachment
    4. Degree Sum
    5. Degree Difference

    Args:
        graph: NetworkX graph
        u, v: Node identifiers

    Returns:
        list: Feature vector [cn, jc, pa, deg_sum, deg_diff]
    """
    cn = compute_common_neighbors(graph, u, v)
    jc = compute_jaccard(graph, u, v)
    pa = compute_preferential_attachment(graph, u, v)
    deg_sum = compute_degree_sum(graph, u, v)
    deg_diff = compute_degree_diff(graph, u, v)

    return [cn, jc, pa, deg_sum, deg_diff]


# NEGATIVE SAMPLING

def generate_negative_samples(graph, num_samples, existing_edges=None):
    """
    Generate negative samples (non-edges) for training.
    Uses degree-biased sampling for more realistic negatives.

    Args:
        graph: NetworkX graph
        num_samples: Number of negative samples to generate
        existing_edges: Set of edges to exclude (optional)

    Returns:
        list: List of (u, v) tuples representing non-edges
    """
    import random

    nodes = list(graph.nodes())
    if len(nodes) < 2:
        return []

    if existing_edges is None:
        existing_edges = set(graph.edges())

    # Make sure we have both directions for undirected graph
    existing_edges_both = set()
    for u, v in existing_edges:
        existing_edges_both.add((u, v))
        existing_edges_both.add((v, u))

    # Compute node degrees for degree-biased sampling
    degrees = dict(graph.degree())
    total_degree = sum(degrees.values())

    # Create probability distribution based on degree
    # Higher degree nodes are more likely to be sampled
    node_probs = [degrees[n] / total_degree for n in nodes]

    negative_samples = []
    attempts = 0
    max_attempts = num_samples * 20

    while len(negative_samples) < num_samples and attempts < max_attempts:
        # Sample nodes with probability proportional to degree
        u, v = np.random.choice(nodes, size=2, replace=False, p=node_probs)

        if (u, v) not in existing_edges_both:
            negative_samples.append((u, v))
            existing_edges_both.add((u, v))
            existing_edges_both.add((v, u))

        attempts += 1

    return negative_samples


# ML MODELS

def train_ml_models(graph, test_edges, heuristic_scores):
    """
    Train ML models for link prediction.

    Uses extended feature set:
    - Common Neighbors
    - Jaccard Coefficient
    - Preferential Attachment
    - Degree Sum
    - Degree Difference

    Args:
        graph: Training graph
        test_edges: Test edges (positive samples)
        heuristic_scores: Pre-computed heuristic scores

    Returns:
        tuple: (ml_predictions dict, lr_model, rf_model, scaler)
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler

    print("   Preparing training data...")

    # Get positive edges from graph
    positive_edges = list(graph.edges())

    # Generate negative samples (same count as positive)
    # Using degree-biased sampling for more realistic negatives
    negative_edges = generate_negative_samples(
        graph,
        num_samples=len(positive_edges),
        existing_edges=set(positive_edges)
    )

    print(f"      Positive training edges: {len(positive_edges)}")
    print(f"      Negative training edges: {len(negative_edges)}")

    # Build feature matrix with extended features
    X_train = []
    y_train = []

    # Positive samples
    for u, v in positive_edges:
        features = extract_features(graph, u, v)
        X_train.append(features)
        y_train.append(1)

    # Negative samples
    for u, v in negative_edges:
        features = extract_features(graph, u, v)
        X_train.append(features)
        y_train.append(0)

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    print(f"      Feature matrix shape: {X_train.shape}")
    print(f"      Features: [CN, Jaccard, PA, DegreeSum, DegreeDiff]")

    # Build test feature matrix
    X_test = []
    for u, v in test_edges:
        features = extract_features(graph, u, v)
        X_test.append(features)

    X_test = np.array(X_test)

    # Scale ALL features for both models
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    ml_predictions = {}

    # Train Logistic Regression with tuned hyperparameters
    print("   Training Logistic Regression...")
    lr_model = LogisticRegression(
        random_state=42,
        max_iter=2000,
        C=0.5,  # Stronger regularization
        solver='lbfgs',
        class_weight='balanced'  # Handle class imbalance
    )
    lr_model.fit(X_train_scaled, y_train)
    lr_probs = lr_model.predict_proba(X_test_scaled)[:, 1]
    ml_predictions['logistic_regression'] = [
        (test_edges[i], lr_probs[i]) for i in range(len(test_edges))
    ]
    print(f"      LR training accuracy: {lr_model.score(X_train_scaled, y_train):.3f}")

    # Train Random Forest with tuned hyperparameters
    print("   Training Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=200,  # More trees
        max_depth=15,  # Deeper trees
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'  # Handle class imbalance
    )
    rf_model.fit(X_train_scaled, y_train)  # Use scaled features for consistency
    rf_probs = rf_model.predict_proba(X_test_scaled)[:, 1]
    ml_predictions['random_forest'] = [
        (test_edges[i], rf_probs[i]) for i in range(len(test_edges))
    ]
    print(f"      RF training accuracy: {rf_model.score(X_train_scaled, y_train):.3f}")

    # Feature importance from Random Forest
    feature_names = ['CN', 'Jaccard', 'PA', 'DegSum', 'DegDiff']
    importances = rf_model.feature_importances_
    print(f"      RF Feature Importances:")
    for name, imp in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True):
        print(f"         {name}: {imp:.3f}")

    # Voting ensemble (average of normalized heuristics + ML)
    print("   Creating Voting Ensemble...")

    # Normalize heuristics for voting
    cn_scores = np.array([s for _, s in heuristic_scores['common_neighbors']])
    jc_scores = np.array([s for _, s in heuristic_scores['jaccard']])
    pa_scores = np.array([s for _, s in heuristic_scores['preferential_attachment']])

    # Min-max normalization
    cn_norm = (cn_scores - cn_scores.min()) / (cn_scores.max() - cn_scores.min() + 1e-10)
    pa_norm = (pa_scores - pa_scores.min()) / (pa_scores.max() - pa_scores.min() + 1e-10)
    # Jaccard is already in [0, 1]

    # Voting: weighted average of all 5 scores
    # Give more weight to ML models
    voting_scores = (cn_norm * 0.15 + jc_scores * 0.10 + pa_norm * 0.15 +
                    lr_probs * 0.30 + rf_probs * 0.30)

    ml_predictions['voting'] = [
        (test_edges[i], voting_scores[i]) for i in range(len(test_edges))
    ]

    print("   ✓ All ML models trained successfully")

    return ml_predictions, lr_model, rf_model, scaler
