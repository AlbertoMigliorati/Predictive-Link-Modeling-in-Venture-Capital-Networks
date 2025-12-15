import pandas as pd
import networkx as nx
from pathlib import Path
from collections import defaultdict
import re


def normalize_name(name):
    """
    Normalize entity names for deduplication.

    Args:
        name: Raw entity name

    Returns:
        str: Normalized name or None if invalid
    """
    if pd.isna(name) or name == '':
        return None

    # Convert to lowercase and strip whitespace
    name = str(name).lower().strip()

    # Remove common suffixes
    suffixes = [' inc', ' llc', ' ltd', ' corp', ' corporation',
                ' venture', ' ventures', ' capital', ' partners']
    for suffix in suffixes:
        if name.endswith(suffix):
            name = name[:-len(suffix)].strip()

    # Remove special characters and extra spaces
    name = re.sub(r'[^\w\s]', '', name)
    name = re.sub(r'\s+', ' ', name)

    return name if name else None


def parse_investors(investor_str):
    """
    Parse comma-separated investor names.

    Args:
        investor_str: Comma-separated string of investor names

    Returns:
        list: List of normalized investor names
    """
    if pd.isna(investor_str) or investor_str == '':
        return []

    investors = [inv.strip() for inv in str(investor_str).split(',')]
    investors = [normalize_name(inv) for inv in investors if inv]
    return [inv for inv in investors if inv is not None]


def load_and_preprocess_data(data_path, cutoff_year=2023, task='coinvestor'):
    """
    Load and preprocess VC investment data.

    Parameters:
    -----------
    data_path : str
        Path to directory containing CSV file
    cutoff_year : int
        Year to split train/test (train <= cutoff_year, test > cutoff_year)
    task : str
        'coinvestor' or 'investor_startup'

    Returns:
    --------
    dict with keys:
        - train_graph: networkx.Graph with edges up to cutoff_year
        - test_edges: list of (node1, node2) tuples for testing
        - investor_names: dict mapping normalized names to original names
    """

    # Find CSV file in data_path
    data_files = list(Path(data_path).glob('*.csv'))
    if not data_files:
        raise FileNotFoundError(f"No CSV file found in {data_path}")

    csv_file = data_files[0]
    print(f"   Loading data from: {csv_file}")

    # Load data with semicolon separator
    df = pd.read_csv(csv_file, sep=';')

    # Clean column names
    df.columns = df.columns.str.strip().str.lower()

    # Expected columns mapping
    col_mapping = {
        'startup_name': 'startup_name',
        'startup name': 'startup_name',
        'industries': 'industries',
        'location': 'location',
        'investor name': 'investor_name',
        'investor_name': 'investor_name',
        'lead investor': 'lead_investor',
        'lead_investor': 'lead_investor',
        'number of investors': 'num_investors',
        'number_of_investors': 'num_investors',
        'funding type': 'funding_type',
        'funding_type': 'funding_type',
        'month': 'month',
        'day': 'day',
        'year': 'year'
    }

    # Rename columns
    df = df.rename(columns=col_mapping)

    # Ensure required columns exist
    required = ['startup_name', 'investor_name', 'year']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}. Available columns: {list(df.columns)}")

    # Remove rows with missing critical data
    df = df.dropna(subset=['startup_name', 'investor_name', 'year'])

    # Convert year to int
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df = df.dropna(subset=['year'])
    df['year'] = df['year'].astype(int)

    # Normalize names
    df['startup_normalized'] = df['startup_name'].apply(normalize_name)
    df = df.dropna(subset=['startup_normalized'])

    # Parse investors
    df['investors_list'] = df['investor_name'].apply(parse_investors)
    df = df[df['investors_list'].apply(len) > 0]

    print(f"   Loaded {len(df)} investment rounds")
    print(f"   Year range: {df['year'].min()} - {df['year'].max()}")

    # Split train/test
    train_df = df[df['year'] <= cutoff_year].copy()
    test_df = df[df['year'] > cutoff_year].copy()

    print(f"   Train: {len(train_df)} rounds (year <= {cutoff_year})")
    print(f"   Test:  {len(test_df)} rounds (year > {cutoff_year})")

    # Create investor name mapping
    investor_names = {}
    for investors_str in df['investor_name']:
        investors = parse_investors(investors_str)
        for inv in investors:
            if inv not in investor_names:
                # Find original name
                orig = [i.strip() for i in str(investors_str).split(',')
                       if normalize_name(i.strip()) == inv]
                if orig:
                    investor_names[inv] = orig[0]

    if task == 'coinvestor':
        # Build co-investor graph
        train_graph = build_coinvestor_graph(train_df)
        test_edges = extract_coinvestor_test_edges(test_df, train_graph)
    else:
        # Build investor-startup bipartite graph
        train_graph = build_bipartite_graph(train_df)
        test_edges = extract_bipartite_test_edges(test_df, train_graph)

    # Return as dictionary (required by main.py)
    return {
        'train_graph': train_graph,
        'test_edges': test_edges,
        'investor_names': investor_names
    }


def build_coinvestor_graph(df):
    """
    Build co-investor projection graph.
    Two investors are connected if they co-invested in same startup.

    Args:
        df: DataFrame with investors_list column

    Returns:
        networkx.Graph: Co-investor graph
    """
    G = nx.Graph()

    for _, row in df.iterrows():
        investors = row['investors_list']

        # Add all investors as nodes
        for inv in investors:
            if not G.has_node(inv):
                G.add_node(inv)

        # Add edges between co-investors
        for i, inv1 in enumerate(investors):
            for inv2 in investors[i+1:]:
                if G.has_edge(inv1, inv2):
                    G[inv1][inv2]['weight'] += 1
                else:
                    G.add_edge(inv1, inv2, weight=1)

    print(f"   Co-investor graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def build_bipartite_graph(df):
    """
    Build bipartite investor-startup graph.

    Args:
        df: DataFrame with startup_normalized and investors_list columns

    Returns:
        networkx.Graph: Bipartite graph
    """
    G = nx.Graph()

    for _, row in df.iterrows():
        startup = row['startup_normalized']
        investors = row['investors_list']

        # Add nodes with bipartite labels
        G.add_node(startup, bipartite=0)  # 0 for startups

        for inv in investors:
            G.add_node(inv, bipartite=1)  # 1 for investors

            # Add edge
            if G.has_edge(inv, startup):
                G[inv][startup]['weight'] += 1
            else:
                G.add_edge(inv, startup, weight=1)

    print(f"   Bipartite graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def extract_coinvestor_test_edges(df, train_graph):
    """
    Extract new co-investor edges from test period.
    Only includes edges where both investors exist in training graph
    and the edge is NEW (not present in training).

    Args:
        df: Test DataFrame
        train_graph: Training graph

    Returns:
        list: List of (investor1, investor2) tuples
    """
    test_edges = set()

    for _, row in df.iterrows():
        investors = row['investors_list']

        # Check all pairs
        for i, inv1 in enumerate(investors):
            for inv2 in investors[i+1:]:
                # Only include if both investors exist in training
                if inv1 in train_graph and inv2 in train_graph:
                    # Check if this is a NEW edge (not in training)
                    if not train_graph.has_edge(inv1, inv2):
                        edge = tuple(sorted([inv1, inv2]))
                        test_edges.add(edge)

    print(f"   Test edges extracted: {len(test_edges)}")
    return list(test_edges)


def extract_bipartite_test_edges(df, train_graph):
    """
    Extract new investor-startup edges from test period.
    Only includes edges where investor exists in training graph.

    Args:
        df: Test DataFrame
        train_graph: Training graph

    Returns:
        list: List of (investor, startup) tuples
    """
    test_edges = set()

    for _, row in df.iterrows():
        startup = row['startup_normalized']
        investors = row['investors_list']

        for inv in investors:
            # Only include if investor exists in training
            if inv in train_graph:
                # Check if this is a NEW edge
                if not train_graph.has_edge(inv, startup):
                    test_edges.add((inv, startup))

    print(f"   Test edges extracted: {len(test_edges)}")
    return list(test_edges)














