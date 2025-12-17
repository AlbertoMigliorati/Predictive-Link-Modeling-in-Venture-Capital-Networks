import argparse
import os
import sys
from datetime import datetime
import random
import numpy as np

# Set global seeds for reproducibility
random.seed(42)
np.random.seed(42)

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scr import (
    load_and_preprocess_data,
    compute_heuristics,
    train_ml_models,
    evaluate_predictions,
    visualize_results
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='VC Link Prediction Pipeline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--task',
        type=str,
        default='coinvestor',
        choices=['coinvestor', 'investor_startup'],
        help='Type of link prediction task'
    )

    parser.add_argument(
        '--cutoff_year',
        type=int,
        default=2023,
        help='Year to split train/test data'
    )

    parser.add_argument(
        '--data_path',
        type=str,
        default='data/raw/',
        help='Path to data directory'
    )

    parser.add_argument(
        '--output_path',
        type=str,
        default='results/',
        help='Path to output directory'
    )

    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_args()

    print("=" * 60)
    print("VC LINK PREDICTION PIPELINE")
    print("=" * 60)
    print(f"Task: {args.task}")
    print(f"Cutoff Year: {args.cutoff_year}")
    print(f"Data Path: {args.data_path}")
    print(f"Output Path: {args.output_path}")
    print("=" * 60)

    # Create output directory if it doesn't exist
    os.makedirs(args.output_path, exist_ok=True)

    # Step 1: Load and preprocess data
    print("\n[Step 1/5] Loading and preprocessing data...")
    data = load_and_preprocess_data(
        data_path=args.data_path,
        cutoff_year=args.cutoff_year,
        task=args.task
    )

    train_graph = data['train_graph']
    test_edges = data['test_edges']

    print(f"   Train graph: {train_graph.number_of_nodes()} nodes, {train_graph.number_of_edges()} edges")
    print(f"   Test edges to predict: {len(test_edges)}")

    if len(test_edges) == 0:
        print("ERROR: No test edges found. Check cutoff_year parameter.")
        sys.exit(1)

    # Step 2: Compute heuristics
    print("\n[Step 2/5] Computing graph-based heuristics...")
    heuristic_scores = compute_heuristics(
        graph=train_graph,
        test_edges=test_edges
    )

    for method, scores in heuristic_scores.items():
        non_zero = sum(1 for _, s in scores if s > 0)
        print(f"   {method}: {non_zero}/{len(scores)} non-zero scores")

    # Step 3: Train ML models
    print("\n[Step 3/5] Training ML models...")
    ml_predictions, lr_model, rf_model, scaler = train_ml_models(
        graph=train_graph,
        test_edges=test_edges,
        heuristic_scores=heuristic_scores
    )

    for method, preds in ml_predictions.items():
        print(f"   {method}: {len(preds)} predictions")

    # Step 4: Evaluate predictions
    print("\n[Step 4/5] Evaluating predictions...")
    results = evaluate_predictions(
        heuristic_scores=heuristic_scores,
        ml_predictions=ml_predictions,
        test_edges=test_edges,
        train_graph=train_graph,
        k_values=[50, 100],
        ml_models={'lr': lr_model, 'rf': rf_model, 'scaler': scaler}
    )

    # Print results summary
    print("\n" + "-" * 40)
    print("RESULTS SUMMARY")
    print("-" * 40)
    print(f"{'Method':<25} {'P@50':<10} {'P@100':<10} {'PR-AUC':<10}")
    print("-" * 40)

    for method, metrics in sorted(results['metrics'].items(),
                                   key=lambda x: x[1].get('pr_auc', 0),
                                   reverse=True):
        p50 = metrics.get('precision@50', 0)
        p100 = metrics.get('precision@100', 0)
        pr_auc = metrics.get('pr_auc', 0)
        print(f"{method:<25} {p50:<10.3f} {p100:<10.3f} {pr_auc:<10.3f}")

    # Step 5: Visualize results
    print("\n[Step 5/5] Generating visualizations...")
    visualize_results(
        results=results,
        output_path=args.output_path,
        task=args.task
    )

    # Save results to JSON
    import json
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(args.output_path, f"results_{args.task}_{timestamp}.json")

    # Convert results to JSON-serializable format
    json_results = {
        'task': args.task,
        'cutoff_year': args.cutoff_year,
        'timestamp': timestamp,
        'metrics': results['metrics'],
        'num_test_edges': len(test_edges),
        'train_graph_nodes': train_graph.number_of_nodes(),
        'train_graph_edges': train_graph.number_of_edges()
    }

    with open(results_file, 'w') as f:
        json.dump(json_results, f, indent=2)

    print(f"\n   Results saved to: {results_file}")

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()
