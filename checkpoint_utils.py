#!/usr/bin/env python3
"""
Checkpoint Management Tool

"""

import argparse
import os
from checkpoint_manager import CheckpointManager, BBHAllCheckpointManager
from tabulate import tabulate

def list_checkpoints():
    """List all checkpoints"""
    checkpoint_manager = CheckpointManager()
    checkpoints = checkpoint_manager.list_checkpoints()

    if not checkpoints:
        print("No checkpoints found")
        return

    # Prepare table data
    headers = ["Method", "Dataset", "Status", "Created At"]
    table_data = []

    for cp in checkpoints:
        table_data.append([
            cp['method'],
            cp['dataset'],
            cp['status'],
            cp['timestamp'][:19] if cp['timestamp'] != 'unknown' else 'unknown'
        ])

    print("All Checkpoints:")
    print(tabulate(table_data, headers=headers, tablefmt="grid"))

def show_checkpoint_details(method: str, dataset: str):
    """Show checkpoint details"""
    if dataset == "bbh_all":
        checkpoint_manager = BBHAllCheckpointManager()
    else:
        checkpoint_manager = CheckpointManager()

    checkpoint = checkpoint_manager.load_checkpoint(method, dataset)

    if not checkpoint:
        print(f"Checkpoint not found: {method}_{dataset}")
        return

    print(f"Checkpoint Details: {method}_{dataset}")
    print("=" * 50)
    print(f"Method: {checkpoint.get('method', 'Unknown')}")
    print(f"Dataset: {checkpoint.get('dataset', 'Unknown')}")
    print(f"Status: {checkpoint.get('status', 'Unknown')}")
    print(f"Created At: {checkpoint.get('timestamp', 'Unknown')}")

    if dataset == "bbh_all":
        completed_tasks = checkpoint.get('completed_tasks', {})
        failed_tasks = checkpoint.get('failed_tasks', {})

        print(f"\nTask Progress:")
        print(f"Completed tasks: {len(completed_tasks)}")
        print(f"Failed tasks: {len(failed_tasks)}")
        print(f"Current index: {checkpoint.get('current_task_index', 0)}")

        if completed_tasks:
            print(f"\nCompleted Tasks:")
            for task_name, result in completed_tasks.items():
                score = result.get('final_score', 0.0)
                print(f"  - {task_name}: {score:.4f}")

        if failed_tasks:
            print(f"\nFailed Tasks:")
            for task_name, result in failed_tasks.items():
                error = result.get('error', 'Unknown error')
                print(f"  - {task_name}: {error}")

def clean_completed_checkpoints():
    """Clean completed checkpoints"""
    checkpoint_manager = CheckpointManager()
    checkpoints = checkpoint_manager.list_checkpoints()

    completed_checkpoints = [cp for cp in checkpoints if cp['status'] == 'completed']

    if not completed_checkpoints:
        print("No completed checkpoints found")
        return

    print(f"Found {len(completed_checkpoints)} completed checkpoints")

    for cp in completed_checkpoints:
        response = input(f"Delete {cp['method']}_{cp['dataset']}? (y/N): ")
        if response.lower() == 'y':
            checkpoint_manager.delete_checkpoint(cp['method'], cp['dataset'])
            print(f"Deleted: {cp['method']}_{cp['dataset']}")

def delete_checkpoint(method: str, dataset: str):
    """Delete specific checkpoint"""
    if dataset == "bbh_all":
        checkpoint_manager = BBHAllCheckpointManager()
    else:
        checkpoint_manager = CheckpointManager()

    if not checkpoint_manager.checkpoint_exists(method, dataset):
        print(f"Checkpoint does not exist: {method}_{dataset}")
        return

    response = input(f"Confirm delete checkpoint {method}_{dataset}? (y/N): ")
    if response.lower() == 'y':
        checkpoint_manager.delete_checkpoint(method, dataset)
        print(f"Checkpoint deleted: {method}_{dataset}")
    else:
        print("Operation cancelled")

def main():
    parser = argparse.ArgumentParser(description="Checkpoint Management Tool")
    parser.add_argument("--list", action="store_true", help="List all checkpoints")
    parser.add_argument("--show", nargs=2, metavar=("METHOD", "DATASET"), help="Show specific checkpoint details")
    parser.add_argument("--clean", action="store_true", help="Clean completed checkpoints")
    parser.add_argument("--delete", nargs=2, metavar=("METHOD", "DATASET"), help="Delete specific checkpoint")

    args = parser.parse_args()

    if args.list:
        list_checkpoints()
    elif args.show:
        show_checkpoint_details(args.show[0], args.show[1])
    elif args.clean:
        clean_completed_checkpoints()
    elif args.delete:
        delete_checkpoint(args.delete[0], args.delete[1])
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 