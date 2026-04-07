import os
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

class CheckpointManager:
    """Manage experiment checkpoint and resume functionality"""

    def __init__(self, base_dir: str = "checkpoints"):
        """
        Initialize checkpoint manager

        Args:
            base_dir: Base directory for checkpoint file storage
        """
        self.base_dir = base_dir
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        
        logging.info(f"Checkpoint directory: {base_dir}")

    def get_checkpoint_path(self, method: str, dataset: str, experiment_id: str = None) -> str:
        """Get checkpoint file path"""
        if experiment_id is None:
            experiment_id = f"{method}_{dataset}"
        return os.path.join(self.base_dir, f"{experiment_id}_checkpoint.json")

    def save_checkpoint(self, method: str, dataset: str, checkpoint_data: Dict[str, Any],
                       experiment_id: str = None) -> None:
        """
        Save checkpoint

        Args:
            method: Optimization method name
            dataset: Dataset name
            checkpoint_data: Checkpoint data to save
            experiment_id: Experiment ID (optional)
        """
        checkpoint_path = self.get_checkpoint_path(method, dataset, experiment_id)

        # Add metadata
        checkpoint_data.update({
            "method": method,
            "dataset": dataset,
            "timestamp": datetime.now().isoformat(),
            "checkpoint_version": "1.0"
        })

        try:
            # Ensure all data is serializable
            serializable_data = self._make_serializable(checkpoint_data)

            with open(checkpoint_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_data, f, indent=2, ensure_ascii=False)

            logging.info(f"Checkpoint saved: {checkpoint_path}")
        except Exception as e:
            logging.error(f"Failed to save checkpoint: {e}")
            # Save error info to checkpoint
            error_checkpoint = {
                "method": method,
                "dataset": dataset,
                "timestamp": datetime.now().isoformat(),
                "checkpoint_version": "1.0",
                "error": str(e),
                "status": "error"
            }
            with open(checkpoint_path, 'w', encoding='utf-8') as f:
                json.dump(error_checkpoint, f, indent=2, ensure_ascii=False)

    def load_checkpoint(self, method: str, dataset: str, experiment_id: str = None) -> Optional[Dict[str, Any]]:
        """
        Load checkpoint

        Args:
            method: Optimization method name
            dataset: Dataset name
            experiment_id: Experiment ID (optional)

        Returns:
            Checkpoint data, or None if not exists
        """
        checkpoint_path = self.get_checkpoint_path(method, dataset, experiment_id)

        if not os.path.exists(checkpoint_path):
            logging.info(f"Checkpoint file not found: {checkpoint_path}")
            return None

        try:
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)

            logging.info(f"Checkpoint loaded: {checkpoint_path}")
            logging.info(f"Checkpoint created at: {checkpoint_data.get('timestamp', 'Unknown')}")

            return checkpoint_data
        except Exception as e:
            logging.error(f"Failed to load checkpoint: {e}")
            return None

    def checkpoint_exists(self, method: str, dataset: str, experiment_id: str = None) -> bool:
        """Check if checkpoint exists"""
        checkpoint_path = self.get_checkpoint_path(method, dataset, experiment_id)
        return os.path.exists(checkpoint_path)

    def delete_checkpoint(self, method: str, dataset: str, experiment_id: str = None) -> None:
        """Delete checkpoint file"""
        checkpoint_path = self.get_checkpoint_path(method, dataset, experiment_id)
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
            logging.info(f"Checkpoint deleted: {checkpoint_path}")

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all available checkpoints"""
        checkpoints = []

        if not os.path.exists(self.base_dir):
            return checkpoints

        for filename in os.listdir(self.base_dir):
            if filename.endswith('_checkpoint.json'):
                filepath = os.path.join(self.base_dir, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    checkpoints.append({
                        "filename": filename,
                        "method": data.get("method", "unknown"),
                        "dataset": data.get("dataset", "unknown"),
                        "timestamp": data.get("timestamp", "unknown"),
                        "status": data.get("status", "unknown")
                    })
                except Exception as e:
                    logging.warning(f"Cannot read checkpoint file {filename}: {e}")

        return checkpoints

    def _make_serializable(self, data: Any) -> Any:
        """Convert data to JSON serializable format"""
        if isinstance(data, dict):
            return {key: self._make_serializable(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._make_serializable(item) for item in data]
        elif hasattr(data, 'to_dict'):  # Objects with to_dict method like PromptStructure
            try:
                return data.to_dict()
            except Exception as e:
                return f"<Non-serializable object: {type(data).__name__}, error: {str(e)}>"
        elif isinstance(data, (str, int, float, bool, type(None))):
            return data
        else:
            # For other non-serializable objects, convert to string representation
            return f"<{type(data).__name__}: {str(data)}>"

class BBHAllCheckpointManager(CheckpointManager):
    """Checkpoint manager for BBH multi-task experiments"""

    def save_task_result(self, method: str, task_name: str, task_result: Dict[str, Any]) -> None:
        """Save single task result"""
        experiment_id = f"{method}_bbh_all"
        checkpoint = self.load_checkpoint(method, "bbh_all", experiment_id) or {
            "completed_tasks": {},
            "failed_tasks": {},
            "current_task_index": 0,
            "total_tasks": 0,
            "status": "running"
        }

        if task_result.get("status") == "success":
            checkpoint["completed_tasks"][task_name] = task_result
            logging.info(f"Task {task_name} completed and saved to checkpoint")
        else:
            checkpoint["failed_tasks"][task_name] = task_result
            logging.info(f"Task {task_name} failed and saved to checkpoint")

        # Update progress
        total_completed = len(checkpoint["completed_tasks"]) + len(checkpoint["failed_tasks"])
        checkpoint["current_task_index"] = total_completed

        self.save_checkpoint(method, "bbh_all", checkpoint, experiment_id)

    def get_remaining_tasks(self, method: str, all_tasks: List[str]) -> List[str]:
        """Get list of remaining incomplete tasks"""
        experiment_id = f"{method}_bbh_all"
        checkpoint = self.load_checkpoint(method, "bbh_all", experiment_id)

        if checkpoint is None:
            return all_tasks

        completed_tasks = set(checkpoint.get("completed_tasks", {}).keys())
        failed_tasks = set(checkpoint.get("failed_tasks", {}).keys())
        processed_tasks = completed_tasks.union(failed_tasks)

        remaining_tasks = [task for task in all_tasks if task not in processed_tasks]

        logging.info(f"Task progress:")
        logging.info(f"   Total tasks: {len(all_tasks)}")
        logging.info(f"   Completed: {len(completed_tasks)}")
        logging.info(f"   Failed: {len(failed_tasks)}")
        logging.info(f"   Remaining: {len(remaining_tasks)}")

        return remaining_tasks

    def finalize_experiment(self, method: str, final_results: Dict[str, Any]) -> None:
        """Finalize experiment and save final results"""
        experiment_id = f"{method}_bbh_all"
        checkpoint = self.load_checkpoint(method, "bbh_all", experiment_id) or {}

        checkpoint.update({
            "status": "completed",
            "final_results": final_results,
            "completion_timestamp": datetime.now().isoformat()
        })

        self.save_checkpoint(method, "bbh_all", checkpoint, experiment_id)
        logging.info(f"Experiment {experiment_id} completed and final results saved") 