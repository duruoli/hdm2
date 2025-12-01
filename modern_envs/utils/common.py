"""
Minimal utilities copied from multiworld to avoid dependencies.

Only includes what's actually needed for modern environments.
"""

import numpy as np
from collections import OrderedDict


class Serializable:
    """
    Simple serializable class for saving/loading environment configs.
    Copied from multiworld.core.serializable.
    """
    def quick_init(self, locals_):
        """Store constructor arguments for serialization."""
        if not hasattr(self, '_serialized'):
            self._serialized = {}
        if '_serialized' in locals_:
            # Avoid infinite recursion
            return
        for k, v in locals_.items():
            if k != 'self' and k != '__class__':
                self._serialized[k] = v


class MultitaskEnv:
    """
    Interface for multi-task/goal-conditioned environments.
    Copied from multiworld.core.multitask_env.
    """
    def get_goal(self):
        """Return current goal."""
        raise NotImplementedError()
    
    def set_goal(self, goal):
        """Set a new goal."""
        raise NotImplementedError()
    
    def sample_goals(self, batch_size):
        """Sample a batch of goals."""
        raise NotImplementedError()


# Utility functions for logging
def get_stat_in_paths(paths, dict_name, scalar_name):
    """
    Extract a statistic from a list of paths.
    
    Args:
        paths: List of trajectory dictionaries
        dict_name: Name of dict in trajectory (e.g., 'env_infos')
        scalar_name: Name of scalar in dict (e.g., 'distance')
        
    Returns:
        List of arrays, one per trajectory
    """
    if len(paths) == 0:
        return np.zeros(0)

    def get_path_stat(path, dict_name, scalar_name):
        if dict_name in path:
            if scalar_name in path[dict_name][0]:
                return [info[scalar_name] for info in path[dict_name]]
        return []
    
    return [
        np.array(get_path_stat(path, dict_name, scalar_name))
        for path in paths
    ]


def create_stats_ordered_dict(
        name,
        data,
        stat_prefix=None,
        always_show_all_stats=False,
        exclude_max_min=False,
):
    """
    Create OrderedDict of statistics from data.
    
    Args:
        name: Name prefix for stats
        data: Data to compute stats on
        stat_prefix: Additional prefix (optional)
        always_show_all_stats: Whether to show all stats
        exclude_max_min: Whether to exclude max/min
        
    Returns:
        OrderedDict with statistics
    """
    if stat_prefix is not None:
        name = f"{stat_prefix} {name}"
    
    if isinstance(data, list):
        data = np.concatenate([np.array(x).flatten() for x in data])
    elif not isinstance(data, np.ndarray):
        data = np.array(data)
    
    if len(data) == 0:
        return OrderedDict()
    
    stats = OrderedDict([
        (f'{name} Mean', np.mean(data)),
        (f'{name} Std', np.std(data)),
    ])
    
    if always_show_all_stats or len(data) > 1:
        if not exclude_max_min:
            stats[f'{name} Max'] = np.max(data)
            stats[f'{name} Min'] = np.min(data)
    
    return stats

