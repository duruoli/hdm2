"""
Setup script for HDM (Hindsight Divergence Minimization)
This makes the hdm and modern_envs packages importable.
"""

from setuptools import setup, find_packages

setup(
    name="hdm",
    version="1.0.0",
    description="Hindsight Divergence Minimization for Goal-Conditioned RL",
    packages=find_packages(include=['hdm', 'hdm.*', 'modern_envs', 'modern_envs.*']),
    python_requires='>=3.10',
    install_requires=[
        'torch>=2.9.1',
        'numpy>=2.2.6',
        'scipy>=1.15.3',
        'gym>=0.26.2',
        'gymnasium>=1.2.2',
        'gymnasium-robotics>=1.4.1',
        'metaworld>=3.0.0',
        'mujoco>=3.2.0',
    ],
    extras_require={
        'dev': ['pytest', 'black', 'flake8'],
        'mpi': ['mpi4py>=3.0.0'],
    },
)

