#!/usr/bin/env python3
"""Analyze Adroit Hand observation structure to find goal."""

import gymnasium as gym
import gymnasium_robotics
import numpy as np

print("=" * 70)
print("Analyzing Adroit Hand Observation Structure")
print("=" * 70)

env = gym.make('AdroitHandDoor-v1')
print(f"\nEnvironment: AdroitHandDoor-v1")
print(f"Observation space: {env.observation_space}")
print(f"Action space: {env.action_space}")

# Reset and get observation
obs, info = env.reset()
print(f"\nObservation shape: {obs.shape}")
print(f"Observation: {obs}")
print(f"\nInfo keys: {info.keys()}")
for key, value in info.items():
    print(f"  {key}: {value}")

# Check if there's goal information in the environment
print(f"\nEnvironment attributes:")
if hasattr(env.unwrapped, 'goal'):
    print(f"  env.unwrapped.goal: {env.unwrapped.goal}")
if hasattr(env.unwrapped, 'target_pos'):
    print(f"  env.unwrapped.target_pos: {env.unwrapped.target_pos}")
if hasattr(env.unwrapped, '_target'):
    print(f"  env.unwrapped._target: {env.unwrapped._target}")

# Try multiple resets to see if goal changes
print(f"\nTesting multiple resets to see goal variation:")
goals = []
for i in range(5):
    obs, info = env.reset()
    print(f"  Reset {i+1}: obs[-3:] = {obs[-3:]}, obs[-6:-3] = {obs[-6:-3]}")
    
    # Check for goal-related attributes
    if hasattr(env.unwrapped, 'goal'):
        print(f"           goal attribute: {env.unwrapped.goal}")
        goals.append(env.unwrapped.goal)

if goals:
    goals = np.array(goals)
    print(f"\nGoal variation: std = {np.std(goals, axis=0)}")
    print(f"Goals are {'RANDOMIZED' if np.any(np.std(goals, axis=0) > 0.01) else 'FIXED'}")

# Take a step and see what happens
print(f"\nTaking a step:")
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
print(f"  Reward: {reward}")
print(f"  Info: {info}")

env.close()


