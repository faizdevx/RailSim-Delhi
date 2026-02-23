import gymnasium as gym
from stable_baselines3 import PPO
from ..env.railway_env import RailwayEnv
import numpy as np

def evaluate():
    env = RailwayEnv(num_trains=3, num_blocks=12)
    model = PPO.load("../models/final/railway_ppo_final.zip")
    
    obs, _ = env.reset()
    total_reward = 0
    done = False
    
    metrics = {"overtakes": 0, "delays": 0, "steps": 0}
    
    print("\n--- Evaluation Mode ---")
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        metrics["overtakes"] += info.get("overtakes", 0)
        metrics["delays"] += info.get("delays", 0)
        metrics["steps"] += 1
        total_reward += reward
        
        done = terminated or truncated
        
    print(f"Total Reward: {total_reward:.2f}")
    print(f"Total Overtakes: {metrics['overtakes']}")
    print(f"Average Weighted Delay: {metrics['delays']/metrics['steps']:.2f}")
    print(f"Throughput: {3 / (metrics['steps'] * 0.1):.2f} trains/unit_time")
    print("-----------------------")

if __name__ == "__main__":
    evaluate()