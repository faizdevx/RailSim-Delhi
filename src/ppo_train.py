import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
from env.railway_env import RailwayEnv
import os

def train():
    # Phase 6: Centralized Control via Vectorized Env
    env = make_vec_env(lambda: RailwayEnv(num_trains=3, num_blocks=12), n_envs=4)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01, # Encourage finding overtake maneuvers
        vf_coef=0.5,
        tensorboard_log="./railway_logs/"
    )

    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./models/', name_prefix='rail_ppo')

    print("Starting Training...")
    model.learn(total_timesteps=200000, callback=checkpoint_callback)
    
    model.save("railway_ppo_final")
    print("Training Complete. Model Saved.")

if __name__ == "__main__":
    train()