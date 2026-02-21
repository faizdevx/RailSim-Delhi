from stable_baselines3 import PPO
from env.railway_env import RailwayEnv

env = RailwayEnv()

model = PPO("MlpPolicy", env, verbose=1, device="cpu")
model.learn(total_timesteps=50000)

model.save("rail_dispatch_phase2")