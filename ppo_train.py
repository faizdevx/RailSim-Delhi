from stable_baselines3 import PPO
from env.railway_env import RailwayEnv

env = RailwayEnv()

model = PPO("MlpPolicy", env, verbose=1, device="cpu")
model.learn(total_timesteps=20000)

model.save("rail_dispatch_agent")

# quick rollout / debug
s, _ = env.reset()

for i in range(20):
    s, r, _, _, _ = env.step(1)
    print("speed:", env.speed, "distance:", env.distance)