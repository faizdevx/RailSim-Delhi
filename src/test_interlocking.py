import gymnasium as gym
from stable_baselines3 import PPO
from env.railway_env import RailwayEnv

# 1. Load the "Stable" Model
model = PPO.load("ppo_railway_phase6_predictive")

# 2. Create a Custom Stress-Test Scenario
env = RailwayEnv(max_steps=500)
obs, _ = env.reset()

# 🔧 FORCE A BOTTLENECK: 
# Put a slow FREIGHT (Priority 1) in Block 1
# Put a fast VANDE (Priority 2) in Block 0
env.trains[0]["priority"] = 1 # Freight
env.trains[1]["priority"] = 2 # Vande Bharat
env.trains[1]["speed"] = 60    # Max speed

print("🚦 STRESS TEST START: Freight blocking Vande Bharat...")

for step in range(100):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    
    # Text-based Visualization
    track_map = ["[" + (" " if b["occupied_main"] is None else f"T{b['occupied_main']}") + "]" for b in env.blocks]
    siding_map = ["(" + (" " if b["occupied_siding"] is None else f"T{b['occupied_siding']}") + ")" for b in env.blocks]
    
    print(f"Step {step} | Action: {action} | Main: {''.join(track_map)} | Siding: {''.join(siding_map)}")
    
    if action == 4:
        print("⚡ INTERLOCKING TRIGGERED: Shunting Freight to Siding!")

    if terminated or truncated:
        print("✅ Simulation Finished.")
        break