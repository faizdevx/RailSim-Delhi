from env.railway_env import RailwayEnv

env = RailwayEnv()

state = env.reset()

for _ in range(5):
    action = 1
    state, reward, done, _ = env.step(action)
    print(state, reward)