import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from env.loader import load_all
from env.state_mapper import build_state

class RailwayEnv(gym.Env):

    def __init__(self):
        super().__init__()

        self.geometry, self.timetable, self.signals, self.physics = load_all()

        # FIX A: Actions are now 0, 1, 2 (Directly mapping to train indices)
        self.action_space = spaces.Discrete(3)

        # 3 trains × 4 + 3 blocks = 15
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(15,),
            dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # FIX C: Hard episode limit initialization
        self.step_count = 0

        # Initialize blocks
        self.blocks = [
            {"id": 0, "occupied_by": None},
            {"id": 1, "occupied_by": None},
            {"id": 2, "occupied_by": None}
        ]

        # Spawn 3 trains
        self.trains = []
        for i in range(3):
            meta = self.timetable.sample().iloc[0]
            self.trains.append({
                "meta": meta,
                "speed": random.uniform(40, 80),
                "distance": random.uniform(50, 200),
                "block": i,
                "done": False
            })
            self.blocks[i]["occupied_by"] = i

        return self._get_state(), {}

    def _get_state(self):
        return build_state(self.trains, self.blocks)

    def step(self, action):
        # FIX C: Increment step count
        self.step_count += 1
        
        # Total remaining distance before action
        total_remaining = sum(t["distance"] for t in self.trains if not t["done"])

        # FIX A: Direct mapping
        selected = action 
        reward = 0.0

        for i, train in enumerate(self.trains):
            if train["done"]:
                continue

            # FIX B: All trains move every step
            train["distance"] -= train["speed"]
            
            # Penalty for every step taken (incentivizes speed)
            reward -= 0.001

            # FIX D: Simplified completion & transition
            if train["distance"] <= 0:
                train["block"] += 1
                train["distance"] = random.uniform(50, 200)

                if train["block"] >= 2:
                    train["done"] = True
                    reward += 20  # Big finish bonus

            # Selection logic: Reward the agent if the train it "focused" on made progress
            if i == selected:
                reward += 0.1 

        # Termination logic
        terminated = all(t["done"] for t in self.trains)
        
        # FIX C: Hard Episode Limit
        if self.step_count >= 200:
            terminated = True
            
        truncated = False

        # Progress shaping
        new_remaining = sum(t["distance"] for t in self.trains if not t["done"])
        progress = (total_remaining - new_remaining) / 3000.0
        reward += progress * 20.0

        reward = np.clip(reward, -10, 10)

        return self._get_state(), reward, terminated, truncated, {}