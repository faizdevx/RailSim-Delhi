import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from collections import defaultdict

class RailwayEnv(gym.Env):
    """
    Phase 1-7 Integrated Railway Traffic Optimization Environment.
    Features: N-Step Lookahead, Siding Overtakes, and Priority Scheduling.
    """
    def __init__(self, num_trains=3, num_blocks=10, max_steps=500):
        super(RailwayEnv, self).__init__()
        
        self.num_trains = num_trains
        self.num_blocks = num_blocks
        self.max_steps = max_steps
        self.lookahead = 5
        
        # Actions per train: 0=Keep Main/Move, 1=Divert to Siding, 2=Hold/Stop
        # Total action space is MultiDiscrete: one choice per train
        self.action_space = spaces.MultiDiscrete([3] * self.num_trains)
        
        # Observation: [Pos, Speed, Priority, TrackType, BlockID, ArrivalStatus] * Num_Trains
        # + [Block Occupancy Main, Block Occupancy Siding] * Num_Blocks
        obs_size = (self.num_trains * 6) + (self.num_blocks * 2)
        self.observation_space = spaces.Box(low=0, high=1, shape=(obs_size,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.reservations = {} # (block_id, time_step) -> train_id
        
        # Track occupancy
        self.occupancy_main = [None] * self.num_blocks
        self.occupancy_siding = [None] * self.num_blocks
        
        # Priority: 3=Vande Bharat, 2=Passenger, 1=Freight
        self.trains = []
        for i in range(self.num_trains):
            priority = random.choice([1, 2, 3])
            train = {
                "id": i,
                "pos": 0.0,
                "speed": 0.0,
                "max_speed": 0.8 + (priority * 0.1), # Vande is faster
                "priority": priority,
                "block": 0,
                "track": "main", # "main" or "siding"
                "done": False,
                "delay": 0.0
            }
            self.trains.append(train)
            self.occupancy_main[0] = i
            
        return self._get_obs(), {}

    def _get_obs(self):
        train_states = []
        for t in self.trains:
            train_states.extend([
                t["pos"] / self.num_blocks,
                t["speed"],
                t["priority"] / 3.0,
                1.0 if t["track"] == "main" else 0.0,
                t["block"] / self.num_blocks,
                float(t["done"])
            ])
        
        block_states = []
        for i in range(self.num_blocks):
            block_states.append(1.0 if self.occupancy_main[i] is not None else 0.0)
            block_states.append(1.0 if self.occupancy_siding[i] is not None else 0.0)
            
        return np.array(train_states + block_states, dtype=np.float32)

    def step(self, actions):
        self.steps += 1
        reward = 0.0
        conflicts = 0
        overtakes = 0
        
        # Update Reservations (Phase 5: Lookahead)
        self.reservations = {}
        for t in self.trains:
            if t["done"]: continue
            for l in range(1, self.lookahead + 1):
                future_block = t["block"] + l
                if future_block < self.num_blocks:
                    # Predictive ownership
                    self.reservations[(future_block, self.steps + l)] = t["id"]

        # Logic per train
        for i, t in enumerate(self.trains):
            if t["done"]: continue
            
            action = actions[i]
            
            # Action logic
            if action == 1 and t["track"] == "main": # Divert to Siding
                if self.occupancy_siding[t["block"]] is None:
                    self.occupancy_main[t["block"]] = None
                    self.occupancy_siding[t["block"]] = i
                    t["track"] = "siding"
                    t["speed"] *= 0.5 # Slow down on diversion
            
            elif action == 0 and t["track"] == "siding": # Release to Main
                if self.occupancy_main[t["block"]] is None:
                    self.occupancy_siding[t["block"]] = None
                    self.occupancy_main[t["block"]] = i
                    t["track"] = "main"

            # Physics & Movement (Phase 1)
            if action != 2: # Not Holding
                t["speed"] = min(t["max_speed"], t["speed"] + 0.1)
            else:
                t["speed"] = max(0.0, t["speed"] - 0.2)
                t["delay"] += 1.0

            # Block Transition
            if t["speed"] > 0:
                new_pos = t["pos"] + t["speed"]
                new_block = int(new_pos)
                
                if new_block > t["block"]:
                    if new_block >= self.num_blocks:
                        # Arrival (Phase 3)
                        t["done"] = True
                        if t["track"] == "main": self.occupancy_main[t["block"]] = None
                        else: self.occupancy_siding[t["block"]] = None
                        reward += 100.0 # Arrival Bonus
                    else:
                        # Check safety (Phase 2 & 5)
                        conflict = False
                        if t["track"] == "main":
                            if self.occupancy_main[new_block] is not None: conflict = True
                        else:
                            if self.occupancy_siding[new_block] is not None: conflict = True
                        
                        # Check Reservations
                        res_owner = self.reservations.get((new_block, self.steps + 1))
                        if res_owner is not None and res_owner != i:
                            if self.trains[res_owner]["priority"] > t["priority"]:
                                conflict = True # Yield to higher priority

                        if not conflict:
                            # Move successful
                            if t["track"] == "main":
                                self.occupancy_main[t["block"]] = None
                                self.occupancy_main[new_block] = i
                            else:
                                self.occupancy_siding[t["block"]] = None
                                self.occupancy_siding[new_block] = i
                            
                            # Check for Overtake (Phase 4)
                            if t["priority"] > 1 and t["track"] == "main":
                                if self.occupancy_siding[new_block] is not None:
                                    overtakes += 1
                                    reward += 30.0
                            
                            t["block"] = new_block
                            t["pos"] = new_pos
                        else:
                            # Safety Stop
                            t["speed"] = 0.0
                            conflicts += 1
                            reward -= 50.0 # Conflict Penalty
                else:
                    t["pos"] = new_pos

        # Reward Calculation (Phase 7)
        total_delay = sum([t["delay"] * t["priority"] for t in self.trains])
        reward -= total_delay * 0.1
        
        terminated = all([t["done"] for t in self.trains])
        truncated = self.steps >= self.max_steps
        
        if truncated and not terminated:
            reward -= 100.0 # Deadlock/Timeout Penalty
            
        # Normalization
        reward = np.tanh(reward / 100.0)
        
        return self._get_obs(), reward, terminated, truncated, {"overtakes": overtakes, "delays": total_delay}