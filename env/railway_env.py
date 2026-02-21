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
        self.speed = 0
        self.prev_speed = 0
        self.distance = 0
        self.braking_rate = 0
        self.platform_available = 0
        self.track_capacity = 0

        # 3 actions: coast, accelerate, brake
        self.action_space = spaces.Discrete(3)

        # now 14-dimensional continuous state
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(14,),
            dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.train = self.timetable.sample().iloc[0]
        self.segment = self.geometry.sample().iloc[0]
        self.env = random.choice(self.physics)

        self.signal = random.randint(0, 3)

        # new dynamics
        self.speed = random.uniform(20, 80)
        self.prev_speed = self.speed
        self.distance = random.uniform(500, 3000)
        self.braking_rate = random.uniform(0.5, 1.5)
        self.platform_available = random.randint(0, 1)
        self.track_capacity = random.uniform(0.5, 1.0)
        self.delay = 0
        self.scheduled_distance = self.distance

        # Occupancy derived from velocity (physical)
        geo_speed = float(self.segment.get("Max_Permissible_Speed_KMH", 130))
        self.occupancy = np.clip(self.speed / max(geo_speed, 1e-5), 0.0, 1.0)

        return self._get_state(), {}

    def _get_state(self):
        state = build_state(
            self.train,
            self.segment,
            self.env,
            self.signal,
            self.occupancy,
            self.speed,
            self.speed - self.prev_speed,
            self.distance,
            self.braking_rate,
            self.platform_available,
            self.track_capacity
        )

        state = np.asarray(state, dtype=np.float32)

        # ensure returned state matches observation_space shape (14,)
        if state.size < 14:
            pad = np.zeros(14 - state.size, dtype=np.float32)
            state = np.concatenate([state, pad])
        elif state.size > 14:
            state = state[:14]

        return state

    def step(self, action):

        # REAL PHYSICS CORE
        dt = 1.0  # 1 second timestep
        g = 9.81
        mu = float(self.env.get("Adhesion_Coefficient", 0.3))

        # Max accel from adhesion
        a_max = mu * g

        # Driver command
        if action == 1:
            desired_accel = a_max
        elif action == 2:
            desired_accel = -self.braking_rate * g
        else:
            desired_accel = 0.0

        # Geometry speed limit
        geo_speed = float(self.segment.get("Max_Permissible_Speed_KMH", 130))

        # Block congestion auto-brake
        # occupancy uses previous step value here
        if self.occupancy > 0.8:
            desired_accel = -self.braking_rate * g

        # Red signal protection
        if self.signal == 0:
            stop_dist = (self.speed ** 2) / (2 * self.braking_rate * g + 1e-5)
            if stop_dist >= self.distance:
                desired_accel = -self.braking_rate * g

        # Apply acceleration
        self.prev_speed = self.speed
        self.speed += desired_accel * dt

        # Speed bounds (geometry speed limit)
        self.speed = np.clip(self.speed, 0.0, geo_speed)

        # Position update
        self.distance -= self.speed * dt

        # Update occupancy physically (congestion relates to velocity)
        self.occupancy = np.clip(self.speed / max(geo_speed, 1e-5), 0.0, 1.0)

        # Update delay
        if self.distance > 0:
            self.delay = min(self.delay + 1, 300)

        # Acceleration (for state)
        acceleration = self.speed - self.prev_speed

        # stochastic signal
        if random.random() < 0.2:
            self.signal = random.randint(0, 3)

        # Overspeed penalty
        speed_limit = self.segment.get("Max_Permissible_Speed_KMH", 130)
        overspeed = max(0.0, self.speed - speed_limit)
        overspeed_penalty = min(overspeed / 10.0, 2.0)

        # Signal violation penalty (red == 0)
        signal_penalty = 0.0
        if self.signal == 0 and self.speed > 5:
            signal_penalty = 5.0

        # Congestion penalty
        congestion_penalty = self.occupancy * 2.0

        # Priority-weighted delay (normalized)
        priority = float(self.train.get("Priority_Score", 1.0))
        delay_penalty = (self.delay / 300.0) * priority

        # Progress reward
        if self.scheduled_distance > 0:
            progress = (self.scheduled_distance - self.distance) / self.scheduled_distance
            progress = max(0.0, progress)
        else:
            progress = 1.0 if self.distance <= 0 else 0.0
        progress_reward = progress * 2.0

        # Arrival bonus (reduced)
        arrival_bonus = 5.0 if self.distance <= 0 else 0.0

        reward = (
            -delay_penalty
            -congestion_penalty
            -overspeed_penalty
            -signal_penalty
            +progress_reward
            +arrival_bonus
        )

        # Termination on arrival
        terminated = self.distance <= 0

        truncated = False

        # Bound total reward
        reward = np.clip(reward, -10, 10)

        return self._get_state(), reward, terminated, truncated, {}