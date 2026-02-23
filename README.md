# Railway Traffic Optimization RL

This project implements a Centralized PPO agent to optimize railway traffic flow using siding overtakes and predictive reservations.

## How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Train the model: `python ppo_train.py`
3. Evaluate the agent: `python evaluation.py`
4. Monitor logs: `tensorboard --logdir ./railway_logs/`

## Key Phases Integrated
- **Phase 5 Reservation Graph**: Prevents head-on conflicts by booking blocks across future timesteps.
- **Phase 6 Multi-Discrete Control**: Centralized policy manages all trains as a single joint-action vector.
- **Phase 7 Normalization**: Uses `tanh` reward scaling to ensure stable gradient descent.