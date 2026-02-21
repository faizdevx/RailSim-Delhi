import numpy as np

def build_state(trains, blocks):

    state = []

    num_blocks = len(blocks)

    for t in trains:
        state += [
            t["block"] / num_blocks,                 # current block
            min(t["distance"] / 3000.0, 1.0),       # distance norm
            t["speed"] / 130.0,                     # speed norm
            int(t["done"])                          # finished flag
        ]

    # block occupancy
    for b in blocks:
        state.append(0 if b["occupied_by"] is None else 1)

    return np.array(state, dtype=np.float32)