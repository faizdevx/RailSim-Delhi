import os
import pandas as pd
import json

def load_all(data_path="./data"):
    geometry = pd.read_csv(os.path.join(data_path, "geometry_constraints.csv"))
    timetable = pd.read_csv(os.path.join(data_path, "master_timetable.csv"))
    signals = pd.read_csv(os.path.join(data_path, "signal_map.csv"))

    with open(os.path.join(data_path, "physics_modifiers.json")) as f:
        physics = json.load(f)

    return geometry, timetable, signals, physics