import numpy as np

TRAIN_TYPE = {"VB":0,"RAJ":1,"EMU":2,"FRT":3}

def build_state(train, segment, env, signal, occupancy,
                speed, accel, distance,
                braking_rate, platform, track_cap):

    state = [

    TRAIN_TYPE[train["Type"]],
    train["Priority_Score"]/5.0,

    speed/130.0,
    (accel+10)/20.0,
    min(distance/3000.0,1.0),
    braking_rate/2.0,

    platform,
    track_cap,

    segment["Max_Permissible_Speed_KMH"]/130.0,
    segment["Max_Permissible_Speed_KMH"]/130.0,

    signal/3.0,

    env["Adhesion_Coefficient"],
    env["Visibility_KM"]/10.0,

    occupancy
    ]

    return np.array(state,dtype=np.float32)