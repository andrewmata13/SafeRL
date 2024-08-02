import gym
import time
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import random
import csv
import os
import json

import sys
sys.path.insert(0,'..')
sys.path.insert(1,'../..')

def make_states(lead_x_list, lead_y_list, wingman_x_list, wingman_y_list, lead_speed_list, wingman_speed_list, lead_heading_list, wingman_heading_list):
    '''make states from raw data'''

    lead_vx_normalized = np.cos(lead_heading_list)
    lead_vy_normalized = np.sin(lead_heading_list)

    # state is lead aircraft that doesn't change direction or speed
    states = list(zip(lead_x_list, lead_y_list, lead_heading_list, lead_speed_list))

    #sigma = [1000, 1, 1, 1000, 1, 1, 400, 1, 1, 400, 1, 1] # normalization

    #states = []

    #for i in range(len(lead_x_list)):
    #    lead_x = lead_x_list[i]
    #    lead_y = lead_y_list[i]
    #    wingman_x = wingman_x_list[i]
    #    wingman_y = wingman_y_list[i]
    #    lead_speed = lead_speed_list[i]
    #    wingman_speed = wingman_speed_list[i]
    #    lead_heading = lead_heading_list[i]
    #    wingman_heading = wingman_heading_list[i]

    return states


file_path = "../output/expr_20240522_143535/PPO_DubinsRejoin_15bc3_00000_0_2024-05-22_14-35-38/eval/ckpt_200/eval.log"
with open(file_path, 'r') as file:
    data = [json.loads(line) for line in file]

print(f"Loaded {len(data)} lines from {file_path}") 

output = []

# Split data into episodes 
batch = []
for data_item in data:
    if not (data_item['info']['failure'] or data_item['info']['success']):
        batch.append(data_item)
    else:
        # end of batch
        print(f"Batch index={len(output)} of length {len(batch)}")
        lead_x = np.array([entry['info']['lead']['x'] for entry in batch])
        lead_y = np.array([entry['info']['lead']['y'] for entry in batch])
        wingman_x = np.array([entry['info']['wingman']['x'] for entry in batch])
        wingman_y = np.array([entry['info']['wingman']['y'] for entry in batch])
        lead_speed = np.array([entry['info']['lead']['v'] for entry in batch])
        wingman_speed = np.array([entry['info']['wingman']['v'] for entry in batch])
        lead_heading = np.array([entry['info']['lead']['heading'] for entry in batch])
        wingman_heading = np.array([entry['info']['wingman']['heading'] for entry in batch])

        actions = [entry['actions'] for entry in batch]

        states = make_states(lead_x, lead_y, wingman_x, wingman_y, lead_speed, wingman_speed, lead_heading, wingman_heading)

        #lead_vx = lead_speed * np.cos(lead_heading)
        #lead_vy = lead_speed * np.sin(lead_heading)

        #lead_vx_normalized = np.cos(lead_heading)
        #lead_vy_normalized = np.sin(lead_heading)

        #states = list(zip(lead_x, lead_y, lead_heading, lead_speed))
        #states = list(zip(lead_x, lead_y, lead_vx, lead_vy, wingman_x, wingman_y))
        #states = list(zip(lead_x, lead_y, wingman_x, wingman_y))
        #states = list(zip(lead_x, lead_y, wingman_x, wingman_y, lead_speed, wingman_speed, lead_heading, wingman_heading))
        #states = list(zip(lead_x, lead_y, lead_speed, lead_heading))
        #states = list(zip(wingman_x, wingman_y, wingman_speed, wingman_heading))

        #states = list(zip(lead_x - wingman_x, lead_y - wingman_y, lead_speed - wingman_speed, lead_heading - wingman_heading))
        
        times = [t for t in range(0,len(states))]

        output.append([actions, times, states])

        batch = []
        
#output = output[0:6]
#print(f"Note: truncated data into {len(output)} episodes")


# Write CSV Files
for i, measurement in enumerate(output):
    current_dir = "DubinsRejoin/measurement_" + str(i) + "/"

    if not os.path.exists(current_dir):
        os.makedirs(current_dir)

    #print(f"saving to {current_dir}")
    
    inputs = measurement[0]
    steps = measurement[1]
    trajectory = measurement[2]
    
    with open(current_dir + "input.csv", "w") as csvfile:
        writer = csv.writer(csvfile)
        for actions in inputs:
            writer.writerow(actions)

    with open(current_dir + "time.csv", "w") as csvfile:
        writer = csv.writer(csvfile)
        for step in steps:
            writer.writerow([step])

    with open(current_dir + "trajectory.csv", "w") as csvfile:
        writer = csv.writer(csvfile)
        for state in trajectory:
            writer.writerow(state)
