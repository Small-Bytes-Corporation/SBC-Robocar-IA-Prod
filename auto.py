# **************************************************************************** #
#                                                                              #
#                        Robocar - Auto, Model                                 #
#                                                                              #
# **************************************************************************** #
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
from agent import NRAgent
import numpy as np
import sys
import os
import json
import time
from settings import (NUM_RAYS,CONFIG_PATH,SIMULATOR_PATH,BASE_PORT,LIDAR_CONFIG)

# **************************************************************************** #
#                                                                              #
#                            ENVIRONMENT CONFIGURATION                         #
#                                                                              #
# **************************************************************************** #

# Write LiDAR config file for Unity agent
with open(CONFIG_PATH, "w") as f:
    json.dump(LIDAR_CONFIG, f)

# **************************************************************************** #
#                                                                              #
#                            AGENT INITIALIZATION                              #
#                                                                              #
# **************************************************************************** #

agent = NRAgent(NUM_RAYS)
agent.load_model(sys.argv[1]) # Load model passed as argument

def normalize_state(obs): # Normalize LiDAR data to range [0, 1]
    lidar = [x / 500.0 for x in obs[:NUM_RAYS]]
    return lidar

# **************************************************************************** #
#                                                                              #
#                         UNITY ENVIRONMENT LAUNCH                             #
#                                                                              #
# **************************************************************************** #

env = UnityEnvironment(
    file_name=SIMULATOR_PATH,
    base_port=BASE_PORT,
    seed=1,
    additional_args=["--config-path", os.path.abspath(CONFIG_PATH)]
)
env.reset()
behavior_name = list(env.behavior_specs)[0]
print(f"Behavior name: {behavior_name}")

# **************************************************************************** #
#                                                                              #
#                                 MAIN LOOP                                    #
#                                                                              #
# **************************************************************************** #

try:
    while True:
        decision_steps, terminal_steps = env.get_steps(behavior_name)

        if len(decision_steps) == 0:
            env.reset()
            continue

        for agent_id in decision_steps.agent_id:
            obs = decision_steps[agent_id].obs[0]
            state = normalize_state(obs)
            action = agent.select_action(state)

            action_tuple = ActionTuple(
                continuous=np.array([[action[0], action[1]]], dtype=np.float32)
            )
            env.set_actions(behavior_name, action_tuple)

        env.step()

finally:
    env.close()
    print("Simulation finished.")
