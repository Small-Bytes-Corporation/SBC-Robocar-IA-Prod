# **************************************************************************** #
#                                                                              #
#                          Robocar - Manual Collector                          #
#                                                                              #
# **************************************************************************** #

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
import numpy as np
import json
import os
from pynput import mouse, keyboard
from module.data_bufferNR import DataBufferNR
from settings import (NUM_RAYS, SCREEN_WIDTH, SCREEN_HEIGHT, CONFIG_PATH, SIMULATOR_PATH, BASE_PORT, LIDAR_CONFIG)

# **************************************************************************** #
#                                                                              #
#                            AGENT CONFIGURATION                               #
#                                                                              #
# **************************************************************************** #

# Write the LiDAR configuration to JSON file used by Unity simulator
with open(CONFIG_PATH, "w") as f:
    json.dump(LIDAR_CONFIG, f)

# **************************************************************************** #
#                                                                              #
#                           INPUT STATE VARIABLES                              #
#                                                                              #
# **************************************************************************** #

steering = 0.0
acceleration = 0.0
keys_pressed = set()
record = False
was_r_pressed = False
data_buffer = DataBufferNR()

# **************************************************************************** #
#                                                                              #
#                             INPUT NORMALIZATION                              #
#                                                                              #
# **************************************************************************** #

def normalize(val, min_val, max_val):
    return np.clip((val - min_val) / (max_val - min_val) * 2 - 1, -1, 1)

def on_move(x, y):
    global steering, acceleration
    steering = normalize(x, 0, SCREEN_WIDTH)
    acceleration = -normalize(y, 0, SCREEN_HEIGHT)

# **************************************************************************** #
#                                                                              #
#                         KEYBOARD INPUT CALLBACKS                             #
#                                                                              #
# **************************************************************************** #

def on_press(key):
    global record, was_r_pressed
    try:
        if key.char:
            keys_pressed.add(key.char.lower())
    except AttributeError:
        keys_pressed.add(key)

    if 'r' in keys_pressed and not was_r_pressed:
        record = not record
        was_r_pressed = True
        print("-> Recording activated." if record else "-> Recording disabled.")

def on_release(key):
    global was_r_pressed
    try:
        if key.char:
            keys_pressed.discard(key.char.lower())
    except AttributeError:
        keys_pressed.discard(key)
    if 'r' not in keys_pressed:
        was_r_pressed = False

mouse.Listener(on_move=on_move).start()
keyboard.Listener(on_press=on_press, on_release=on_release).start()

print("\033[92mMouse-activated control!\033[0m")
print("\tR button: activate/deactivate recording")
print("\tQ button: exit")

# **************************************************************************** #
#                                                                              #
#                            UNITY ENVIRONMENT SETUP                           #
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
        if 'q' in keys_pressed:
            print("Closing...")
            break

        decision_steps, terminal_steps = env.get_steps(behavior_name)

        if len(decision_steps) == 0:
            env.reset()
            continue

        num_agents = len(decision_steps)
        continuous_actions = np.zeros((num_agents, 2), dtype=np.float32)

        for i, agent_id in enumerate(decision_steps.agent_id):
            obs = decision_steps[agent_id].obs[0]

            lidar = [float(x) for x in obs[:NUM_RAYS]]

            if record:
                data = {
                    "lidar": lidar
                }
                data_buffer.add_to_buffer(data, [acceleration, steering])

            continuous_actions[i] = [acceleration, steering]

        action = ActionTuple(continuous=continuous_actions)
        env.set_actions(behavior_name, action)
        env.step()

finally:
    data_buffer.save_data()
    env.close()
    print("Data backup and closed environment.")
