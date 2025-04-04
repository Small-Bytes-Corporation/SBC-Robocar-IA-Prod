#!/bin/bash

# ---------------------------------------------------------------------------- #
#                                                                              #
#                          Robocar - Environment Setup                         #
#                                                                              #
# ---------------------------------------------------------------------------- #


# Check if Python 3 is installed
if ! command -v python3 &>/dev/null; then
    echo "Python3 is not installed. Please install it before continuing."
    exit 1
fi

# Create a virtual environment named 'venv'
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Upgrade pip to the latest version
pip install --upgrade pip

# Install all required Python packages
pip install numpy torch matplotlib mlagents_envs pynput protobuf==3.20.3

echo "Virtual environment created and required packages installed successfully."