# !/bin/bash

export NVIDIA_TF32_OVERRIDE=0 # disable tensorcore
python problem_script_onenode.py
