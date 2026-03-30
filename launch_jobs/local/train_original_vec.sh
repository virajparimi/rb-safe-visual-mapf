#!/bin/bash

#echo "[Action Needed] need to launch the debugger client"
#python -m debugpy \
#    --listen localhost:5678 \
#    --wait-for-client \
#    pud/algos/train_vec_pointenv.py configs/config_PointEnv.yaml

python pud/algos/train_vec_pointenv.py configs/config_PointEnv.yaml