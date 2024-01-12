#!/bin/bash

# Copyright  2022  Microsoft (author: Ke Wang)

set -euo pipefail

region="southcentralus"        # eastus, southcentralus, westus2
cluster="spch-sing-tts-sc"     # spch-sing-tts-sc, spch-sing-ttsprod-sc
num_nodes=1                    # 1 GPU node
gpus_per_node=4                # each node with 4 GPUs
memory_size=16                 # 16GB
gpu_type="V100"                # V100 GPU
interconnect_type="Empty"      # "Empty", "IB", "NvLink", "xGMI", "IB-xGMI", "NvLink-xGMI", "IB-NvLink"
sla_tier="Premium"             # Basic, Standard or Premium
distributed="true"             # enable distributed training or not

project_name="amlt_test_singularity"    # project name (e.g., tacotron/fastspeech)
exp_name="test_mnist_sing_4gpu_nvidia"  # experimental name (e.g., Evan/Guy/Aria)

# if the packages not installed in the docker, you can install them here
extra_env_setup_cmd="pip install --upgrade pip" # or extra_env_setup_cmd=""

# ======================= parameters for running script =======================
# All parameters are optional except "--distributed" which will be parsed by
# utils/amlt_submit.py. Others will be parsed by your own script.
dist_method="torch"      # torch or horovod
data_dir="/datablob"     # will download data to /datablob/{alias}/Data/MNIST
                         # or read data from /datablob/{alias}/Data/MNIST
extra_params="--distributed ${distributed}"
extra_params=${extra_params}" --dist-method ${dist_method}"
extra_params=${extra_params}" --data-dir ${data_dir}"
# add some personal config
# extra_params=${extra_params}" --config ${config}"
# ============================================================================

python -u utils/amlt_submit.py \
  --service "singularity" --region ${region} --cluster ${cluster} \
  --num-nodes ${num_nodes} --gpus-per-node ${gpus_per_node} \
  --memory-size ${memory_size} --gpu-type ${gpu_type} --sla-tier ${sla_tier} \
  --interconnect-type ${interconnect_type} --distributed ${distributed} \
  --image-registry "azurecr.io" --image-repo "sramdevregistry" \
  --key-vault-name "exawatt-philly-ipgsp" --docker-username "tts-itp-user" \
  --image-name "pytorch:2.0.1-py39-cuda11.7-ubuntu20.04" \
  --data-container-name "philly-ipgsp" --model-container-name "philly-ipgsp" \
  --extra-env-setup-cmd "${extra_env_setup_cmd}" --local-code-dir "$(pwd)" \
  --amlt-project ${project_name} --exp-name ${exp_name} \
  --run-cmd "python -u train.py" --extra-params "${extra_params}" \
  --enable-cyber-eo "false"
