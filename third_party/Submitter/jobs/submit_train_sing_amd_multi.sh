#!/bin/bash

# Copyright  2023  Microsoft (author: Ke Wang)

set -euo pipefail

region="rrlab"                 # rrlab (rrlab is near to wus2 region)
cluster="cogsvc-sing-amd-vc01" # cogsvc-sing-amd-vc01
num_nodes=1                    # 1 GPU node
gpus_per_node=1                # each node with 1 GPU
memory_size=32                 # 32GB
gpu_type="MI100"               # MI100 GPU
interconnect_type="Empty"      # "Empty", "IB", "NvLink", "xGMI", "IB-xGMI", "NvLink-xGMI", "IB-NvLink"
sla_tier="Standard"            # Basic, Standard or Premium
distributed="false"            # enable distributed training or not

project_name="amlt_test_singularity"  # project name (e.g., tacotron/fastspeech)
exp_name="test_mnist_sing_1gpu_amd"   # experimental name (e.g., Evan/Guy/Jessa)

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

# running job twice with different name
jobs_params=""
for i in $(seq 1 2); do
  job_name="${exp_name}_"${i}
  jobs_params="${jobs_params}${job_name}&&&${extra_params};;;"
done

preemptible="false"  # if true, can access resources outside your quota

python -u utils/amlt_submit.py \
  --service "singularity" --region ${region} --cluster ${cluster} \
  --num-nodes ${num_nodes} --gpus-per-node ${gpus_per_node} \
  --memory-size ${memory_size} --gpu-type ${gpu_type} --sla-tier ${sla_tier} \
  --interconnect-type ${interconnect_type} --distributed ${distributed} \
  --image-registry "azurecr.io" --image-repo "sramdevregistry" \
  --key-vault-name "exawatt-philly-ipgsp" --docker-username "tts-itp-user" \
  --image-name "pytorch:2.0.1-py38-rocm5.4-ubuntu20.04" \
  --data-container-name "philly-ipgsp" --model-container-name "philly-ipgsp" \
  --extra-env-setup-cmd "${extra_env_setup_cmd}" --local-code-dir "$(pwd)" \
  --amlt-project ${project_name} --exp-name ${exp_name} \
  --run-cmd "python -u train.py" --extra-params "${jobs_params}" \
  --enable-cyber-eo "false"