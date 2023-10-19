#!/bin/bash

set -euo pipefail

region="southcentralus"        # eastus, southcentralus, westus2
cluster="spch-sing-tts-sc"     # spch-sing-tts-sc, spch-sing-ttsprod-sc
num_nodes=1                    # 1 GPU node
gpus_per_node=2              # each node with 1 GPU
memory_size=16                 # 16GB
gpu_type="V100"                # V100 GPU
interconnect_type="Empty"      # "Empty", "IB", "NvLink", "xGMI", "IB-xGMI", "NvLink-xGMI"
sla_tier="Standard"             # Basic, Standard or Premium
distributed="true"
dist_config="ddp"

project_name="tortoise"  # project name (e.g., tacotron/fastspeech)
exp_name="sydney_AM_finetuning"
cyber_eo=False

# if the packages not installed in the docker, you can install them here or set it as ""
extra_env_setup_cmd="pip install --user -r requirements.laxed.txt"
extra_env_setup_cmd=${extra_env_setup_cmd}" && pip install -e ."

config="../experiments/EXAMPLE_gpt_sydney_singularity.yml"
cp ${config} ./${project_name}_${exp_name}.yml

gpu_ids="["
for ((i=0; i<${gpus_per_node}-1; i++)); do
    gpu_ids=${gpu_ids}"${i},"
done
gpu_ids=${gpu_ids}"$((gpus_per_node-1))]"
# replace the gpu_ids in the config file with the gpu_ids of the current job
sed -i "s/gpu_ids: \[0\]/gpu_ids: ${gpu_ids}/g" ./${project_name}_${exp_name}.yml

cmd="python train.py -opt ./${project_name}_${exp_name}.yml --launcher=pytorch"

python ../Submitter/utils/amlt_submit.py \
    --service "singularity" --region ${region} --cluster ${cluster} \
    --num-nodes ${num_nodes} --gpus-per-node ${gpus_per_node} \
    --memory-size ${memory_size} --gpu-type ${gpu_type} --sla-tier ${sla_tier} \
    --interconnect-type ${interconnect_type} --distributed ${distributed} \
    --image-registry "azurecr.io" --image-repo "azurespeechdockers" \
    --key-vault-name "exawatt-philly-ipgsp" --docker-username "default-pull" \
    --image-name "torchtts:nvidia_pytorch2.0.1_38360346" \
    --data-container-name "philly-ipgsp" --model-container-name "philly-ipgsp" \
    --extra-env-setup-cmd "${extra_env_setup_cmd}" --local-code-dir "$(pwd)" \
    --amlt-project ${project_name} --exp-name ${exp_name} \
    --run-cmd "${cmd}" \
    --enable-cyber-eo ${cyber_eo}