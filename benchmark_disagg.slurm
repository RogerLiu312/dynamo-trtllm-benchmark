#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# CW worked llama 70b cmd sample
# sbatch --output=/home/rogliu/dynamo-benchmark/trtllm/250919/dynamo/components/backends/trtllm/performance_sweeps/%j_log.out --error=/home/rogliu/dynamo-benchmark/trtllm/250919/dynamo/components/backends/trtllm/performance_sweeps/%j_log.err --gpus-per-node 8 --nodes=2  --time=01:00:00 --partition=batch --account=general_perflab --job-name=dynamo-new-trtllm benchmark_disagg.slurm 1 8 1 8448 false 1 8 64 128 false 0.8 0 0 '32 64' 1 1 dynamo_disagg /lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_ci/artifacts/model/deepseek-r1_distill-llama_70b_pyt/safetensors_mode-instruct/hf-1772b07-bf16/ DeepSeek-R1-distill-70b nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime:0.5.0 128 128 8

# CW Workers DS-R1 cmd sample
# sbatch --gpus-per-node 8 --output=/home/rogliu/dynamo-benchmark/0.5.0/rogliu/dynamo-trtllm-benchmark/logs/%j_log.out --error=/home/rogliu/dynamo-benchmark/0.5.0/rogliu/dynamo-trtllm-benchmark/logs/%j_log.err --nodes=4  --time=00:20:00 --partition=batch --account=general_perflab --job-name=dynamo-new-trtllm benchmark_disagg.slurm 1 16 1 8448 true 1 16 16 16 true 0.7 0 0 '32 64' 2 2 dynamo_disagg  /lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_ci/artifacts/model/deepseek-r1_pyt/safetensors_mode-instruct/hf-56d4cbb-nim_fp8   DeepSeek-R1 nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime:0.5.0 128 128 8 /home/rogliu/dynamo-benchmark/0.5.0/rogliu/dynamo-trtllm-benchmark/logs

# ptyche worked cmd sample
# sbatch --output=./%j_log.out --error=./%j_log.err --nodes=4  --time=00:30:00 --partition=36x2-a01r --account=general_perflab --job-name=dynamo-new-trtllm benchmark_disagg.slurm 2 4 1 8448 true 2 4 64 128 true 0.7 0 0 '32 64' 2 2 dynamo_disagg /lustre/fsw/general_perflab/rogliu/gcn/builds/HuggingFace_Models_nvidia_DeepSeek_R1_0528_FP4_2/  DeepSeek-R1-FP4 nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime:0.5.1-rc0.pre3 128 128 4

set -x
MULTI_ROUND="${MULTI_ROUND:-8}"

# set MOUNT_DIR
MOUNT_DIR="${MOUNT_DIR:-${PWD}}"
CONTAINER_NAME=disaggr-test


STREAMING=true
# CTX_GPU_FRAC=0.75
# CACHE_TRANSCEIVER_MAX_NUM_TOKENS=8448

num_ctx_servers=$1
ctx_tp_size=$2
ctx_batch_size=$3
ctx_max_num_tokens=$4
ctx_enable_attention_dp=$5
num_gen_servers=$6
gen_tp_size=$7
gen_batch_size=$8
gen_max_num_tokens=$9
gen_enable_attention_dp=${10}
gen_gpu_memory_fraction=${11}
eplb_num_slots=${12}
mtp_size=${13}
concurrency_list=${14}
ctx_nodes=${15}
gen_nodes=${16}
kind=${17}
model_path=${18}
served_model_name=${19}
image=${20}
isl=${21}
osl=${22}
num_gpus_per_node=${23}
log_dir=${24}
benchmark_image="${25:-nvcr.io/nvidia/tritonserver:25.08-py3-sdk}"
server_ready_timeout=${26:-20}
ctx_gpu_memory_fraction=${27:-0.75}
cache_transceiver_max_num_tokens=${28:-8448}

ctx_max_seq_len=$((${isl} + 203))
gen_max_seq_len=$((${isl} + ${osl} + 203))

WORK_DIR=${MOUNT_DIR}
LOG_DIR=${log_dir:-${WORK_DIR}/${kind}-bm-${isl}-${osl}-${SLURM_JOB_ID}}
SCRIPTS_DIR=${WORK_DIR}
set_clock_cmd="bash ${SCRIPTS_DIR}/set_clock.sh"
mkdir -p ${LOG_DIR}
echo "trying to submit job"

sub_dir=${LOG_DIR}

echo "concurrency_list: ${concurrency_list}"

ctx_gpus=$((num_ctx_servers * ctx_tp_size))
gen_gpus=$((num_gen_servers * gen_tp_size))

echo "enable_attention_dp: ${ctx_enable_attention_dp}, ${gen_enable_attention_dp}, gpu_memory_fraction: ${gen_gpu_memory_fraction}"

enable_pdl=false
if [ "${gen_enable_attention_dp}" = "false" ]; then
    enable_pdl=true
    echo "enable_pdl: ${enable_pdl}"
    sub_dir=${LOG_DIR}
fi

full_logdir=${sub_dir}
artifacts_dir=${full_logdir}/genai_perf_artifacts
mkdir -p ${artifacts_dir}


# Set clock
# srun ${set_clock_cmd}

container_mounts=${MOUNT_DIR}:${MOUNT_DIR},${model_path}:${model_path},${LOG_DIR}:${LOG_DIR}

srun --overlap --ntasks $((ctx_nodes + gen_nodes)) --ntasks-per-node=1 --cpus-per-task=1 --export=full_logdir=${full_logdir} /bin/bash -c 'nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory,temperature.gpu,temperature.memory,power.draw,clocks.sm,clocks.mem,memory.total,memory.used --format=csv,noheader,nounits -lms 2000 | while IFS= read -r input || [ -n "$input" ] ; do timestamp=$(date +%s%3N); printf "%s.%s,%s\n" "${timestamp:0:10}" "${timestamp:10:3}" "${input}"; done  >> ${full_logdir}/${SLURM_JOB_ID}_${SLURM_NODEID}_${SLURMD_NODENAME}_nvsmi_output.txt & while true; do top -b -o -c %CPU | head -n 16 >> ${full_logdir}/${SLURM_JOB_ID}_${SLURM_NODEID}_${SLURMD_NODENAME}_top_output.txt; sleep 2; done & nvidia-smi dmon -s umt -d 2 >> ${full_logdir}/${SLURM_JOB_ID}_${SLURM_NODEID}_${SLURMD_NODENAME}_dmon.txt & mpstat 2 >> ${full_logdir}/${SLURM_JOB_ID}_${SLURM_NODEID}_${SLURMD_NODENAME}_cpu_output.txt & free -h -s 2 >> ${full_logdir}/${SLURM_JOB_ID}_${SLURM_NODEID}_${SLURMD_NODENAME}_free_output.txt' & 

# start the container
srun -l --container-image=${image} \
        --container-name=${CONTAINER_NAME} \
        --container-mounts=${container_mounts} \
        --mpi=pmix \
        echo "Container up - $(env | grep CPU)"

# generate the yaml file
srun -l --container-name=${CONTAINER_NAME} \
        --container-mounts=${container_mounts} \
        --mpi=pmix --overlap \
	-n 1 -N 1 \
        python3 ${SCRIPTS_DIR}/scripts/gen_yaml.py --config ${full_logdir}/config.yaml \
            --model ${model_path} \
            --num_ctx_servers ${num_ctx_servers} \
            --ctx_tp_size ${ctx_tp_size} \
            --ctx_batch_size ${ctx_batch_size} \
            --ctx_max_num_tokens ${ctx_max_num_tokens} \
            --ctx_max_seq_len ${ctx_max_seq_len} \
            --ctx_free_gpu_memory_fraction ${ctx_gpu_memory_fraction} \
            --cache_transceiver_max_num_tokens ${cache_transceiver_max_num_tokens} \
            --num_gen_servers ${num_gen_servers} \
            --gen_tp_size ${gen_tp_size} \
            --gen_batch_size ${gen_batch_size} \
            --gen_max_num_tokens ${gen_max_num_tokens} \
            --gen_max_seq_len ${gen_max_seq_len} \
            --gen_gpu_memory_fraction ${gen_gpu_memory_fraction} \
            --eplb_num_slots ${eplb_num_slots} \
            --gpus_per_node ${num_gpus_per_node} \
            $(if [ "${gen_enable_attention_dp}" = "true" ]; then echo "--gen_enable_attention_dp"; fi) \
            $(if [ "${ctx_enable_attention_dp}" = "true" ]; then echo "--ctx_enable_attention_dp"; fi) \
            $(if [ "${mtp_size}" -gt 0 ]; then echo "--mtp_size ${mtp_size}"; fi)

echo "YAML file generated."

nsys_on=""
# nsys_on=${full_logdir}

nodes=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))

export HEAD_NODE="${nodes[0]}"
export HEAD_NODE_IP="$(hostname -i)"
export ETCD_ENDPOINTS="${HEAD_NODE_IP}:2379"
export NATS_SERVER="nats://${HEAD_NODE_IP}:4222"

# Create a temporary file to store PIDs
PID_FILE=$(mktemp)
trap 'cleanup_and_exit' EXIT

cleanup_and_exit() {
    if [ -f "$PID_FILE" ]; then
        echo "Cleaning up spawned processes..."
        while read -r pid; do
            if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
                echo "Sending TERM to process $pid"
                kill -TERM "$pid" 2>/dev/null
                sleep 2
                if kill -0 "$pid" 2>/dev/null; then
                    echo "Process $pid still running, sending KILL"
                    kill -KILL "$pid" 2>/dev/null
                fi
            fi
        done < "$PID_FILE"
        rm -f "$PID_FILE"
    fi
}

# start the server
srun -l --container-name=${CONTAINER_NAME} \
        --container-mounts=${container_mounts} \
        --mpi=pmix --overlap -N 1 -n 1 \
	--container-env ETCD_ENDPOINTS,NATS_SERVER,HEAD_NODE_IP,HEAD_NODE \
        -w ${nodes[0]} \
        bash ${SCRIPTS_DIR}/scripts/start_frontend.sh &> ${full_logdir}/${SLURM_JOB_ID}_output_server.log &
SERVER_PID=$!
echo "$SERVER_PID" >> "$PID_FILE"

# wait for the server to start
sleep 10

PREFILL_COUNT=$(grep 'prefill_count:' "${full_logdir}/instance_config.yaml" | awk '{print $2}')
if [ -z "$PREFILL_COUNT" ]; then
    echo "Error: Failed to extract prefill_count from instance_config.yaml"
    exit 1
fi
echo "Prefill Count: $PREFILL_COUNT"

# start the prefill workers
prefill_pids=()
num_ctx_nodes=$((ctx_nodes/num_ctx_servers))
for ((i=1; i<=PREFILL_COUNT; i++)); do
  echo "Running Prefill Worker: ${i}"
  # Calculate ntasks and nodes for prefill: each node has 8 GPUs, so nodes = ceil(ctx_tp_size/8)
  prefill_node_list=()
  for ((j=0; j<num_ctx_nodes; j++)); do
    node_idx=$(((i-1)*num_ctx_nodes + j))
    prefill_node_list+=("${nodes[node_idx]}")
  done
  prefill_nodes_csv=$(IFS=, ; echo "${prefill_node_list[*]}")
  echo "Running Prefill Nodes: ${prefill_nodes_csv}"
  srun -l --container-name=${CONTAINER_NAME} \
      --container-mounts=${container_mounts} \
      --mpi=pmix --overlap -w ${prefill_nodes_csv} \
      --ntasks ${ctx_tp_size} \
      --ntasks-per-node ${num_gpus_per_node} \
      --nodes ${num_ctx_nodes} \
      bash ${SCRIPTS_DIR}/scripts/start_disagg_worker.sh ${full_logdir}/prefill_config.yaml "${enable_pdl}" ${ctx_gpus} ${nsys_on} ${served_model_name} ${model_path} 'prefill' &> ${full_logdir}/${SLURM_JOB_ID}_prefill_workers_${i}.log &
  prefill_pids+=($!)
  echo "$!" >> "$PID_FILE"
done

DECODE_COUNT=$(grep 'decode_count:' "${full_logdir}/instance_config.yaml" | awk '{print $2}')
if [ -z "$DECODE_COUNT" ]; then
    echo "Error: Failed to extract decode_count from instance_config.yaml"
    exit 1
fi
echo "Decode Count: $DECODE_COUNT"

num_gen_nodes=$((gen_nodes/num_gen_servers))
decode_start_idx=$((ctx_nodes))
for ((i=1; i<=DECODE_COUNT; i++)); do
  echo "Running Decode Worker: ${i}"
  decode_node_list=()
  for ((j=0; j<num_gen_nodes; j++)); do
    node_idx=$((decode_start_idx + (i-1)*num_gen_nodes + j))
    decode_node_list+=("${nodes[node_idx]}")
  done
  decode_nodes_csv=$(IFS=, ; echo "${decode_node_list[*]}")
  echo "Running Decode Nodes: ${decode_nodes_csv}"
  srun -l --container-name=${CONTAINER_NAME} \
      --container-mounts=${container_mounts} \
      --mpi=pmix \
      -w ${decode_nodes_csv} \
      --nodes ${num_gen_nodes} \
      --ntasks $gen_tp_size \
      --ntasks-per-node ${num_gpus_per_node} \
      --overlap \
      bash ${SCRIPTS_DIR}/scripts/start_disagg_worker.sh ${full_logdir}/decode_config.yaml "${enable_pdl}" ${ctx_gpus} ${nsys_on} ${served_model_name} ${model_path} 'decode' &> ${full_logdir}/${SLURM_JOB_ID}_decode_workers_${i}.log &
  echo "$!" >> "$PID_FILE"
done

total_gpus=$((ctx_gpus + gen_gpus))

sleep 20

# start the loadgen
srun -l --container-name=${CONTAINER_NAME} \
        --container-mounts=${container_mounts},${artifacts_dir}:${artifacts_dir} \
        --container-env HEAD_NODE_IP,HEAD_NODE,SCRIPTS_DIR \
        --mpi=pmix --overlap -N 1 -n 1 \
	-w ${nodes[0]} \
        bash ${SCRIPTS_DIR}/scripts/bench.sh ${served_model_name} ${MULTI_ROUND} ${num_gen_servers} "${concurrency_list}" ${STREAMING} ${full_logdir} ${total_gpus} ${artifacts_dir} ${model_path} ${isl} ${osl} ${kind} ${server_ready_timeout} > ${full_logdir}/${SLURM_JOB_ID}_bench.log 2>&1


# Cleanup will be handled by the EXIT trap
set +x