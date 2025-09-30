#! /bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -x

cmd=$1
mode=$2
model_name=$3
model_path=$4
nodes_per_worker=$5
head_node_ip=$6
node_rank=$7
tp_size=$8
deepep_config_path="${9:-scripts/deepep.json}"
mem_fraction=${10}
max_batch_size=${11}
unset UCX_TLS
echo " cmd: ${cmd}, mode: ${mode}, model_name: ${model_name}, model_path: ${model_path}"

# Validate mode argument
if [ "$mode" != "prefill" ] && [ "$mode" != "decode" ]; then
    echo "Error: mode must be 'prefill' or 'decode', got '$mode'"
    print_usage
fi


# Construct command based on mode
if [ "$mode" = "prefill" ]; then
    if [ "$cmd" = "dynamo" ]; then
        # H100 dynamo prefill command
        python3 -m dynamo.sglang \
            --model-path ${model_path} \
            --served-model-name ${model_name} \
            --skip-tokenizer-init \
            --disaggregation-mode prefill \
            --disaggregation-transfer-backend nixl \
            --disaggregation-bootstrap-port 30001 \
            --dist-init-addr ${head_node_ip}:27500 \
            --nnodes ${nodes_per_worker} \
            --node-rank ${node_rank} \
            --tp-size ${tp_size} \
            --dp-size ${tp_size} \
            --enable-dp-attention \
            --decode-log-interval 1 \
            --enable-deepep-moe \
            --page-size 1 \
            --trust-remote-code \
            --moe-dense-tp-size 1 \
            --enable-dp-lm-head \
            --disable-radix-cache \
            --watchdog-timeout 1000000 \
            --enable-two-batch-overlap \
            --deepep-mode normal \
            --mem-fraction-static ${mem_fraction} \
            --deepep-config ${deepep_config_path} \
            --ep-num-redundant-experts 32 \
            --ep-dispatch-algorithm dynamic \
            --eplb-algorithm deepseek
    elif [ "$cmd" = "sglang" ]; then
        # H100 sglang prefill command
        python3 -m sglang.launch_server \
            --model-path ${model_path} \
            --served-model-name ${model_name} \
            --disaggregation-transfer-backend nixl \
            --disaggregation-mode prefill \
            --dist-init-addr ${head_node_ip}:27500 \
            --nnodes ${nodes_per_worker} \
            --node-rank ${node_rank} \
            --tp-size ${tp_size} \
            --dp-size ${tp_size} \
            --enable-dp-attention \
            --decode-log-interval 1 \
            --enable-deepep-moe \
            --page-size 1 \
            --host 0.0.0.0 \
            --trust-remote-code \
            --moe-dense-tp-size 1 \
            --enable-dp-lm-head \
            --disable-radix-cache \
            --watchdog-timeout 1000000 \
            --enable-two-batch-overlap \
            --deepep-mode normal \
            --mem-fraction-static ${mem_fraction} \
            --ep-num-redundant-experts 32 \
            --ep-dispatch-algorithm dynamic \
            --eplb-algorithm deepseek \
            --deepep-config ${deepep_config_path}
    fi
elif [ "$mode" = "decode" ]; then
    if [ "$cmd" = "dynamo" ]; then
        # H100 dynamo decode command
        python3 -m dynamo.sglang \
            --model-path ${model_path} \
            --served-model-name ${model_name} \
            --skip-tokenizer-init \
            --disaggregation-mode decode \
            --disaggregation-transfer-backend nixl \
            --disaggregation-bootstrap-port 30001 \
            --dist-init-addr ${head_node_ip}:27500 \
            --nnodes ${nodes_per_worker} \
            --node-rank ${node_rank} \
            --tp-size ${tp_size} \
            --dp-size ${tp_size} \
            --enable-dp-attention \
            --decode-log-interval 1 \
            --enable-deepep-moe \
            --page-size 1 \
            --trust-remote-code \
            --moe-dense-tp-size 1 \
            --enable-dp-lm-head \
            --disable-radix-cache \
            --watchdog-timeout 1000000 \
            --enable-two-batch-overlap \
            --deepep-mode low_latency \
            --mem-fraction-static ${mem_fraction} \
            --ep-num-redundant-experts 32 \
            --cuda-graph-bs ${max_batch_size}
    elif [ "$cmd" = "sglang" ]; then
        # H100 sglang decode command
        python3 -m sglang.launch_server \
            --model-path ${model_path} \
            --disaggregation-transfer-backend nixl \
            --disaggregation-mode decode \
            --dist-init-addr ${head_node_ip}:27500 \
            --nnodes ${nodes_per_worker} \
            --node-rank ${node_rank} \
            --tp-size ${tp_size} \
            --dp-size ${tp_size} \
            --enable-dp-attention \
            --decode-log-interval 1 \
            --enable-deepep-moe \
            --page-size 1 \
            --host 0.0.0.0 \
            --trust-remote-code \
            --moe-dense-tp-size 1 \
            --enable-dp-lm-head \
            --disable-radix-cache \
            --watchdog-timeout 1000000 \
            --enable-two-batch-overlap \
            --deepep-mode low_latency \
            --mem-fraction-static ${mem_fraction} \
            --ep-num-redundant-experts 32 \
            --cuda-graph-bs ${max_batch_size}
    fi
fi


