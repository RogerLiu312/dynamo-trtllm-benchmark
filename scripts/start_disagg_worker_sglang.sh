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
gpus_per_node=${12}
unset UCX_TLS
echo " cmd: ${cmd}, \
mode: ${mode}, \
model_name: ${model_name}, \
model_path: ${model_path}, \
nodes_per_worker: ${nodes_per_worker}, \
head_node_ip: ${head_node_ip}, \
node_rank: ${node_rank}, \
tp_size: ${tp_size}, \
deepep_config_path: ${deepep_config_path}, \
mem_fraction: ${mem_fraction}, \
max_batch_size: ${max_batch_size}, \
gpus_per_node: ${gpus_per_node}"

# Validate mode argument
if [ "$mode" != "prefill" ] && [ "$mode" != "decode" ]; then
    echo "Error: mode must be 'prefill' or 'decode', got '$mode'"
    exit 1
fi

# Construct command based on mode
if [ "$gpus_per_node" = "4" ]; then
    if [ "$mode" = "prefill" ]; then
        # GB200 dynamo prefill command
        DYN_SKIP_SGLANG_LOG_FORMATTING=1 \
        SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=2048 \
        MC_TE_METRIC=true \
        SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE=100000 \
        SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=100000 \
        SGLANG_DISAGGREGATION_WAITING_TIMEOUT=100000 \
        SGLANG_MOONCAKE_CUSTOM_MEM_POOL=True \
        MC_FORCE_MNNVL=1 \
        NCCL_MNNVL_ENABLE=1 \
        NCCL_CUMEM_ENABLE=1 \
        SGLANG_USE_MESSAGE_QUEUE_BROADCASTER=0 \
        SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK=1 \
        PYTHONUNBUFFERED=1 \
        python3 -m dynamo.sglang.worker \
            --served-model-name ${model_name} \
            --model-path ${model_path} \
            --skip-tokenizer-init \
            --trust-remote-code \
            --disaggregation-mode prefill \
            --dist-init-addr ${head_node_ip}:27500 \
            --disaggregation-bootstrap-port 30001 \
            --nnodes ${nodes_per_worker} \
            --node-rank ${node_rank} \
            --tp-size ${tp_size} \
            --dp-size ${tp_size} \
            --enable-dp-attention \
            --host 0.0.0.0 \
            --decode-log-interval 1 \
            --max-running-requests 12288 \
            --context-length 9600 \
            --disable-radix-cache \
            --enable-deepep-moe \
            --deepep-mode low_latency \
            --ep-dispatch-algorithm dynamic \
            --moe-dense-tp-size 1 \
            --enable-dp-lm-head \
            --disable-shared-experts-fusion \
            --ep-num-redundant-experts 32 \
            --eplb-algorithm deepseek \
            --attention-backend cutlass_mla \
            --watchdog-timeout 1000000 \
            --disable-cuda-graph \
            --chunked-prefill-size 16384 \
            --max-total-tokens 65536 \
            --deepep-config ${deepep_config_path} \
            --stream-interval 50 \
            --log-level debug
    elif [ "$mode" = "decode" ]; then
        # GB200 dynamo decode command
        DYN_SKIP_SGLANG_LOG_FORMATTING=1 \
        SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=512 \
        MC_TE_METRIC=true \
        SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE=100000 \
        SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=100000 \
        SGLANG_DISAGGREGATION_WAITING_TIMEOUT=100000 \
        SGLANG_HACK_SEQ_BOOTSTRAP_ROOM=1 \
        SGLANG_MOONCAKE_CUSTOM_MEM_POOL=True \
        NCCL_MNNVL_ENABLE=1 \
        MC_FORCE_MNNVL=1 \
        NCCL_CUMEM_ENABLE=1 \
        SGLANG_USE_MESSAGE_QUEUE_BROADCASTER=0 \
        SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK=1 \
        PYTHONUNBUFFERED=1 \
        python3 -m dynamo.sglang.decode_worker \
            --served-model-name ${model_name} \
            --model-path ${model_path} \
            --skip-tokenizer-init \
            --trust-remote-code \
            --disaggregation-mode decode \
            --dist-init-addr ${head_node_ip}:27500 \
            --disaggregation-bootstrap-port 30001 \
            --nnodes ${nodes_per_worker} \
            --node-rank ${node_rank} \
            --tp-size ${tp_size} \
            --dp-size ${tp_size} \
            --enable-dp-attention \
            --host 0.0.0.0 \
            --decode-log-interval 1 \
            --max-running-requests 36864 \
            --context-length 9600 \
            --disable-radix-cache \
            --enable-deepep-moe \
            --deepep-mode low_latency \
            --moe-dense-tp-size 1 \
            --enable-dp-lm-head \
            --cuda-graph-bs 1 2 4 8 16 24 32 40 48 56 64 80 96 112 128 160 192 224 256 320 384 448 512 \
            --cuda-graph-max-bs ${max_batch_size} \
            --disable-shared-experts-fusion \
            --ep-num-redundant-experts 32 \
            --ep-dispatch-algorithm static \
            --eplb-algorithm deepseek \
            --attention-backend cutlass_mla \
            --watchdog-timeout 1000000 \
            --chunked-prefill-size 36864 \
            --stream-interval 50 \
            --mem-fraction-static ${mem_fraction}
    fi
else
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
fi
