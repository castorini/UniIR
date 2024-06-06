#!/bin/bash
export CACHE_DIR=/store2/scratch/s8sharif

# TODO: make the bash file work with flag values for different runs
bs=1000
query_path=/mnt/users/s8sharif/UniIR/data/UniIR/retrieval_results/CLIP_SF/Large/Instruct/InBatch/retrieved_candidates/mbeir_mscoco_task3_union_pool_test_k10_retrieved.jsonl
image_count=$(wc -l < $query_path)
echo $image_count

for ((i = 0; i < $image_count; i += $bs)); do
    next_index=$((i + bs))
    echo "batch "$i"_"$next_index":"
    CUDA_VISIBLE_DEVICES=3 python caption_generation_inference.py \
    --prompt_mode fewshot_rag \
    --max_output_tokens 400 \
    --base_mbeir_path /mnt/users/s8sharif/M-BEIR \
    --candidates_file_path /mnt/users/s8sharif/M-BEIR/cand_pool/global/mbeir_union_test_cand_pool.jsonl \
    --prompt_file gpt-caption-prompt-with-examples.txt \
    --k 1 \
    --model_name gpt \
    --index $i"_"$next_index \
    --output_dir /mnt/users/s8sharif/UniIR/llm_outputs \
    --retrieved_results_path $query_path \
    --retriever_name "CLIP_SF"
done

for ((i = 0; i < $image_count; i += $bs)); do
    next_index=$((i + bs))
    echo "batch "$i"_"$next_index":"
    CUDA_VISIBLE_DEVICES=3 python caption_generation_inference.py \
    --prompt_mode fewshot_rag \
    --max_output_tokens 400 \
    --base_mbeir_path /mnt/users/s8sharif/M-BEIR \
    --candidates_file_path /mnt/users/s8sharif/M-BEIR/cand_pool/global/mbeir_union_test_cand_pool.jsonl \
    --prompt_file gpt-caption-prompt-with-examples.txt \
    --k 5 \
    --model_name gpt \
    --index $i"_"$next_index \
    --output_dir /mnt/users/s8sharif/UniIR/llm_outputs \
    --retrieved_results_path $query_path \
    --retriever_name "CLIP_SF"
done
