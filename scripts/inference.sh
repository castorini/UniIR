#!/bin/bash
export PYSERINI_CACHE=/store2/scratch/s8sharif

# TODO: make the bash file work with flag values for different runs
bs=512
image_path=/mnt/users/s8sharif/M-BEIR/mbeir_images/mscoco_images/val2014
image_count=$(ls $image_path | wc -l)
echo $image_count
k=1

for ((i = 0; i < $image_count; i += $bs)); do
    next_index=$((i + bs))
    CUDA_VISIBLE_DEVICES=7 python inference.py --prompt_file caption-prompt-with-examples.txt \
    --k=$k --model_name gpt \
    --image_path $image_path \
    --output_dir /store2/scratch/s8sharif/UniIR/llm_outputs \
    --retrieved_results_path "/store2/scratch/s8sharif/UniIR/data/UniIR/retrieval_results/CLIP_SF/Large/Instruct/InBatch/run_files/mbeir_mscoco_task3_union_pool_test_k10_run_2024-03-27 15:28:49.276449.jsonl" \
    --index $i"_"$next_index --retriever_name "CLIP_SF"
done



# for ((i = 0; i < $image_count; i += $bs)); do
#     next_index=$((i + bs))
#     CUDA_VISIBLE_DEVICES=7 python inference.py --prompt_file caption-prompt-without-examples.txt \
#     --k=$k --model_name llava \
#     --image_path $image_path \
#     --output_dir /store2/scratch/s8sharif/UniIR/llm_outputs \
#     --retrieved_results_path "/store2/scratch/s8sharif/UniIR/data/UniIR/retrieval_results/CLIP_SF/Large/Instruct/InBatch/run_files/mbeir_mscoco_task3_union_pool_test_k10_run_2024-03-27 15:28:49.276449.jsonl" \
#     --index $i"_"$next_index --retriever_name "CLIP_SF"
# done
