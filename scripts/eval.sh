for run_name in "CLIP_SF_k1_fewshot_random" "CLIP_SF_k0_zeroshot" "CLIP_SF_k5_fewshot_rag" "CLIP_SF_k1_fewshot_rag" "CLIP_SF_k5_fewshot_random"; do
        CUDA_VISIBLE_DEVICES=5 python eval.py \
        --candidate_path /mnt/users/s8sharif/M-BEIR/cand_pool/global/mbeir_union_test_cand_pool.jsonl \
        --result_dir /mnt/users/s8sharif/UniIR/llm_outputs/llava_outputs/${run_name} \
        --retrieval_jsonl_path /mnt/users/s8sharif/UniIR/data/UniIR/retrieval_results/CLIP_SF/Large/Instruct/InBatch/retrieved_candidates/mbeir_mscoco_task3_union_pool_test_k10_retrieved.jsonl \
        --calculate_retriever_metrics 
    done

for run_name in "BLIP_FF_k1_fewshot_rag" "BLIP_FF_k5_fewshot_random" "BLIP_FF_k1_fewshot_random" "BLIP_FF_k5_fewshot_rag"; do
        CUDA_VISIBLE_DEVICES=5 python eval.py \
        --candidate_path /mnt/users/s8sharif/M-BEIR/cand_pool/global/mbeir_union_test_cand_pool.jsonl \
        --result_dir /mnt/users/s8sharif/UniIR/llm_outputs/llava_outputs/${run_name} \
        --retrieval_jsonl_path /mnt/users/s8sharif/UniIR/data/UniIR/retrieval_results/BLIP_FF/Large/Instruct/InBatch/retrieved_candidates/mbeir_mscoco_task3_union_pool_test_k10_retrieved.jsonl \
        --calculate_retriever_metrics 
    done