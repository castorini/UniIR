#!/bin/bash
export PYSERINI_CACHE=/store2/scratch/sjupadhy

#!/bin/bash
bs=96
for ((i = 0; i < 10126; i += $bs)); do
    next_index=$((i + bs))
    CUDA_VISIBLE_DEVICES=0 python inference.py --prompt_file caption-prompt.txt --model_name llava \
    --image_path /mnt/users/s8sharif/M-BEIR/mbeir_images/mscoco_images/val2014 \
    --index $i"_"$next_index
done
