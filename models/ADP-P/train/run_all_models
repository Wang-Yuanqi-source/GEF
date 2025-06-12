#!/bin/bash

folders=(
    "stereovision2" 
)

# create models directors
mkdir -p models

for folder in "${folders[@]}"; do
    # 创建子文件夹
    mkdir -p "models/$folder"
    chmod +x "models/$folder/run_new_gnn.sh"

    cd "models/$folder"
    nohup bsub -I ./run_new_gnn.sh &
    cd ../..
done

echo "All jobs are submitted"
