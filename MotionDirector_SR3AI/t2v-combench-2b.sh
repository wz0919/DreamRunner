#!/bin/bash

jsonl_name=$1
dimension=$(echo "$jsonl_name" | cut -d'_' -f2-)
out_dir="T2V-CompBench/videos/cogvideox-2b-plan"
plan=plan
mkdir -p "$out_dir/masks"
mkdir -p "$out_dir/seed_log"
model="../pretrained_models/CogVideoX-2b"
width=720
height=480
numframes=48

gpus=(0)

flags="--model ${model}
       --width ${width}
       --height ${height}
       --num-frames ${numframes}"

plans=()
video_names=()
while IFS= read -r line; do
    description=$(echo "$line" | jq -r '.[1]')
    video_name=$(echo "$line" | jq -r '.[0]')
    plans+=("$description")
    video_names+=("$video_name")
done < "./t2v-combench/$plan/$jsonl_name.jsonl"

echo "Plans list: ${plans[@]}"
echo "Video name list: ${video_names[@]}"

length=${#plans[@]}
echo 'length', $length

length_gpus=${#gpus[@]}
echo "${out_dir}/${dimension}"
for ((i=0; i<length; i+=$length_gpus))
do
  for idx in "${!gpus[@]}"
    do
      (
        gpu="${gpus[$idx]}"
        sample_idx=$(($idx+$i))
        plan="${plans[$sample_idx]}"
        video_name="${video_names[$sample_idx]}"
        echo $video_name
        CUDA_VISIBLE_DEVICES=$gpu python -u t2v_combench_MotionDirector_inference_sr3ai_cogvideox.py $flags \
        --plan "$plan" \
        --spatial_path_folder "none, none" \
        --temporal_path_folder "none, none" "none, none" \
        --noise_prior 0. \
        --num-steps 50 \
        --guidance-scale 9.0 \
        --scheduler 'dpm' \
        --output_dir "${out_dir}/${dimension}" \
        --generate_mask_videos \
        --video_name $video_name
      ) &
    done
  wait
done