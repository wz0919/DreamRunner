trap 'kill 0' SIGINT

model="/your_workspace/pretrained_models/CogVideoX-2b"
width=720
height=480
numframes=48

flags="--model ${model}
       --width ${width}
       --height ${height}
       --num-frames ${numframes}"

input="Background: mysterious forest with luminous flowers and moonlight\nFrame_1: [[\"witch\", \"walking\", \"A [v4] girl witch is walking among the luminous flowers in her mysterious forest\"], [0.6, 0.0, 1.0, 0.8]], [[\"flowers\", \"none\", \"luminous flowers in the mysterious forest\"], [0.0, 0.6, 1.0, 1.0]], [[\"cat\", \"none\", \"A [v6] cat is arranging flowers near the witch among the luminous flowers\"], [0.0, 0.2, 0.4, 0.8]]\nFrame_2: [[\"witch\", \"walking\", \"A [v4] girl witch is walking among the luminous flowers in her mysterious forest\"], [0.6, 0.0, 1.0, 0.8]], [[\"flowers\", \"none\", \"luminous flowers in the mysterious forest\"], [0.0, 0.6, 1.0, 1.0]], [[\"cat\", \"none\", \"A [v6] cat is arranging flowers near the witch among the luminous flowers\"], [0.0, 0.2, 0.4, 0.8]]\nFrame_3: [[\"witch\", \"walking\", \"A [v4] girl witch is walking among the luminous flowers in her mysterious forest\"], [0.5, 0.0, 0.9, 0.8]], [[\"flowers\", \"none\", \"luminous flowers in the mysterious forest\"], [0.0, 0.6, 1.0, 1.0]], [[\"cat\", \"none\", \"A [v6] cat is arranging flowers near the witch among the luminous flowers\"], [0.0, 0.2, 0.4, 0.8]]\nFrame_4: [[\"witch\", \"whispering\", \"A [v4] girl witch is waving hands o the luminous flowers\"], [0.5, 0.0, 0.9, 0.8]], [[\"flowers\", \"none\", \"luminous flowers in the mysterious forest\"], [0.0, 0.6, 1.0, 1.0]], [[\"cat\", \"none\", \"A [v6] cat is arranging flowers near the witch among the luminous flowers\"], [0.0, 0.2, 0.4, 0.8]]\nFrame_5: [[\"witch\", \"whispering\", \"A [v4] girl witch is waving hands o the luminous flowers\"], [0.5, 0.2, 0.9, 0.8]], [[\"flowers\", \"none\", \"luminous flowers in the mysterious forest\"], [0.0, 0.6, 1.0, 1.0]], [[\"cat\", \"none\", \"A [v6] cat is arranging flowers near the witch among the luminous flowers\"], [0.0, 0.2, 0.4, 0.8]]\nFrame_6: [[\"witch\", \"whispering\", \"A [v4] girl witch is waving hands o the luminous flowers\"], [0.5, 0.2, 0.9, 0.8]], [[\"flowers\", \"none\", \"luminous flowers in the mysterious forest\"], [0.0, 0.6, 1.0, 1.0]], [[\"cat\", \"none\", \"A [v6] cat is arranging flowers near the witch among the luminous flowers\"], [0.0, 0.2, 0.4, 0.8]]|/your_workspace/MotionDirector/outputs/train_story_65_motions/train_person_is_walking/checkpoints/checkpoint-120/temporal/lora, walking|/your_workspace/MotionDirector/outputs/train_story_65_motions/train_person_is_waving/checkpoints/checkpoint-300/temporal/lora, waving|/your_workspace/MotionDirector/outputs/train_story_65_motions/train_person_is_arranging_flowers/checkpoints/checkpoint-360/temporal/lora, arranging flowers|/your_workspace/MotionDirector/outputs/train_image/train_cogvideox_A_[v4]_girl_witch_mp_16f_every20_3:4_ratio_rerun2_{2}/checkpoints/checkpoint-160/spatial/lora, A [v4] girl witch|/your_workspace/MotionDirector/outputs/train_image/train_cogvideox_A_[v6]_cat_mp_rerun_h100_{3}/checkpoints/checkpoint-160/spatial/lora, A [v6] cat"
name=example_input


IFS='|' read -r -a splits <<< "${a}"
plan="${splits[0]}"
temporal_path_folder1="${splits[1]}"
temporal_path_folder2="${splits[2]}"
temporal_path_folder3="${splits[3]}"
spatial_path_folder1="${splits[4]}"
spatial_path_folder2="${splits[5]}"

echo temporal_path_folder1, $temporal_path_folder1
echo temporal_path_folder2, $temporal_path_folder2
echo temporal_path_folder3, $temporal_path_folder3
echo spatial_path_folder1, $spatial_path_folder1
echo spatial_path_folder2, $spatial_path_folder2

out_dir="/your_workspace/MotionDirector/outputs/inference/stories"
gpus=(0 1 2 3 4 5 6 7)
do
for i in "${!gpus[@]}"
do  
    gpu="${gpus[$i]}"
    seed="${seeds[$i]}"
    echo "Using GPU: $gpu, Seed: $seed"
    (
        CUDA_VISIBLE_DEVICES=$gpu python MotionDirector_inference_any_alloc_cogvideox.py  $flags \
        --plan "$plan" \
        --spatial_path_folder "$spatial_path_folder1" "$spatial_path_folder2" \
        --temporal_path_folder "$temporal_path_folder1" "$temporal_path_folder2" "$temporal_path_folder3" \
        --noise_prior 0. \
        --num-steps 50 \
        --guidance-scale 9.0 \
        --scheduler 'dpm' \
        --output_dir "${out_dir}/examples/${name}" \
        --generate_mask_videos \
    ) &
    done
    wait
done

wait