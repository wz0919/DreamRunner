import argparse
from ast import arg
import os
import platform
import re
import ast
import warnings
from typing import Optional

import torch
from copy import deepcopy

from einops import rearrange, repeat
from torch import Tensor
from torch.nn.functional import interpolate
from tqdm import trange
import random

from transformers import T5EncoderModel, T5Tokenizer
from diffusers.models import AutoencoderKLCogVideoX, CogVideoXTransformer3DModel
from diffusers import CogVideoXDDIMScheduler, CogVideoXDPMScheduler, CogVideoXPipeline

from MotionDirector_train_cogvideox import export_to_video, handle_memory_attention, transformer_and_text_g_c, freeze_models
from utils.lora_handler_cogvideox import LoraHandler

from utils.lora_handler_cogvideox_inference_multilora import LoraHandler_MultiLoRA
# from utils.lora_handler import LoraHandler
from utils.ddim_utils import ddim_inversion
import imageio

import numpy as np
import cv2
import cv2
import numpy as np
import imageio

def create_video_from_bbox_list(bbox_list, width=640, height=480, fps=1, video_name='video.mp4'):
    import cv2
    import numpy as np
    import imageio

    # Create a video writer using imageio
    writer = imageio.get_writer(video_name, fps=fps, codec='libx264', quality=8)

    # Predefined set of five distinct colors
    color_palette = [
        (255, 0, 0),  # Red
        (0, 255, 0),  # Green
        (0, 0, 255),  # Blue
        (255, 255, 0),  # Cyan
        (255, 0, 255)  # Magenta
    ]

    def draw_frame(frame_data):
        img = np.ones((height, width, 3), dtype=np.uint8) * 255  # White background

        for idx, (description, bbox) in enumerate(frame_data):
            entity, motion, caption = description
            # Calculate bounding box coordinates
            left, top, right, bottom = bbox
            x1, y1 = int(left * width), int(top * height)
            x2, y2 = int(right * width), int(bottom * height)
            
            # Select color from the palette
            color = color_palette[idx % len(color_palette)]

            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # Prepare for caption drawing
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.0
            text_thickness = 2
            max_text_height = y2 - y1  # Maximum height for text within bbox
            max_text_width = x2 - x1  # Maximum width for text within bbox

            # Split text into lines
            words = caption.split(' ')
            lines = []
            current_line = ''
            for word in words:
                test_line = current_line + word + ' '
                (line_width, _) = cv2.getTextSize(test_line, font, font_scale, text_thickness)[0]
                if line_width <= (max_text_width - 10):  # Allow margin
                    current_line = test_line
                else:
                    lines.append(current_line.strip())  # Save the current line
                    current_line = word + ' '  # Start a new line
            if current_line:
                lines.append(current_line.strip())  # Add the last line

            # Adjust font size if text height exceeds the box height
            while True:
                total_text_height = len(lines) * (cv2.getTextSize("Test", font, font_scale, text_thickness)[0][1] + 5)
                if total_text_height <= max_text_height - 10:
                    break
                font_scale -= 0.1  # Reduce font size
                if font_scale < 0.3:  # Minimum font size
                    font_scale = 0.3
                    break

            # Draw text within the bounding box
            text_x = x1 + 5  # Margin from left
            text_y = y1 + 5 + cv2.getTextSize("Test", font, font_scale, text_thickness)[0][1]  # Start from top
            for line in lines:
                if text_y + cv2.getTextSize(line, font, font_scale, text_thickness)[0][1] <= y2 - 5:
                    cv2.putText(img, line, (text_x, text_y), font, font_scale, color, text_thickness, cv2.LINE_AA)
                    text_y += cv2.getTextSize(line, font, font_scale, text_thickness)[0][1] + 5
                else:
                    break  # Stop if text exceeds box height

        return img

    # Write each frame to the video
    for frame_data in bbox_list:
        frame_img = draw_frame(frame_data)
        for _ in range(fps):
            writer.append_data(cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for imageio

    # Close the video writer
    writer.close()
    print(f"Bbox video saved as {video_name}")


def path_text_pair(arg):
    try:
        path, text = arg.split(', ')
        return (path, text)
    except ValueError:
        raise argparse.ArgumentTypeError("Arguments must be in the format 'path,text'")

def load_primary_models(pretrained_model_path):
    noise_scheduler = CogVideoXDDIMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
    tokenizer = T5Tokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
    text_encoder = T5EncoderModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
    vae = AutoencoderKLCogVideoX.from_pretrained(pretrained_model_path, subfolder="vae")
    transformer = CogVideoXTransformer3DModel.from_pretrained(pretrained_model_path, subfolder="transformer")

    return noise_scheduler, tokenizer, text_encoder, vae, transformer

def initialize_pipeline(
    model: str,
    device: str = "cuda",
    xformers: bool = False,
    sdp: bool = False,
    spatial_lora_paths: list = [],
    temporal_lora_paths: list = [],
    lora_rank: int = 64,
    spatial_lora_scale: float = 1.0,
    temporal_lora_scale: float = 1.0,
    scheduler_name='ddim',
):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        scheduler, tokenizer, text_encoder, vae, transformer = load_primary_models(model)

    # Freeze any necessary models
    freeze_models([vae, text_encoder, transformer])

    # Enable xformers if available
    handle_memory_attention(xformers, sdp, transformer)

    if all([os.path.exists(spatial_lora_path) for spatial_lora_path in spatial_lora_paths]):
        lora_manager_spatial = LoraHandler_MultiLoRA(
            version="cloneofsimo",
            use_transformer_lora=True,
            use_text_lora=False,
            save_for_webui=False,
            only_for_webui=False,
            transformer_replace_modules=[("CogVideoXBlock",'only_even_blocks')],
            text_encoder_replace_modules=None,
            lora_bias=None
        )

        transformer_lora_params, transformer_negation = lora_manager_spatial.add_lora_to_model(
            True, transformer, lora_manager_spatial.transformer_replace_modules, 0, spatial_lora_paths, r=lora_rank, scale=spatial_lora_scale)
    else:
        print('spatial lora path not exist')

    if all([os.path.exists(temporal_lora_path) for temporal_lora_path in temporal_lora_paths]):
        lora_manager_temporal = LoraHandler_MultiLoRA(
            version="cloneofsimo",
            use_transformer_lora=True,
            use_text_lora=False,
            save_for_webui=False,
            only_for_webui=False,
            transformer_replace_modules=[("CogVideoXBlock",'only_odd_blocks')],
            text_encoder_replace_modules=None,
            lora_bias=None
        )

        transformer_lora_params, transformer_negation = lora_manager_temporal.add_lora_to_model(
            True, transformer, lora_manager_temporal.transformer_replace_modules, 0, temporal_lora_paths, r=lora_rank, scale=temporal_lora_scale)
    else:
        print('temporal lora path not exist')

    transformer.eval()
    text_encoder.eval()
    transformer_and_text_g_c(transformer, text_encoder, False, False)

    pipe = CogVideoXPipeline.from_pretrained(
        pretrained_model_name_or_path=model,
        scheduler=scheduler,
        tokenizer=tokenizer,
        text_encoder=text_encoder.to(device=device, dtype=torch.half),
        vae=vae.to(device=device, dtype=torch.half),
        transformer=transformer.to(device=device, dtype=torch.half),
    )
    if scheduler_name == 'ddim':
        pipe.scheduler = CogVideoXDDIMScheduler.from_config(pipe.scheduler.config)
        print('using DDIMScheduler')
    elif scheduler_name == 'dpm':
        pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config)
        print('using DPMSolverMultistepScheduler')
    else:
        print('no scheduler')

    return pipe

def get_prompt_ids(prompt, tokenizer):
    prompt_ids = tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
    ).input_ids

    return prompt_ids

def inverse_video(pipe, latents, num_steps):
    print('using DDIMScheduler for inverse videos')
    ddim_inv_scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    ddim_inv_scheduler.set_timesteps(num_steps)

    ddim_inv_latent = ddim_inversion(
        pipe, ddim_inv_scheduler, video_latent=latents.to(pipe.device),
        num_inv_steps=num_steps, prompt="")[-1]
    return ddim_inv_latent



def prepare_input_latents(
    pipe: CogVideoXPipeline,
    batch_size: int,
    num_frames: int,
    height: int,
    width: int,
    latents_path:str,
    noise_prior: float
):
    # initialize with random gaussian noise
    # scale = pipe.vae_scale_factor
    shape = (batch_size, pipe.transformer.config.in_channels, num_frames // pipe.vae_scale_factor_temporal, height // pipe.vae_scale_factor_spatial, width // pipe.vae_scale_factor_spatial)
    if noise_prior > 0.:
        cached_latents = torch.load(latents_path)
        if 'inversion_noise' not in cached_latents:
            latents = inverse_video(pipe, cached_latents['latents'].unsqueeze(0), 50).squeeze(0)
        else:
            latents = torch.load(latents_path)['inversion_noise'].unsqueeze(0)
        if latents.shape[0] != batch_size:
            latents = latents.repeat(batch_size, 1, 1, 1, 1)
        if latents.shape != shape:
            latents = interpolate(rearrange(latents, "b c f h w -> (b f) c h w", b=batch_size), (height // pipe.vae_scale_factor_spatial, width // pipe.vae_scale_factor_spatial), mode='bilinear')
            latents = rearrange(latents, "(b f) c h w -> b c f h w", b=batch_size)
        noise = torch.randn_like(latents, dtype=torch.half)
        latents = (noise_prior) ** 0.5 * latents + (1 - noise_prior) ** 0.5 * noise
    else:
        latents = torch.randn(shape, device='cuda', dtype=torch.half).cpu() # diffuser same latents

    return latents


def encode(pipe: CogVideoXPipeline, pixels: Tensor, batch_size: int = 8):
    nf = pixels.shape[2]
    pixels = rearrange(pixels, "b c f h w -> (b f) c h w")

    latents = []
    for idx in trange(
        0, pixels.shape[0], batch_size, desc="Encoding to latents...", unit_scale=batch_size, unit="frame"
    ):
        pixels_batch = pixels[idx : idx + batch_size].to(pipe.device, dtype=torch.half)
        latents_batch = pipe.vae.encode(pixels_batch).latent_dist.sample()
        latents_batch = latents_batch.mul(pipe.vae.config.scaling_factor).cpu()
        latents.append(latents_batch)
    latents = torch.cat(latents)

    latents = rearrange(latents, "(b f) c h w -> b c f h w", f=nf)

    return latents

def extract_info(input_text):
    # Adjust regex to handle both actual newlines (\n) and escaped newlines (\\n)
    background_regex = r'Background:\s*(.*?)(?:\\n|\n)'
    background_match = re.search(background_regex, input_text)

    # Extract the background if found
    if background_match:
        background = background_match.group(1).strip()  # Extract the background
    else:
        background = None  # In case no background is found

    print("Background:", background)

    # Use regular expressions to find all frame data
    # The new regex will capture all content after Frame_X: until the next frame
    frame_regex = r'(Frame_\d+:\s*(\[\[.*?\]\](?:,\s*\[\[.*?\]\])*))'

    # Find all frame data using the regex
    matches = re.findall(frame_regex, input_text, re.DOTALL)

    print('Bbox list:')
    # Convert the found matches (which are strings) into Python lists using ast.literal_eval
    output_list = []
    for match in matches:
        frame_data = match[1]  # The second group contains the frame list
        frame_list = ast.literal_eval(f'[{frame_data}]')  # Add brackets to make it a valid list
        output_list.append(frame_list)
        print(frame_list)

    return background, output_list

def get_alloc(bbox_list, background, spatial_loras, temporal_loras):
    
    #preparing mask

    # Define the dimensions for bounding box
    frame_height = 30
    frame_width = 45
    mask_height = 12

    # Initialize an empty dictionary to store masks for each caption
    caption_masks = {}

    # Define a function to draw bounding boxes on the mask using rounded coordinates
    def draw_bbox_on_mask(mask, bbox, frame_idx):
        left, top, right, bottom = bbox
        # Use round() to round the bounding box coordinates
        x1, y1 = round(left * frame_width), round(top * frame_height)
        x2, y2 = round(right * frame_width), round(bottom * frame_height)

        # Each frame occupies two rows in the mask
        start_row = 2 * frame_idx
        mask[start_row:start_row + 2, y1:y2, x1:x2] = 1  # Fill the region with 1 to represent the entity

    # Iterate through each frame to process the data
    for frame_idx, frame_data in enumerate(bbox_list):
        for description, bbox in frame_data:
            entity, motion, caption = description
            # If the caption is not already in the dictionary, create a new mask for it
            if caption not in caption_masks:
                caption_masks[caption] = torch.zeros((mask_height, frame_height, frame_width), dtype=torch.uint8)
            
            # Draw the bounding box for the caption in the respective mask
            draw_bbox_on_mask(caption_masks[caption], bbox, frame_idx)

    prompts = []
    num_prompts = len(caption_masks) + 1
    mask = torch.ones((226*num_prompts+16200,226*num_prompts+16200))
    conditon_mask_list = [torch.zeros((226, 226))]*num_prompts
    
    for c_idx, caption in enumerate(caption_masks):
        prompts.append(caption)
        c_mask_list_current = deepcopy(conditon_mask_list)
        c_mask_list_current[c_idx] = torch.ones((226, 226))
        mask_c = torch.cat(c_mask_list_current + [caption_masks[caption].flatten().repeat(226,1)], dim=1)
        mask[226*c_idx:226*(c_idx+1)] = mask_c

    c_idx += 1
    c_mask_list_current = deepcopy(conditon_mask_list)
    c_mask_list_current[c_idx] = torch.ones((226, 226))
    mask_c = torch.cat(c_mask_list_current + [(mask[:226*(num_prompts-1),226*num_prompts:].sum(0) == 0).repeat(226,1)*1], dim=1)
    mask[226*c_idx:226*(c_idx+1)] = mask_c
    prompts.append(background)

    mask_T = mask.T
    mask[226*num_prompts:,:226*num_prompts] = mask_T[226*num_prompts:,:226*num_prompts]
    mask =  mask.cuda()

    # preparing lora masks
    spatial_lora_masks = []
    for item in spatial_loras:
        charactor = item[1]
        spatial_lora_mask = torch.zeros((1,226*num_prompts+16200,1)).cuda()
        for pidx, prompt in enumerate(prompts):
            if charactor in prompt:
                spatial_lora_mask += mask[pidx*226][None,:,None]
        spatial_lora_masks.append(spatial_lora_mask)

    temporal_lora_masks = []
    for item in temporal_loras:
        motion = item[1]
        temporal_lora_mask = torch.zeros((1,226*num_prompts+16200,1)).cuda()
        for pidx, prompt in enumerate(prompts):
            if motion.split(' ')[0] in prompt:
                temporal_lora_mask += mask[pidx*226][None,:,None]
        temporal_lora_masks.append(temporal_lora_mask)

    return mask, prompts, spatial_lora_masks, temporal_lora_masks

@torch.inference_mode()
def inference(
    model: str,
    prompt: str,
    negative_prompt: Optional[str] = None,
    width: int = 256,
    height: int = 256,
    num_frames: int = 24,
    num_steps: int = 50,
    guidance_scale: float = 15,
    device: str = "cuda",
    xformers: bool = False,
    sdp: bool = False,
    spatial_lora_paths: list = [],
    temporal_lora_paths: list = [],
    lora_rank: int = 64,
    spatial_lora_scale: float = 1.0,
    temporal_lora_scale: float = 1.0,
    seed: Optional[int] = None,
    latents_path: str="",
    noise_prior: float = 0.,
    repeat_num: int = 1,
    scheduler: str = 'ddim',
    mask: torch.tensor = None,
    spatial_lora_masks: list = [],
    temporal_lora_masks: list = [],
    seed_log_dir: str=""
):
    with torch.autocast(device, dtype=torch.half):
        # prepare models
        pipe = initialize_pipeline(model, device, xformers, sdp, spatial_lora_paths, temporal_lora_paths, lora_rank,
                                   spatial_lora_scale, temporal_lora_scale, scheduler)

        # pipe.enable_model_cpu_offload()
        # pipe.enable_sequential_cpu_offload()

        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()

        print('schduler', pipe.scheduler)
        for i in range(repeat_num):
            if seed is None:
                random_seed = random.randint(100, 10000000)
                torch.manual_seed(random_seed)
            else:
                random_seed = seed
                torch.manual_seed(seed)

            # prepare input latents
            init_latents = prepare_input_latents(
                pipe=pipe,
                batch_size=1,
                num_frames=num_frames,
                height=height,
                width=width,
                latents_path=latents_path,
                noise_prior=noise_prior
            ).permute(0,2,1,3,4)

            spatial_exists = all([os.path.exists(spatial_lora_path) for spatial_lora_path in spatial_lora_paths])
            temporal_exists = all([os.path.exists(temporal_lora_path) for temporal_lora_path in temporal_lora_paths])

            for idx in range(30):
                if (idx+1)%2 == 0 and spatial_exists: # spatial Lora
                    pipe.transformer.transformer_blocks[idx].attn1.to_q.masks = spatial_lora_masks
                    pipe.transformer.transformer_blocks[idx].attn1.to_k.masks = spatial_lora_masks
                    pipe.transformer.transformer_blocks[idx].attn1.to_v.masks = spatial_lora_masks
                    pipe.transformer.transformer_blocks[idx].attn1.to_out[0].masks = spatial_lora_masks
                    pipe.transformer.transformer_blocks[idx].ff.net[0].proj.masks = spatial_lora_masks
                    pipe.transformer.transformer_blocks[idx].ff.net[2].masks = spatial_lora_masks

                if (idx+1)%2 == 1 and temporal_exists: # spatial Lora
                    pipe.transformer.transformer_blocks[idx].attn1.to_q.masks = temporal_lora_masks
                    pipe.transformer.transformer_blocks[idx].attn1.to_k.masks = temporal_lora_masks
                    pipe.transformer.transformer_blocks[idx].attn1.to_v.masks = temporal_lora_masks
                    pipe.transformer.transformer_blocks[idx].attn1.to_out[0].masks = temporal_lora_masks
                    pipe.transformer.transformer_blocks[idx].ff.net[0].proj.masks = temporal_lora_masks
                    pipe.transformer.transformer_blocks[idx].ff.net[2].masks = temporal_lora_masks
                pipe.transformer.transformer_blocks[idx].mask = mask.bool()[None,None,...].cuda()

            # init_latents = torch.cat([init_latents[:,6:],init_latents[:,:6]],dim=1)
            with torch.no_grad():
                video_frames = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    width=width,
                    height=height,
                    num_frames=num_frames,
                    num_inference_steps=num_steps,
                    guidance_scale=guidance_scale,
                    latents=init_latents,
                    # use_dynamic_cfg=True
                ).frames[0]

            # =========================================
            # ========= write outputs to file =========
            # =========================================
            os.makedirs(args.output_dir, exist_ok=True)

            # save to mp4
            export_to_video(video_frames, f"{out_name}", args.fps)

            if not os.path.isfile(seed_log_dir):
                os.makedirs(seed_log_dir.split('seed_log')[0]+'seed_log', exist_ok=True)
            with open(seed_log_dir, 'a') as file:
                file.write("%s, %s\n"%(out_name, random_seed))  

    return video_frames


if __name__ == "__main__":
    import decord

    decord.bridge.set_bridge("torch")

    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, required=True,
                        help="HuggingFace repository or path to model checkpoint directory")
    parser.add_argument("-p", "--prompt", nargs='+', help='prompts to generate', default=None)
    parser.add_argument("-pr", "--prefix", type=str, default=None, help="Prefix added to the final output name.")
    parser.add_argument("-n", "--negative-prompt", nargs='+', type=str, default=None, help="Text prompt to condition against")
    parser.add_argument("-o", "--output_dir", type=str, default="./outputs/inference", help="Directory to save output video to")
    parser.add_argument("-B", "--batch-size", type=int, default=1, help="Batch size for inference")
    parser.add_argument("-W", "--width", type=int, default=384, help="Width of output video")
    parser.add_argument("-H", "--height", type=int, default=384, help="Height of output video")
    parser.add_argument("-T", "--num-frames", type=int, default=16, help="Total number of frames to generate")
    parser.add_argument("-s", "--num-steps", type=int, default=30, help="Number of diffusion steps to run per frame.")
    parser.add_argument("-g", "--guidance-scale", type=float, default=12, help="Scale for guidance loss (higher values = more guidance, but possibly more artifacts).")
    parser.add_argument("-sch", "--scheduler", type=str, default='dpm', help="diffusion schduler, dpm as default (similar to zeroscopev2).")
    parser.add_argument("-f", "--fps", type=int, default=8, help="FPS of output video")
    parser.add_argument("-d", "--device", type=str, default="cuda", help="Device to run inference on (defaults to cuda).")
    parser.add_argument("-x", "--xformers", action="store_true", help="Use XFormers attnetion, a memory-efficient attention implementation (requires `pip install xformers`).")
    parser.add_argument("-S", "--sdp", action="store_true", help="Use SDP attention, PyTorch's built-in memory-efficient attention implementation.")
    parser.add_argument("-slp", "--spatial_path_folders", nargs='+', type=path_text_pair, default=None, help="Paths and corresponding texts to Low Rank Adaptation checkpoint file (defaults to empty string, which uses no LoRA).")
    parser.add_argument("-tlp", "--temporal_path_folders", nargs='+', type=path_text_pair, default=None,
                        help="Paths and corresponding texts to Low Rank Adaptation checkpoint file (defaults to empty string, which uses no LoRA).")
    parser.add_argument("-lr", "--lora_rank", type=int, default=32, help="Size of the LoRA checkpoint's projection matrix (defaults to 32).")
    parser.add_argument("-sps", "--spatial_path_scale", type=float, default=1.0, help="Scale of spatial LoRAs.")
    parser.add_argument("-tps", "--temporal_path_scale", type=float, default=1.0, help="Scale of temporal LoRAs.")
    parser.add_argument("-r", "--seed", type=int, default=None, help="Random seed to make generations reproducible.")
    parser.add_argument("-np", "--noise_prior", type=float, default=0., help="Scale of the influence of inversion noise.")
    parser.add_argument("-ci", "--checkpoint_index", type=str, default="default",
                        help="The index of checkpoint, such as 300.")
    parser.add_argument("-rn", "--repeat_num", type=int, default=1,
                        help="How many results to generate with the same prompt.")
    parser.add_argument("-pl", "--plan", type=str, default=None, help="The generated plans for LLMs.")
    parser.add_argument("-vn", "--video_name", type=str, default=None, help="Outputed video name.")
    parser.add_argument("--generate_mask_videos", action="store_true", help="If provided, this will generate mask videos.")
    args = parser.parse_args()
    # fmt: on

    # =========================================
    # ====== validate and prepare inputs ======
    # =========================================

    if args.plan is not None:
        background, bbox_list = extract_info(args.plan)
        mask, prompts, spatial_lora_masks, temporal_lora_masks = get_alloc(bbox_list, background, args.spatial_path_folders, args.temporal_path_folders)
        print('spatial_lora_masks',[torch.sum(m) for m in spatial_lora_masks])
        print('temporal_lora_masks',[torch.sum(m) for m in temporal_lora_masks])
        args.prompt = prompts
        args.spatial_path_folders = [item[0] for item in args.spatial_path_folders]
        args.temporal_path_folders = [item[0] for item in args.temporal_path_folders]

    out_name = f"{args.output_dir}/"
    out_name += args.video_name

    source_dir = args.output_dir.split('/')[:-1]
    dimension_name = args.output_dir.split('/')[-1]
    seed_log_dir = os.path.join(*source_dir,'seed_log',dimension_name+'.txt')
    mask_dir= os.path.join(*source_dir,'masks',dimension_name,args.video_name)

    os.makedirs(os.path.join(*source_dir,'masks',dimension_name), exist_ok=True)

    print('seed log dir',seed_log_dir)
    print('output dir',out_name)
    print('bbox dir',mask_dir)

    # args.prompt = [prompt] * args.batch_size
    # if args.negative_prompt is not None:
    #     args.negative_prompt = [args.negative_prompt] * args.batch_size

    # assert os.path.exists(args.spatial_path_folder)
    # assert os.path.exists(args.temporal_path_folder)

    if args.noise_prior > 0:
        latents_folder = f"{os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(args.temporal_path_folder))))}/cached_latents"
        latents_path = f"{latents_folder}/{random.choice(os.listdir(latents_folder))}"
        assert os.path.exists(latents_path)
    else:
        latents_path = None

    # =========================================
    # ============= sample videos =============
    # =========================================

    video_frames = inference(
        model=args.model,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        width=args.width,
        height=args.height,
        num_frames=args.num_frames,
        num_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
        device=args.device,
        xformers=args.xformers,
        sdp=args.sdp,
        spatial_lora_paths=args.spatial_path_folders,
        temporal_lora_paths=args.temporal_path_folders,
        lora_rank=args.lora_rank,
        spatial_lora_scale=args.spatial_path_scale,
        temporal_lora_scale=args.temporal_path_scale,
        seed=args.seed,
        latents_path=latents_path,
        noise_prior=args.noise_prior,
        repeat_num=args.repeat_num,
        scheduler=args.scheduler,
        mask=mask,
        spatial_lora_masks=spatial_lora_masks,
        temporal_lora_masks=temporal_lora_masks,
        seed_log_dir=seed_log_dir
    )

    if args.plan is not None and args.generate_mask_videos:
        create_video_from_bbox_list(bbox_list,width=720,height=480,fps=8,video_name=mask_dir)