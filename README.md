# DreamRunner: Fine-Grained Compositional Story-to-Video Generation with Retrieval-Augmented Motion Adaptation

[![Project Website](https://img.shields.io/badge/Project-Website-blue)](https://zunwang1.github.io/DreamRunner)  [![arXiv](https://img.shields.io/badge/arXiv-2411.16657-b31b1b.svg)](https://arxiv.org/pdf/2411.16657)   

#### [Zun Wang](https://zunwang1.github.io/), [Jialu Li](https://jialuli-luka.github.io/), [Han Lin](https://hl-hanlin.github.io/), [Jaehong Yoon](https://jaehong31.github.io), [Mohit Bansal](https://www.cs.unc.edu/~mbansal/)
<br>
<img width="950" src="files/teaser.gif"/>
<br>

#### Code coming soon!

## ToDos
- [x] Release the inference code on T2V-ComBench.
- [ ] Release the code for retrieving videos and training character and motion loras.
- [ ] Release the inference code for storytelling video genetation.

## Setup

### Environment Setup 
```shell
conda create -n dreamrunner python==3.10
conda activate dreamrunner
pip install -r requirements.txt 
```

### Download Models 
DreamRunner is implemented using CogVideoX-2B. You can download it [here](/https://huggingface.co/THUDM/CogVideoX-2b) and put it to `pretrained_models/CogVideoX-2b`.

## Running the Code

### T2V-Combench

#### Inference
We provide the plans we used for T2V-ComBench in `MotionDirector_SR3AI/t2v-combench/plan`.
You can specify the GPUs you want use in `MotionDirector_SR3AI/t2v-combench-2b.sh` for parallel inference.
Then directly Infer 600 videos on 6 dimensions of T2V-ComBnech with the following script
```
cd MotionDirector_SR3AI
bash run_bench_2b.sh
```
The generated videos will be saved at `MotionDirector_SR3AI/T2V-CompBench`.

#### Evaluation
Please follow [T2V-ComBench](https://github.com/KaiyueSun98/T2V-CompBench) for evaluating the generated videos.

### Storytell Video Generation
#### Coming soon!

# Citation

If you find our project useful in your research, please cite the following paper:

```
@article{wang2024dreamrunner,
  author  = {Wang, Zun and Li, Jialu and Lin, Han and Yoon, Jaehong and Bansal, Mohit},
  title   = {DreamRunner: Fine-Grained Compositional Story-to-Video Generation with Retrieval-Augmented Motion Adaptation},
  journal = {arXiv preprint arXiv:2411.16657},
  year    = {2024},
  url     = {https://arxiv.org/abs/2411.16657}
}
``
