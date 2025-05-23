# UniMoCo: Unified Modality Completion for Robust Multi-Modal Embeddings

This repository is the official implementation of UniMoCo: Unified Modality Completion for Robust Multi-Modal Embeddings. 

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Training

To train the model(s), run:

```train
torchrun --nproc_per_node=8 --master_port=22447 --max_restarts=0 train.py \
    --model_name <path/to/Phi2Vit> --bf16 --pooling last \
    --dataset_name TIGER-Lab/MMEB-train \
    --subset_name ImageNet_1K N24News HatefulMemes VOC2007 SUN397 OK-VQA A-OKVQA DocVQA InfographicsVQA ChartQA Visual7W VisDial CIRR VisualNews_i2t VisualNews_t2i MSCOCO_t2i MSCOCO_i2t NIGHTS WebQA MSCOCO \
    --num_sample_per_subset 50000 \
    --image_dir <path/to/MMEB-train> \
    --max_len 256 --num_crops 4 --output_dir <path/to/output> --logging_steps 1 \
    --lr_scheduler_type linear --learning_rate 6e-5 --max_steps 2000 \
    --warmup_steps 200 --save_steps 1000 --normalize True \
    --temperature 0.02 --per_device_train_batch_size 128 \
    --lora --lora_r 8 \
    --grad_cache True --gc_q_chunk_size 2 --gc_p_chunk_size 2 
```

UniMoCo uses a pre-trained vision-language backbone (e.g., Phi-3.5V) with dual image encoders initialized from identical weights. Training leverages the publicly available MMEB dataset.

## Evaluation

To evaluate on MMEB-eval, run:

```eval
python eval.py --lora --model_name </path/to/Phi2Vit> --checkpoint_path </path/to/unimoco-ckpt>   \
  --encode_output_path <path/to/output>  \
  --model_backbone phi35v\
  --num_crops 4 --max_len 256 \
  --pooling last --normalize True \
  --dataset_name </path/to/MMEB-eval> \
  --subset_name ImageNet-1K N24News HatefulMemes VOC2007 SUN397 Place365 ImageNet-A ImageNet-R ObjectNet Country211 VisDial CIRR VisualNews_t2i VisualNews_i2t MSCOCO_t2i MSCOCO_i2t NIGHTS WebQA FashionIQ Wiki-SS-NQ OVEN EDIS OK-VQA A-OKVQA DocVQA InfographicsVQA ChartQA Visual7W ScienceQA VizWiz GQA TextVQA MSCOCO RefCOCO RefCOCO-Matching Visual7W-Pointing \
  --dataset_split test --per_device_eval_batch_size 16 \
  --image_dir </path/to/MMEB-eval>
```

## Pre-trained Models

Pre-trained models will be made available after publication to maintain anonymity.

## Results

Our model achieves the following performance on MMEB-eval:

| Models               | Classification | VQA  | Retrieval | Grounding | (T+I,T) | (T,T+I) | (T+I,T+I) | IND  | OOD  | Overall |
|----------------------|----------------|------|-----------|-----------|---------|---------|-----------|------|------|---------|
| # of Datasets →      | 10             | 10   | 12        | 4         | 22      | 6       | 8         | 20   | 16   | 36      |
| CLIP                | 42.8           | 9.1  | 53.0      | 51.8      | 29.8    | 62.1    | 41.6      | 37.1 | 38.7 | 37.8    |
| OpenCLIP            | 47.8           | 10.9 | 52.3      | 53.3      | 33.1    | 57.9    | 44.2      | 39.3 | 40.2 | 39.7    |
| BLIP2               | 27.0           | 4.2  | 33.9      | 47.0      | 15.7    | 43.1    | 37.9      | 25.3 | 25.1 | 25.2    |
| SigLIP              | 40.3           | 8.4  | 31.6      | 59.5      | 27.0    | 44.6    | 49.0      | 32.3 | 38.0 | 34.8    |
| UniIR               | 42.1           | 15.0 | 60.1      | 62.2      | 32.5    | 58.2    | 59.7      | 44.7 | 40.4 | 42.8    |
| *Δ w/o fine-tune*   | ↑12.8          | ↑43.2| ↑4.9      | ↑20.2     | ↑26.5   | ↑15.4   | ↑5.4      | ↑23.5| ↑18.0| ↑20.4   |
| CLIP-FT             | 50.0           | 27.0 | 55.3      | 64.8      | 40.3    | 64.7    | 49.3      | 52.2 | 38.9 | 47.0    |
| OpenCLIP-FT         | 51.4           | 30.8 | 58.1      | 66.3      | 43.1    | 66.8    | 54.4      | 56.9 | 40.4 | 49.6    |
| VLM2VEC (Phi-3.5V) | 54.8           | 54.9 | 62.3      | 79.5      | 56.1    | 72.6    | 61.5      | 66.5 | 52.0 | 60.1    |
| **UniMoCo (Phi-3.5V)** | 55.0        | 58.2 | 63.2      | **82.4**  | 57.7    | 72.8    | 64.1      | 68.2 | 53.5 | 61.7    |
| **UniMoCo (Qwen2-VL-7B)** | **62.6**  | 55.5 | **65.0**  | 78.2      | **59.6**| **73.6**| **65.1**  | 67.0 | 58.4 | **63.2**|
| *Δ w/ fine-tune*    | ↑7.8           | ↑3.3 | ↑2.7      | ↑2.9      | ↑3.5    | ↑1.0    | ↑3.6      | ↑1.7 | ↑6.4 | ↑3.1    |