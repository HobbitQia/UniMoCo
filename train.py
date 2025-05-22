# Adapted from Tevatron code
import logging
import sys

from transformers import AutoTokenizer, AutoProcessor
from transformers import LlavaNextProcessor
from transformers import (
    HfArgumentParser,
)

from src.dataset import TrainDataset
from src.collator import TrainCollator
from src.arguments import ModelArguments, DataArguments, TrainingArguments
from src.model import MMEBModel
from src.trainer import MMEBTrainer, GradCacheTrainer
import wandb
import torch
import torch.distributed as dist

from src.vlm_backbone.phi3_v.processing_phi3_v import Phi3VProcessor

logger = logging.getLogger(__name__)


def main():
    # a hack for torch.distributed.launch: https://github.com/huggingface/transformers/issues/22171
    for arg in sys.argv:
        if arg.startswith("--local-rank="):
            rank = arg.split("=")[1]
            sys.argv.remove(arg)
            sys.argv.append('--local_rank')
            sys.argv.append(rank)
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))

    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments

    if model_args.model_backbone == "llava":
        print("111111")
        processor = LlavaNextProcessor.from_pretrained(
            model_args.processor_name if model_args.processor_name else model_args.model_name,
            trust_remote_code=True)
        processor.tokenizer.padding_side = "left"
    elif model_args.model_backbone == "phi35v":
        print("2222222")
        processor = Phi3VProcessor.from_pretrained(
            model_args.processor_name if model_args.processor_name else model_args.model_name,
            trust_remote_code=True)
        processor.tokenizer.padding_side = "right"
    else:
        print("3333333")
        processor = AutoProcessor.from_pretrained(
            model_args.processor_name if model_args.processor_name else model_args.model_name,
            trust_remote_code=True,
            num_crops=model_args.num_crops
        )
        processor.tokenizer.padding_side = "right"

    train_dataset = TrainDataset(data_args, model_args)
    collator = TrainCollator(data_args, model_args, processor)

    template = "Image modality is missing in this case. So we use a t2i model to generate enerate a highly detailed and visually descriptive text prompt for an image that encapsulates the instruction and the query. Characters <|endoftext|> are just place-holders. Query: \n"

    tmp = processor.tokenizer(template, None, return_tensors="pt", truncation=True)
    another_tmp = processor.tokenizer("[PAD]", None, return_tensors="pt", truncation=True)

    trainer_cls = GradCacheTrainer if training_args.grad_cache else MMEBTrainer
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
    )
    train_dataset.trainer = trainer
    trainer.train()
    trainer.save_model(training_args.output_dir)

    if trainer.is_world_process_zero():
        processor.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
