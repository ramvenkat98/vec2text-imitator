from experiments import experiment_from_args
from run_args import DataArguments, ModelArguments, TrainingArguments

import copy
from typing import List

import torch
import transformers

import vec2text
from vec2text.models.model_utils import device
from vec2text import analyze_utils

def invert_embeddings(
    embeddings: torch.Tensor,
    inversion_trainer: vec2text.trainers.InversionTrainer,
    num_steps: int = None,
) -> List[str]:

    inversion_trainer = vec2text.trainers.InversionTrainer(
        model=inversion_model,
        train_dataset=None,
        eval_dataset=None,
        data_collator=transformers.DataCollatorForSeq2Seq(
            inversion_model.tokenizer,
            label_pad_token_id=-100,
        ),
    )
    inversion_trainer.model.eval()
    gen_kwargs = copy.copy(corrector.gen_kwargs)
    gen_kwargs["min_length"] = 1
    gen_kwargs["max_length"] = 50

    regenerated = inversion_trainer.generate(
        inputs={
            "frozen_embeddings": embeddings,
        },
        generation_kwargs=gen_kwargs,
    )
    output_strings = corrector.tokenizer.batch_decode(
        regenerated, skip_special_tokens=True
    )
    return output_strings

def load_inversion_model():
    '''
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    experiment = experiment_from_args(model_args, data_args, training_args)
    experiment.evaluate()
    '''
    experiment, trainer = analyze_utils.load_experiment_and_trainer('saves/gtr-1/checkpoint-109000')
    eval_dataset = trainer.eval_dataset['nq']
    trainer.args.per_device_eval_batch_size = 16
    trainer.sequence_beam_width = 1
    trainer.num_gen_recursive_steps = 0
    trainer.evaluate(eval_dataset = eval_dataset)

if __name__ == "__main__":
    load_inversion_model()

