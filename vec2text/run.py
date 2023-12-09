import transformers
import experiments
# from experiments import experiment_from_args
# from run_args import DataArguments, ModelArguments, TrainingArguments
import torch
import os

def main():
    parser = transformers.HfArgumentParser(
        (experiments.ModelArguments, experiments.DataArguments, experiments.TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    '''
    torch.save(
        data_args, os.path.join(training_args.output_dir, "data_args.bin")
    )
    torch.save(
        model_args,
        os.path.join(training_args.output_dir, "model_args.bin"),
    )
    '''
    experiment = experiments.experiment_from_args(model_args, data_args, training_args)
    experiment.run()


if __name__ == "__main__":
    main()
