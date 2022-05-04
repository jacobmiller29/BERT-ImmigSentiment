### This script continutes the standard BERT pre-training process (language-modeling task) on the cable news text
# load in the relevant packages
from simpletransformers.language_modeling import LanguageModelingModel, LanguageModelingArgs
import torch

# Check if a CUDA device is available, if so use the GPU, if not use CPU
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
    usecuda = True
else:
    print("We will use the CPU.")
    usecuda = False

# configure the model
print("Configure model for training")

model_args = LanguageModelingArgs(
    dataset_type="simple",
    sliding_window=False,
    num_train_epochs=2,
    output_dir="../models/pretraining/",
    overwrite_output_dir=True,
    max_seq_length=512,
    do_lower_case=True,
    wandb_project="cable-news-bert",
    train_batch_size=24,
    eval_batch_size=512,
    save_steps=-1,
    save_model_every_epoch=True
    )

model = LanguageModelingModel(
    "roberta",
    "distilroberta-base",
    args=model_args,
    use_cuda=usecuda
)

train_file = "../raw-data/pretraining_training_data.txt"
test_file = "../raw-data/pretraining_test_data.txt"

# get a baseline loss rate on the cable news test
print("Getting a baseline evaluation")
result_baseline = model.eval_model(test_file)
eval_txt = open("../models/pretraining/eval_results_baseline.txt", "w")
eval_txt.write(str(result_baseline))
eval_txt.close()
# train on the language-modeling task for 2 epochs
print("Training model on media segments")
model.train_model(train_file, eval_file=test_file)
# evaluate the trained model
print("Evaluating model")
result = model.eval_model(test_file)
eval_txt = open("../models/pretraining/eval_results_2epoch.txt", "w")
eval_txt.write(str(result))
eval_txt.close()
print("Done!")
