### Conduct a WandB training sweep with the DistilBERT model to find out the best
### hyperparameters: learning rate and number of epochs
# load in required libraries
from simpletransformers.classification import ClassificationModel, ClassificationArgs
import pandas as pd
import torch
import wandb
import logging
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import minmax_scale
from sklearn.metrics import r2_score

# check if a CUDA device is available, if so use the GPU, if not use CPU
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
    usecuda=True
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")
    usecuda=False

# load in congressional speeches about immigration with immigration grade labels
print("Reading in congressional speeches about immigration")
cong_immig_training = pd.read_csv("../raw-data/immigration_training_data.csv", encoding="ISO-8859-1")
# keep only speeches from the 111th to the 114th congresses (overlaps with sample period)
cong_immig_training = cong_immig_training[cong_immig_training["congress"] >= 111]
# convert text to lower case
cong_immig_training["speech"] = cong_immig_training["speech"].apply(lambda x: x.lower())

# shuffle the data into 80% training and 20% test
print("Shuffling into training and test data")
cong_immig_training = cong_immig_training[["speech", "rating"]]
cong_immig_training = cong_immig_training.rename(columns={"speech":"text", "rating":"labels"})
cong_fine_tuning_data = shuffle(cong_immig_training, random_state=42)
# normalize the immigration grade (speeds up training)
cong_fine_tuning_data["labels"] = minmax_scale(cong_fine_tuning_data["labels"])
train_data,test_data = train_test_split(
    cong_fine_tuning_data,
    test_size=0.2,
    shuffle=False
    )


# wandb sweep arguments
sweep_config = {
    "name": "slant-model-optimization",
    "method": "bayes",
    "metric": {"name": "rsq", "goal": "maximize"},
    "parameters": {
        "num_train_epochs": {"min": 1, "max": 10},
        "learning_rate": {"min": float(0), "max": float(5e-5)}
        #"weight_decay": {"min": float(0), "max": float(1)},
    },
    "early_terminate": {"type": "hyperband", "min_iter": 1,},
}

sweep_id = wandb.sweep(sweep_config, project="Slant Hyperparamater Optimization")

# set logging parameters
logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)


# model arguments
model_args = ClassificationArgs(
    num_train_epochs=1,
    output_dir="../models/immig-sentiment/",
    best_model_dir="../models/immig-sentiment/best-model/",
    evaluate_during_training=True,
    evaluate_during_training_verbose=True,
    evaluate_during_training_steps=50000,
    save_steps=-1,
    save_model_every_epoch=False,
    overwrite_output_dir=True,
    max_seq_length=512,
    sliding_window=True,
    train_batch_size=32,
    eval_batch_size=128,
    regression=True,
    do_lower_case=True,/
    wandb_project="Immigration-Sentiment-Hyperparamater-Optimization"
    #weight_decay=0.1
    #use_early_stopping=True
    )

# set up wandb training
def train():
    # initialize a new wandb run
    wandb.init()

    # create a TransformerModel
    # use model that was pre-trained on cable news text
    model = ClassificationModel(
        "distilbert",
        "../models/pretraining/media-pretraining-bert/",
        args=model_args,
        use_cuda=usecuda,
        num_labels=1,
        sweep_config=wandb.config,
    )

    # train the model
    model.train_model(
        train_data,
        eval_df=test_data,
        rsq=lambda truth, predictions: r2_score(
            truth, predictions
        ),
    )

    # sync wandb
    wandb.join()

wandb.agent(sweep_id, train)

print("Done!")
