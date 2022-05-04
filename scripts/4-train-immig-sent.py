### The WandB sweep suggests that 3 epochs and a learning rate of 2e-05 is optimal for the
### immigration sentiment model
### Train such a model
# import necessary packages
from simpletransformers.classification import ClassificationModel, ClassificationArgs
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import minmax_scale
from scipy.stats import pearsonr
from sklearn.metrics import r2_score

# check if a CUDA device is available, if so use the GPU, if not use CPU
if torch.cuda.is_available():
    # Tell PyTorch to use the GPU.
    device = torch.device("cuda:0")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
    usecuda = True
# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")
    usecuda=False

# load in speeches with immig labels
immig_training_data = pd.read_csv("../raw-data/immigration_training_data.csv",encoding="ISO-8859-1")
# convert speeches to lower case
immig_training_data["speech"] = immig_training_data["speech"].apply(lambda x: x.lower())
# normalize the immig labels to speed up training
immig_training_data["immig_label"] = minmax_scale(immig_training_data["rating"])
# shuffle data and split into training and test sets
immig_fine_tuning_data = immig_training_data[["speech", "immig_label"]]
immig_fine_tuning_data = immig_fine_tuning_data.rename(columns={"speech":"text", "immig_label":"labels"})
immig_fine_tuning_data = shuffle(immig_fine_tuning_data, random_state=42)
train_data,test_data = train_test_split(
    immig_fine_tuning_data,
    test_size=0.2,
    shuffle=False
    )

# configure the bert model
model_args = ClassificationArgs(
    num_train_epochs=3,
    learning_rate = 2.1669410734285063e-05,
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
    do_lower_case=True,
    wandb_project="bert-immig",
    #weight_decay=0.5395183313115828
    #use_early_stopping=True
    )

# create the bert model
model = ClassificationModel(
    "distilbert",
    "../models/pretraining/media-pretraining-bert/",
    args=model_args,
    use_cuda=usecuda,
    num_labels=1
)

# train the model
model.train_model(
    train_data,
    eval_df=test_data,
    rsq=lambda truth, predictions: r2_score(truth, predictions)
    )
# evalue the model's performance
print("Evaluating model")
#model.eval_model(test_data)
to_predict = list(test_data["text"])
preds = model.predict(to_predict)[1]
preds = [np.mean(x) for x in preds]
test_data["predicted"] = preds
print("-----------------------------------")
print("Immigration Rating Results: ")
print("Correlation: " + str(pearsonr(test_data["labels"], test_data["predicted"])))
print("R-Squared: " + str(r2_score(test_data["labels"], test_data["predicted"])))
eval_txt = open("../models/immig-classifier/eval_results.txt", "a")
eval_txt.write("r:" + str(pearsonr(test_data["labels"], test_data["predicted"])))
eval_txt.write("r2:" + str(r2_score(test_data["labels"], test_data["predicted"])))
eval_txt.close()
print("Done!")
