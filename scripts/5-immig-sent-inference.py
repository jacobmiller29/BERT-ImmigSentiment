### Use the trained immig sentiment model for inference: get the predicted sentiment scores
### given the text of cable news broadcasts
# load in necessary packages
import torch
from simpletransformers.classification import ClassificationModel, ClassificationArgs
import pandas as pd
import numpy as np
from pandarallel import pandarallel
import dask.dataframe as dd

# check if a CUDA device is available, if so use the GPU, if not use CPU
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
    usecuda = True
else:
    print("We will use the CPU.")
    usecuda = False

pandarallel.initialize()

# load in the cable news data
print("Loading in media data")
media_data = dd.read_csv("../raw-data/cable_news_data.txt",
    sep=";",
    header=None,
    names=["id", "link", "segment", "start", "end", "text"],
    usecols=["id", "text", "start", "end"]
    )
media_data = media_data.compute()
# drop missings
media_data = media_data.dropna(subset=["text"])
# convert text column to strings
media_data["text"] = media_data["text"].parallel_apply(lambda x: str(x))
# keep only segments that contain at least one term related to immigration
# (word that reduces to stem word "immigr")
media_data["immigration"] = media_data["text"].parallel_apply(lambda x: "immigr" in x)
media_data = media_data[media_data["immigration"] == True]

# convert pandas dataframe into list
immig_segments = list(media_data["text"])


print("Loading in immigration sentiment model")

# set up model arguments for inference
model_args = ClassificationArgs(
    sliding_window=True,
    eval_batch_size=2048
    )
# load in the trained immig sentiment model
model = ClassificationModel(
    "distilbert",
    "../models/immigration-sentiment/best-model/",
    use_cuda=usecuda,
    num_labels=1,
    args=model_args
)

# run the model on the cable news broadcasts about immigration
print("Running the model on the media segments")
immigration_scores = model.predict(immig_segments)[1]


# create pandas column from the list
media_data["immigration_score"] = immigration_scores
# drop the text column (saves disk space)
media_data = media_data.drop(columns=["text"], axis=0)

# save the results
print("Saving the results")
media_data.to_csv("../output-data/media-immig-sent-scores.csv")
