### This script splits the cable news text data into training and test sets
# load in necessary packages
from pandarallel import pandarallel
from sklearn.model_selection import train_test_split
import dask.dataframe as dd
pandarallel.initialize()

# load in the data and keep relevant columns
print("Loading in cable news data")
media_segments = dd.read_csv("../raw-data/cable_news_data.txt",
    sep=";",
    header=None,
    names=["id", "link", "segment", "start", "end", "text"],
    usecols=["id", "text"]
)
media_segments = media_segments.compute()

# drop missings, convert all text data to strings, convert to lower case
media_segments = media_segments.dropna(subset=["text"])
media_segments["text"] = media_segments["text"].parallel_apply(lambda x: str(x))
media_segments["text"] = media_segments["text"].apply(lambda x: x.lower())

# keep only the text column
media_segments = media_segments["text"]

# split into 80% training and 20% test test
print("Splitting into training and test")
training_data,test_data = train_test_split(
    media_segments,
    test_size=.2,
    shuffle=True,
    random_state=42
    )

# save data to disk
print("Saving to disk")
training_data.to_csv("../raw-data/pretraining_training_data.txt", header=None, index=None)
test_data.to_csv("../raw-data/pretraining_test_data.txt", header=None, index=None)
