"""
This will store all the settings to be used by the models

"""


# Directories 
#check git push

#IMG_DIRECTORY = "/Users/jacobport/.cache/kagglehub/datasets/adityajn105/flickr8k/versions/1/Images"
#CAPTIONS_DIRECTORY = "/Users/jacobport/.cache/kagglehub/datasets/adityajn105/flickr8k/versions/1/captions.txt"

IMG_DIRECTORY ="/Users/Althea/Desktop/FinalProject/archive/Images"
CAPTIONS_DIRECTORY ="/Users/Althea/Desktop/FinalProject/archive/captions.txt"
checkpoints = None

BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4

EMBED_SIZE = 512
HIDDEN_SIZE = 1024

#change hidden to 60

NUM_LAYERS = 2

MAX_LEN = 35
PAD_TOKEN = "<PAD>"
START_TOKEN = "<START>"
END_TOKEN = "<END>"
UNK_TOKEN = "<UNK>"

THRESHOLD = 5
