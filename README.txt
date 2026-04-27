

Notes on what each file does: 

config.py: establishes the hyperparameters and environment settings (hidden_layers, learning_rate, batch_size, embedding_size, threshold, max_len) so they can be adjusted in one place. Please make sure the paths of IMG_DIRECTORY and CAPTIONS_DIRECTORY are correct and point to your Flicker8k folders.

vocab.py: processes the raw text to build a mapping of word to index, creating the vocabulary used by the model

dataset.py: build the data loader, pair images with their captions, and pad the captions to the same length.

data.py: download and print out the path of flicker8k.

model.py: contains the NN architecture, including an Encoder (pre-trained ResNet), a Decoder (LSTM), and a class of CNNtoRNN that connects the two classes. Notes: If you want to use a non-pretrained ResNet, please change the self.encoder = EncoderCNN(config.EMBED_SIZE) to EncoderCNN_NoPretrain(nn.Module). 

train.py: Manages the training process. It splits the dataset into a training and testing set, iterates through epochs, calculates the loss, and performs backpropagation to update weights. It logs train/val loss to a CSV file and saves the model states as a pth file. Notes: If you don't want to use the full dataset, adjust lines 99 and 100 to subset it.

inference.py: Loads the saved .pth weight and vocabulary to generate captions for new images. Verify that the my_image and my_checkpoints paths are correctly set. You can test different images and models by adjusting these two lines.

test.jpg: A test JPG file for inference.py.

We want to upload a final_model.pth file, but unfortunately, the file is too large. So we will include that in the canvas zip file.
