"""
compiles the dataset here 
"""
import config 
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import os #for file path combining
from torch.nn.utils.rnn import pad_sequence #for padding 

#this will be a class inherited from pytorch 
class FlickrDataset(Dataset):
    def __init__(self,image_directory, captions_file, vocabulary, transform):
        self.image_directory = image_directory
        self.captions_file = captions_file
        self.vocabulary = vocabulary
        self.transform = transform 

    #inits the list of captions and images 
        self.images = []
        self.captions_list = []

        #parse the data into lists of images and captions
        with open(captions_file, 'r') as f:
            lines = f.readlines()
        
        for line in lines[1:]: #skips the header 
            parts = line.strip().split(",", 1) #splits into the firm caption 
            self.images.append(parts[0])  #the first part is the image 
            self.captions_list.append(parts[1]) #the rest of it is the captions
          
    def __len__(self): 
        return len(self.captions_list) #returns the length of the dataset 
    
    #returns the image caption pair for a specific index
    def __getitem__(self,index): 
        caption = self.captions_list[index]
        image_path = self.images[index]

        #load the iamge 
        image = Image.open(os.path.join(self.image_directory, image_path)).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        caption_indices = self.vocabulary.caption_to_idx(caption)
        caption_tensor = torch.tensor(caption_indices)

        return image, caption_tensor

#will take in the list of image caption pairs, and will pad the captions so that they are all the same length, 
def collate_fn(batch):
    images = [item[0] for item in batch] #splits the images from the captions 
    captions = [item[1] for item in batch]

    images = torch.stack(images) #stacks everything together
    padded_captions = pad_sequence(captions, batch_first=True, padding_value=0) #torches built in padding function 

    return images, padded_captions
