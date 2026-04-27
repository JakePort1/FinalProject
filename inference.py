"""
for testing the individual images
"""

import torch
import torchvision.transforms as transforms
from PIL import Image
from model import CNNtoRNN
from vocab import Vocabulary
import config
from dataset import FlickrDataset, collate_fn

def test_case(image_path,model_path,vocabulary):

    device= torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #load image
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)
    # 3. Load the model and checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Ensure vocab_size matches the checkpoint
    vocab_size = checkpoint["vocab_size"]
    model = CNNtoRNN(vocab_size).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    #Generate Caption
    with torch.no_grad():
        #This calls your greedy search method
        caption_indices = model.caption_image(img_tensor, vocabulary)

        #join words
        sentence = "".join(caption_indices).replace("<START>", "").replace("<END>", "").strip()
        
    print(f"\n--- Result ---")
    print(f"Image: {image_path}")
    print(f"Generated Caption: {sentence}\n")

if __name__ == "__main__":

    transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
    ])


    vocabulary = Vocabulary()
    dataset = FlickrDataset(
        image_directory=config.IMG_DIRECTORY,
        captions_file=config.CAPTIONS_DIRECTORY,
        vocabulary=vocabulary,
        transform=transform
    )
    

    vocabulary.build_vocab(dataset.captions_list)

    my_image = "edge_case4.jpeg" 
    my_checkpoint = "checkpoint_epoch_10.pth"

    test_case(my_image, my_checkpoint, vocabulary)