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

    # 4. Generate Caption
    with torch.no_grad():
        # This calls your greedy search method
        caption_indices = model.caption_image(img_tensor, vocabulary)

        # Join words into a clean sentence
        sentence = "".join(caption_indices).replace("<START>", "").replace("<END>", "").strip()
        
    print(f"\n--- Result ---")
    print(f"Image: {image_path}")
    print(f"Generated Caption: {sentence}\n")

if __name__ == "__main__":
    # 1. Setup the same transform used in training
    transform = transforms.Compose([
        transforms.Resize((356, 356)),
        transforms.RandomCrop((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 2. Initialize Vocabulary and Dataset 
    # This ensures word <-> index mapping is identical to training

    vocabulary = Vocabulary()
    dataset = FlickrDataset(
        image_directory=config.IMG_DIRECTORY,
        captions_file=config.CAPTIONS_DIRECTORY,
        vocabulary=vocabulary,
        transform=transform
    )
    
    # Build the vocab based on the full captions list
    vocabulary.build_vocab(dataset.captions_list)

    # 3. Path to your image and your best checkpoint
    # Make sure "test_image.jpg" is in your folder!
    my_image = "test.jpg" 
    my_checkpoint = "unfreeze1000_checkpoint_epoch_49.pth"

    # 4. Run the test
    test_case(my_image, my_checkpoint, vocabulary)