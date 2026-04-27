
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Subset
import csv
import config
from model import CNNtoRNN
from dataset import FlickrDataset, collate_fn
from vocab import Vocabulary

def train_epoch(model, loader, optimizer, criterion, device,  step):

  
    model.train()
    total_loss = 0.0
    
    for batch_idx, (imgs, captions) in enumerate(loader):
        imgs = imgs.to(device)
        captions = captions.to(device)

        #Forward pass
        outputs = model(imgs, captions[:, :-1])
        targets = captions[:, 1:]
        
        loss = criterion(
            outputs.reshape(-1, outputs.shape[2]), 
            targets.reshape(-1)
        )

        #Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
        optimizer.step()

        #Logging
        total_loss += loss.item()
        step += 1
        
    return total_loss / len(loader), step

def validate(model, loader, criterion, device):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for imgs, captions in loader:
            imgs, captions = imgs.to(device), captions.to(device)
            outputs = model(imgs, captions[:, :-1])
            targets = captions[:, 1:]
            loss = criterion(outputs.reshape(-1, outputs.shape[2]), targets.reshape(-1))
            val_loss += loss.item()
    return val_loss / len(loader)


def save_checkpoint(model, optimizer, epoch, vocab_size, filename):

    checkpoint = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
        "vocab_size": vocab_size,
        "embed_size": config.EMBED_SIZE,
        "hidden_size": config.HIDDEN_SIZE,
    }
    torch.save(checkpoint, filename)
    print(f"-> Checkpoint saved: {filename}")

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    vocab_size = len(vocabulary.word_to_idx)

    #Correct Subset splitting
    train_indices = list(range(0, 2000))
    test_indices = list(range(2000, 4500))
    train_set = Subset(dataset, train_indices)
    test_set = Subset(dataset, test_indices)

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )

    test_loader = DataLoader(
        dataset=test_set,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )

    print(f"Vocab Size: {vocab_size} | Train size: {len(train_set)} | Val size: {len(test_set)}")

    #Model, Criterion, Optimizer
    model = CNNtoRNN(vocab_size).to(device)
    

    criterion = nn.CrossEntropyLoss(ignore_index=vocabulary.word_to_idx[config.PAD_TOKEN])
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    #Training Loop
    csv_file = open("training_log.csv", "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["Epoch", "train loss", "val loss"])
    
    step = 0
    for epoch in range(1, config.NUM_EPOCHS + 1):
        avg_train_loss, step = train_epoch(model, train_loader, optimizer, criterion, device,  step)
        avg_test_loss = validate(model, test_loader, criterion, device)
        
        print(f"Epoch [{epoch}/{config.NUM_EPOCHS}] | Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f}")

        
        save_checkpoint(model, optimizer, epoch, vocab_size, f"checkpoint_epoch_{epoch}.pth")
        csv_writer.writerow([epoch,  avg_train_loss,avg_test_loss])
        csv_file.flush()

    csv_file.close()

    print("Training Complete.")

if __name__ == "__main__":
    main()