"""
This file will have the CNN and LSTM in it 
"""
import torch
import torch.nn as nn
import torchvision.models as models
import config
import vocab

class TransformerModel(nn.Module):
    def __init__(self, vocab_size):
        super(TransformerModel, self).__init__()

        #pretrained encoder, containing pretrain weights 
        self.vit=models.vit_b_16()

        #remove label, raw features
        self.vit.heads=nn.Identity()

        self.encoder_linear=nn.Linear(768,config.EMBED_SIZE)

        #decoder
        self.embedding = nn.Embedding(vocab_size,config.EMBED_SIZE)
        self.pos_encoder= nn.Parameter(torch.rand(1,config.MAX_LEN,config.EMBED_SIZE))



class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()

        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        for param in resnet.parameters():
            param.requires_grad = False


        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)

        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, images):

        features = self.resnet(images)
        
        #batch_size, 2048
        features = features.view(features.size(0), -1)
        
        #Map to embed_size
        features = self.embed(features)
        
        return self.dropout(features)


class Encoder(nn.Module):

    def __init__(self, embed_size, train_CNN=False):

        super(Encoder, self).__init__()

        self.train_CNN = train_CNN
 

        self.inception = models.inception_v3(
            weights=models.Inception_V3_Weights.DEFAULT,
            aux_logits=True
        )
        self.inception.aux_logits=False
        #self.inception.AuxLogits=None

        #Replace final FC layer
        self.inception.fc = nn.Linear(
            self.inception.fc.in_features,
            embed_size
        )

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

        # Freeze CNN parameters (DO THIS ONCE)
        for name, param in self.inception.named_parameters():

            if "fc.weight" in name or "fc.bias" in name:
                param.requires_grad = True
            else:
                param.requires_grad = train_CNN


    def forward(self, images):

        features = self.inception(images)

        return self.dropout(
            self.relu(features)
        )



class Decoder(nn.Module):

    def __init__(
        self,
        embed_size,
        hidden_size,
        vocab_size,
        num_layers
    ):

        super(Decoder, self).__init__()

        self.embed = nn.Embedding(vocab_size,embed_size)

        self.lstm = nn.LSTM(embed_size,hidden_size,num_layers,batch_first=True)

        self.linear = nn.Linear(hidden_size,vocab_size)

        self.dropout = nn.Dropout(0.5)

    def forward(self, features, captions):

        embeddings = self.dropout(self.embed(captions))
        #embeddings = embeddings.permute(1, 0, 2)

        #embeddings=embeddings[:-1]
        embeddings = torch.cat((features.unsqueeze(1), embeddings),dim=1)

        hiddens, _ = self.lstm(embeddings)
        outputs = self.linear(hiddens)

        return outputs[:, 1:, :]  


class CNNtoRNN(nn.Module):

    def __init__(self, vocab_size):

        super(CNNtoRNN, self).__init__()

        self.encoder = EncoderCNN(config.EMBED_SIZE)

        self.decoder = Decoder(config.EMBED_SIZE, config.HIDDEN_SIZE, vocab_size, config.NUM_LAYERS)


    def forward(self, images, captions):

        features = self.encoder(images)

        outputs = self.decoder(features,captions)

        return outputs

    # Caption generation

    def caption_image(self, image, vocabulary):

        result_caption = []

        with torch.no_grad():

            x = self.encoder(image)

            # Add sequence dimension
            x = x.unsqueeze(0)

            states = None

            for _ in range(config.MAX_LEN):
                hiddens, states = self.decoder.lstm(x, states)
                outputs = self.decoder.linear(hiddens.squeeze(0))
                predicted = outputs.argmax(1)
    
                predicted_word = vocabulary.idx_to_captions([predicted.item()])
                
                #break if it hit end token
                if predicted_word == "" or predicted.item() == vocabulary.word_to_idx[config.END_TOKEN]:
                        break

                result_caption.append(predicted.item())
                x = self.decoder.embed(predicted).unsqueeze(0)

        return vocabulary.idx_to_captions(result_caption)

