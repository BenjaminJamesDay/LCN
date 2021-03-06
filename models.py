import torch
import torch.nn as nn
import torch.nn.functional as F

import parts

class LCN10plain(nn.Module):
    """
    A latent space conditional model for the CIFAR-10 dataset.
    The model consists of an encoder, decoder & classifier as
    well as two subnetworks that generate conditionals from the
    latent space for use on the classifier and from the class
    prediction for use on the decoder.
    
    The system works as follows:
        Notation:
            x - input
            y - label
            l - latent embedding
            y-hat - prediction
            x-hat - replication
    1 - the encoder takes x and embeds it as l
    2 - the first conditioner takes l and produces conditioning
        parameters for the classifier
    3 - the classifier takes x and produces y-hat under the
        conditioing from (2)
    4 - the decoder takes l and produces x-hat
    """
    def __init__(self):
        super(LCN10plain, self).__init__()
        
        self.encoder = parts.encoder()
        self.latent2classifier = parts.latent2classifier(10)
        self.classifier = parts.classifier(10)
        self.decoder = parts.plaindecoder()
        
    def forward(self, x):
        l = self.encoder(x)
        classConditionals = self.latent2classifier(l)
        yHat = self.classifier(x, classConditionals)
        xHat = self.decoder(l)
        
        return [yHat, xHat]

class LCN10(nn.Module):
    """
    A latent space conditional model for the CIFAR-10 dataset.
    The model consists of an encoder, decoder & classifier as
    well as two subnetworks that generate conditionals from the
    latent space for use on the classifier and from the class
    prediction for use on the decoder.
    
    The system works as follows:
        Notation:
            x - input
            y - label
            l - latent embedding
            y-hat - prediction
            x-hat - replication
    1 - the encoder takes x and embeds it as l
    2 - the first conditioner takes l and produces conditioning
        parameters for the classifier
    3 - the classifier takes x and produces y-hat under the
        conditioing from (2)
    4 - the second conditioner takes y-hat and produces conditioning
        parameters for decoder
    5 - the decoder takes l and produces x-hat under the conditioning
        from (4)
    """
    def __init__(self):
        super(LCN10, self).__init__()
        
        self.encoder = parts.encoder()
        self.latent2classifier = parts.latent2classifier(10)
        self.classifier = parts.classifier(10)
        self.class2decoder = parts.class2decoder(10)
        self.decoder = parts.decoder()
        
    def forward(self, x):
        l = self.encoder(x)
        classConditionals = self.latent2classifier(l)
        yHat = self.classifier(x, classConditionals)
        decoderConditionals = self.class2decoder(yHat)
        xHat = self.decoder(l, decoderConditionals)
        
        return [yHat, xHat]

class LCN100(nn.Module):
    """
    The same system as LCN10 but for the CIFAR-100 dataset.
    The networks are the same shape and size, the only difference
    is the output layer size of the classifier (100) and as such
    the input size to the second conditioner.
    """
    def __init__(self):
        super(LCN100, self).__init__()
        
        self.encoder = parts.encoder()
        self.latent2classifier = parts.latent2classifier(100)
        self.classifier = parts.classifier(100)
        self.class2decoder = parts.class2decoder(100)
        self.decoder = parts.decoder()
        
    def forward(self, x):
        l = self.encoder(x)
        classConditionals = self.latent2classifier(l)
        yHat = self.classifier(x, classConditionals)
        decoderConditionals = self.class2decoder(yHat)
        xHat = self.decoder(l, decoderConditionals)
        
        return [yHat, xHat]

class LCN100plain(nn.Module):
    """
    LCN100 analogue of LCN10plain
    """
    def __init__(self):
        super(LCN100plain, self).__init__()
        
        self.encoder = parts.encoder()
        self.latent2classifier = parts.latent2classifier(100)
        self.classifier = parts.classifier(100)
        self.decoder = parts.plaindecoder()
        
    def forward(self, x):
        l = self.encoder(x)
        classConditionals = self.latent2classifier(l)
        yHat = self.classifier(x, classConditionals)
        xHat = self.decoder(l)
        
        return [yHat, xHat]