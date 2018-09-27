import torch
import torch.nn as nn
import torch.nn.functional as F

import parts


class latentConditionerModel10(nn.Module):
    def __init__(self):
        super(latentConditionerModel10, self).__init__()
        
        self.encoder = parts.encoder()
        self.latent2classifier = parts.latent2classifier()
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