
# coding: utf-8

# In[7]:

import torch
import torch.nn as nn
import torch.nn.functional as F


# In[9]:


class encoder(nn.Module):
    def __init__(self):
        super(encoder, self).__init__()
        # input: 3,32,32 ; V = 3072
        self.conv1 = nn.Conv2d(3,16,3, stride=2, padding=1)  # 16,16,16 ; V = 4096
        self.conv2 = nn.Conv2d(16,32,3, stride=2, padding=1) # 32,8,8 ; V = 2048
        self.conv3 = nn.Conv2d(32,64,3, stride=2, padding=1) # 64,4,4 ; V = 1024
        self.conv4 = nn.Conv2d(64,128,2, stride=2) # 128,2,2 ; V = 512
        self.conv5 = nn.Conv2d(128,256,2) #  256,1,1 ; V = 256
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        
        return x


# In[6]:


class decoder(nn.Module):
    def __init__(self):
        super(decoder, self).__init__()
        
        self.unconv1 = nn.ConvTranspose2d(256,128,2) # 128,2,2
        self.unconv2 = nn.ConvTranspose2d(128,64,2, stride=2) # 64,4,4
        self.unconv3 = nn.ConvTranspose2d(64,32,3, stride=2, padding=1, output_padding = 1) # 32,8,8
        self.unconv4 = nn.ConvTranspose2d(32,16,3, stride=2, padding=1, output_padding = 1) # 16,16,16
        self.unconv5 = nn.ConvTranspose2d(16,3,3, stride=2, padding=1, output_padding = 1) # 3,32,32
        
    def forward(self, x, conditionals):
        g,b = conditionals
        x = F.relu(self.unconv1(x))
        x = (g[0].view(-1,g[0].size()[-1],1,1).expand_as(x) + 1)*(x-b[0].view(-1,b[0].size()[-1],1,1).expand_as(x))
        x = F.relu(self.unconv2(x))
        x = (g[1].view(-1,g[1].size()[-1],1,1).expand_as(x) + 1)*(x-b[1].view(-1,b[1].size()[-1],1,1).expand_as(x))
        x = F.relu(self.unconv3(x))
        x = (g[2].view(-1,g[2].size()[-1],1,1).expand_as(x) + 1)*(x-b[2].view(-1,b[2].size()[-1],1,1).expand_as(x))
        x = F.relu(self.unconv4(x))
        x = (g[3].view(-1,g[3].size()[-1],1,1).expand_as(x) + 1)*(x-b[3].view(-1,b[3].size()[-1],1,1).expand_as(x))
        x = F.sigmoid(self.unconv5(x))
        
        return x

class plaindecoder(nn.Module):
    def __init__(self):
        super(plaindecoder, self).__init__()
        
        self.unconv1 = nn.ConvTranspose2d(256,128,2) # 128,2,2
        self.unconv2 = nn.ConvTranspose2d(128,64,2, stride=2) # 64,4,4
        self.unconv3 = nn.ConvTranspose2d(64,32,3, stride=2, padding=1, output_padding = 1) # 32,8,8
        self.unconv4 = nn.ConvTranspose2d(32,16,3, stride=2, padding=1, output_padding = 1) # 16,16,16
        self.unconv5 = nn.ConvTranspose2d(16,3,3, stride=2, padding=1, output_padding = 1) # 3,32,32
        
    def forward(self, x):
        x = F.relu(self.unconv1(x))
        x = F.relu(self.unconv2(x))
        x = F.relu(self.unconv3(x))
        x = F.relu(self.unconv4(x))
        x = F.sigmoid(self.unconv5(x))
        return x
    
# In[10]:


class latent2classifier(nn.Module):
    """
    """
    def __init__(self):
        super(latent2classifier, self).__init__()
        
        # the number kernels in each layer of the AllConv net
        layers = [3,96,96,96,192,192,192,192,192,10]
        
        # the latent space is 256.1.1
        self.linear1 = nn.Linear(256, 32)
        # the number of final layers (parallel) is determined by those layers that are being conditioned (defined in
        # layers_to_condition)
        # also initialise the weights to be small to not be overly distruptive early in training
        self.gamma_outs = nn.ModuleList([nn.Linear(32,i) for i in layers])
        self.beta_outs = nn.ModuleList([nn.Linear(32,i) for i in layers])
        
    def forward(self, x):
        gammas = []
        betas = []
        
        x = torch.squeeze(x)
        x = F.relu(self.linear1(x))
        
        gammas = [gs(x) for gs in self.gamma_outs]
        betas = [bs(x) for bs in self.beta_outs]
        
        return gammas,betas


# In[11]:


class class2decoder(nn.Module):
    """
    """
    def __init__(self, num_classes):
        super(class2decoder, self).__init__()
        
        # the number kernels in each layer of the AllConv net
        layers = [128,64,32,16]
        
        # classification is 10
        self.linear1 = nn.Linear(num_classes, 32)
        # the number of final layers (parallel) is determined by those layers that are being conditioned (defined in
        # layers_to_condition)
        # also initialise the weights to be small to not be overly distruptive early in training
        self.gamma_outs = nn.ModuleList([nn.Linear(32,i) for i in layers])
        self.beta_outs = nn.ModuleList([nn.Linear(32,i) for i in layers])
        
    def forward(self, x):
        gammas = []
        betas = []
        
        x = F.relu(self.linear1(x))
        
        gammas = [gs(x) for gs in self.gamma_outs]
        betas = [bs(x) for bs in self.beta_outs]
        
        return gammas,betas


# In[12]:


class classifier(nn.Module):

    def __init__(self, num_classes):
        super(classifier, self).__init__()
        
        self.num_classes = num_classes
        
        # ********** All-convolutional network **********
        # Input is 32.32.3
        # Block 1 (1,2,3) : 32.32.3 -> 32.32.96 -> 32.32.96 -> 16.16.96
        # Block 2 (4,5,6) : 16.16.96 -> 16.16.192 -> 16.16.192 -> 8.8.192
        # Block 3 (7,8,9) : 8.8.192 -> 8.8.192 -> 8.8.192 -> 8.8.10
        # Pool & linear   : 8.8.10 -> 1.1.10 -> num_classes (10 or 100)
        # Convolutions
        self.conv1 = nn.Conv2d(3, 96, 3, padding=1)
        self.conv2 = nn.Conv2d(96, 96, 3, padding=1)
        self.conv3 = nn.Conv2d(96, 96, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(96, 192, 3, padding=1)
        self.conv5 = nn.Conv2d(192, 192, 3, padding=1)
        self.conv6 = nn.Conv2d(192, 192, 3, stride=2, padding=1)
        self.conv7 = nn.Conv2d(192, 192, 3, padding = 1)
        self.conv8 = nn.Conv2d(192, 192, 1)
        self.conv9 = nn.Conv2d(192, 10, 1)
        
        # Pooling layer
        self.pool = nn.AvgPool2d(8)
        self.linear = nn.Linear(10, self.num_classes)
        
        self.drop_in = nn.Dropout(0.2)
        self.drop_block = nn.Dropout(0.5)
    
    def forward(self, x, conditionals):
        g,b = conditionals
        # initial drop out pre-loops (all loops have the same input)
        x = self.drop_in(x)        
        x = (g[0].view(-1,g[0].size()[-1],1,1).expand_as(x) + 1)*(x-b[0].view(-1,b[0].size()[-1],1,1).expand_as(x))
        # The conditioning multiplies each featuremap by gamma and adds beta before
        # the activation takes place (to allow the conditioning to zero-out features).
        #
        # The only tricky bit is reshaping the conditioning vectors to fit dynamically
        # for the batch size. This is achieved using 'a.view(b)' and 'a.expand_as(b)':
        # - the conditioners (a) will be of size [batch, features]
        # - the block (b) moving through the network has size [batch, features, x, y]
        # so we first reshape the conditioner using view to be [batch, features, 1, 1]
        # -> a.view(-1,a.size()[-1],1,1)
        # then we duplicate the elements this to be the same size as the block using expand_as
        # -> a.view(-1,a.size()[-1],1,1).expand_as(b)
        # with the result being [batch, features, x, y]
        # the behaviour of expand_as is to replicate the elements:
        # [1].expand_as(2,2) = [[1,1],[1,1]]

        x = self.conv1(F.relu(x))
        x = (g[1].view(-1,g[1].size()[-1],1,1).expand_as(x) + 1)*(x-b[1].view(-1,b[1].size()[-1],1,1).expand_as(x))
        x = self.conv2(F.relu(x))
        x = (g[2].view(-1,g[2].size()[-1],1,1).expand_as(x) + 1)*(x-b[2].view(-1,b[2].size()[-1],1,1).expand_as(x))
        x = self.conv3(F.relu(x))
        x = (g[3].view(-1,g[3].size()[-1],1,1).expand_as(x) + 1)*(x-b[3].view(-1,b[3].size()[-1],1,1).expand_as(x))
        
        x = self.conv4(self.drop_block(F.relu(x)))
        x = (g[4].view(-1,g[4].size()[-1],1,1).expand_as(x) + 1)*(x-b[4].view(-1,b[4].size()[-1],1,1).expand_as(x))
        x = self.conv5(F.relu(x))
        x = (g[5].view(-1,g[5].size()[-1],1,1).expand_as(x) + 1)*(x-b[5].view(-1,b[5].size()[-1],1,1).expand_as(x))
        x = self.conv6(F.relu(x))
        x = (g[6].view(-1,g[6].size()[-1],1,1).expand_as(x) + 1)*(x-b[6].view(-1,b[6].size()[-1],1,1).expand_as(x))
        
        x = self.conv7(self.drop_block(F.relu(x)))
        x = (g[7].view(-1,g[7].size()[-1],1,1).expand_as(x) + 1)*(x-b[7].view(-1,b[7].size()[-1],1,1).expand_as(x))
        x = self.conv8(F.relu(x))
        x = (g[8].view(-1,g[8].size()[-1],1,1).expand_as(x) + 1)*(x-b[8].view(-1,b[8].size()[-1],1,1).expand_as(x))
        x = self.conv9(F.relu(x))
        x = (g[9].view(-1,g[9].size()[-1],1,1).expand_as(x) + 1)*(x-b[9].view(-1,b[9].size()[-1],1,1).expand_as(x))
        
        x = self.pool(F.relu(x))
        x = x.view(-1, 10)
        x = self.linear(x)
                
        return x

