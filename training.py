import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import pandas as pd
import time

import parts
import models

from lib import data_loader_CIFAR10

import os
os.environ['CUDA_VISIBLE_DEVICES']='3'

# training parameters
batchsize = 64
lr0 = 1e-4
max_epochs = 500

# get data
trainvalid = data_loader_CIFAR10.get_train_valid_loader('./data', batch_size=batchsize, augment=True, random_seed=False)
trainloader, validloader = trainvalid

testloader = data_loader_CIFAR10.get_test_loader('./data', batch_size=batchsize)

# Run parameters
max_epochs = 500
time_label = time.strftime('%Y%m%d%H%M')
valacc_save = './logs/' + time_label + '_VA.csv'
tests_save = './logs/' + time_label + '_TA.csv'

# early stopping
patience = 50
min_improvement = 0.

def combined_loss(classification, replication, ins, label, classifier_crit, AE_crit):
    """
    Thanks to gradient descent, to concurrently minimize two objectives
    we need only minimize their sum. So in this case (having a class &
    AE output) we can just sum the loss for each and off we go.
    """
    return sum([classifier_crit(classification,label), AE_crit(replication,ins)])

val_acc_df =  pd.DataFrame()
tests_df = pd.DataFrame()

model = models.LCN10plain()
if torch.cuda.is_available():
    model.cuda()

classcrit = nn.CrossEntropyLoss()
aecrit = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr0)

# early stopping stuff
val_max = 0.
streak = 0
top_epoch = 0
savepath = './best' + time_label
val_acc = []

for epoch in range(max_epochs):
    running_loss = 0.0
    # set model to train mode (for the dropout)
    model.train()
    # go through the full dataset
    for i, data in enumerate(trainloader,0):
        # zero the gradient
        optimizer.zero_grad()
        # load data
        inputs, labels = data
        if torch.cuda.is_available():
            inputs, labels = inputs.cuda(), labels.cuda()
        # generate outputs
        classification, replication = model(inputs)
        # find loss
        loss = combined_loss(classification, replication, inputs, labels, classcrit, aecrit)
        # propagate
        loss.backward()
        # make a step
        optimizer.step()
        # update total loss for the epoch
        running_loss += loss.item()
            
    correct = 0
    total = 0
    AEloss = 0.0
    
    # validation
    # set model to eval mode (dropout)
    model.eval()

    with torch.no_grad():
        for data in validloader:
            inputs, labels = data
            if torch.cuda.is_available():
                inputs, labels = inputs.cuda(), labels.cuda()
            classification, replication = model(inputs)
            _, predicted = torch.max(classification.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            AEloss += aecrit(replication, inputs)
        
    score = 100*correct/total
    val_acc.append(score)

    print('[Epoch - Validation Accuracy - AE Loss]\t%d\t%.3f\t%.3f' % (epoch+1, score, AEloss))

    # check if improvement was made
    if score >= val_max:
        val_max = score
        streak = 0
        torch.save(model.state_dict(), savepath)
        top_epoch = epoch + 1
    else: 
        streak += 1

    # check if we're at the end of our patience
    if streak >= patience:
        break

print('Lost patience. Loading top model (epoch %d)' % top_epoch)

model.load_state_dict(torch.load(savepath))

correct = 0
total = 0
AEloss = 0

# set model to eval mode (dropout)
model.eval()

with torch.no_grad():
    for data in testloader:
        inputs, labels = data
        if torch.cuda.is_available():
            inputs, labels = inputs.cuda(), labels.cuda()
        classification, replication = model(inputs)
        _, predicted = torch.max(classification.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        AEloss += aecrit(replication, inputs)

# add to the results dataframe
# generate an index (may differ due to early stopping)
ind = list(np.arange(len(val_acc)))
# form a dataframe for this set
# val_acc is the list of records, ind is formed based on its length
df1 = pd.DataFrame(val_acc, index=ind)
# merge with the current set of results
val_acc_df = pd.concat([val_acc_df,df1],axis=1)
# save
val_acc_df.to_csv(valacc_save)

df2 = pd.DataFrame([100*correct.item()/total], index=[1])
tests_df = pd.concat([tests_df,df2], axis=1)
tests_df.to_csv(tests_save)

print('Accuracy on the testset: %.2f' % (100*correct/total))

os.remove(savepath)

