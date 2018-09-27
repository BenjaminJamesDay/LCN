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


print('Imports successful\n')

# training parameters
batchsize = 64
lr0 = 1e-4
max_epochs = 500

# get data
trainvalid = data_loader_CIFAR10.get_train_valid_loader('./data', batch_size=batchsize, augment=True, random_seed=False)
trainloader, validloader = trainvalid

testloader = data_loader_CIFAR10.get_test_loader('./data', batch_size=batchsize)

print('Data loaded\n')

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

model = models.latentConditionerModel10()
print('Model loaded\n')
if torch.cuda.is_available():
    model.cuda()
    print('Model moved to GPU\n')

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
    print('Starting epoch %d \n' % (epoch+1))
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
        if i % 10 == 0:
            print('Finished batch %d' % (i))
            
    # arrays to store results (need 1 place for a 1-loop, 2 places for a 2-loop etc.)
    correct = 0
    total = 0
    AEloss = 0.0
    
    # validation
    # set model to eval mode (dropout)
    model.eval()
    for data in validloader:

        # load data
        inputs, labels = data
        if torch.cuda.is_available():
            inputs, labels = inputs.cuda(), labels.cuda()
        classification, replication = model(inputs)
        
        _, predicted = torch.max(classification.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
        
        AEloss += aecrit(replication, inputs)

    score = 100*correct/total
    val_acc.append(score)

    print('[Loop - Epoch - Validation Accuracy - AE Loss]\t%d\t%d\t%.3f\t%.3f' % (loop, epoch+1, 100*correct/total, AEloss))

    # check if improvement was made in the final loop
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

print('Lost patience at loop', loop, 'loading top model (epoch %d)' % top_epoch)


# In[ ]:


model.load_state_dict(torch.load(savepath))

correct = 0
total = 0

# set model to eval mode (dropout)
model.eval()

for data in testloader:
    # load data
    inputs, labels = data
    if torch.cuda.is_available():
        inputs, labels = inputs.cuda(), labels.cuda()
    classification, replication = model(inputs)

    _, predicted = torch.max(classification.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()

    AEloss += aecrit(replication, inputs)

# add to the results dataframe
# generate an index (may differ due to early stopping)
ind = list(np.arange(len(val_acc)))
# form a dataframe for this set
# val_acc is the list of records, ind is formed based on its length
# and the column name is the current loop number
df1 = pd.DataFrame(val_acc, index=ind, columns=headers)
# merge with the current set of results
val_acc_df = pd.concat([val_acc_df,df1],axis=1)
# save
val_acc_df.to_csv(valacc_save)

df2 = pd.DataFrame([100*correct.item()/total], index=[1], columns=[loop])
tests_df = pd.concat([tests_df,df2], axis=1)
tests_df.to_csv(tests_save)

print('Loop %d accuracy on the test images: %d %%' % (loop,100*correct/total))


# In[ ]:


os.remove(savepath)

