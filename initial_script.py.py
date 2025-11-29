#!/usr/bin/env python
# coding: utf-8

# # Wild Edible Plants Classifier

# In[1]:


import numpy as np

from functions.model import Classifier
from functions.plotting import Plotter
from functions.utils import Utilities

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets
import torchvision.models as models

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# Run once if permission error when training
# torch.hub.list('pytorch/vision:v0.8.0', force_reload=True)


# In[3]:


# Set hyperparameters
EPOCHS = 20
LEARNING_RATE = 0.001
BATCH_SIZE = 64 # samples per batch to load
SPLIT_SIZE = 0.15 # validation & test dataset size
H_LAYERS = [512, 256]
N_CLASSES = 35
N_PREDS = 5 # number of model predictions (e.g. top-5 error rate)
MAIN_FILEPATH = 'dataset/resized'
SAMPLE_FILEPATH = 'dataset/sample'
SEED = 1


# # 1. Data Preparation

# ## 1.1 Visualize Plant Classes

# In[4]:


# Load sample of data with basic transforms
SAMPLE_TRANSFORM = transforms.Compose([
    transforms.Resize(400),
    transforms.CenterCrop(400),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
sample = torchvision.datasets.ImageFolder(SAMPLE_FILEPATH,
                                          transform=SAMPLE_TRANSFORM)
print(sample)


# In[5]:


# Create global class labels
LABELS = np.array(list(sample.class_to_idx), dtype=object)

# Set Plotter and Helper class objects
vis = Plotter(LABELS)
utils = Utilities()


# In[6]:


# Obtain the batch of images (full sample)
sample_loader = torch.utils.data.DataLoader(sample, batch_size=LABELS.size)
sample_iter = iter(sample_loader)
s_img, s_label = sample_iter.next()
s_img = s_img.numpy() # Convert to numpy so can display

# Visualize each class
vis.visualize_imgs(s_img, s_label)

# Save plot
# fig.savefig("class_sample.png")


# # 2. Data Segregation

# ## 2.1 Data Augmentation

# In[7]:


# Set transformations for batch data
TRANSFORM = transforms.Compose([
    transforms.Resize(224), # Resize images to 224
    transforms.CenterCrop(224), # Make images 224x224
    transforms.RandomHorizontalFlip(), # Randomly flip some samples (50% chance)
    transforms.RandomRotation(20), # Randomly rotate some samples
    transforms.ToTensor(), # Convert image to a tensor
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)) # Normalize image values
])

# Get the data
dataset = torchvision.datasets.ImageFolder(MAIN_FILEPATH,
                                              transform=TRANSFORM)
print(dataset)


# ## 2.2 Split the Data

# In[8]:


# Split data
train_loader, valid_loader, test_loader = utils.split_data(dataset, BATCH_SIZE, 
                                                           SPLIT_SIZE, SEED)


# ## 2.3 Visualize a Batch of Training Data

# In[9]:


# Obtain the batch of images
train_iter = iter(train_loader)
imgs, lbls = train_iter.next()
imgs = imgs.numpy() # Convert to numpy so can display

# Visualize each class
vis.visualize_imgs(imgs, lbls, figsize=(25, 8), num_rows=3, num_cols=30)


# # 3. Model Training

# ## 3.1 Create CNN Architectures
# As the dataset used is small and contains similarities to the ImageNet database, the model itself can be reused. The only adjustment that needs to be made is to add one or two new fully-connected layers to be trained on the wild edible plant dataset.

# In[10]:


# Create instances of pretrained CNN architectures
googlenet = models.googlenet(pretrained=True)
mobilenetv2 = models.mobilenet_v2(pretrained=True)
resnet34 = models.resnet34(pretrained=True)


# Next it is important to check the layers of the architectures to understand the naming of the ending layers. Taking those names we can then add in the new layers for training the dataset.

# In[11]:


googlenet


# In[12]:


mobilenetv2


# In[13]:


resnet34


# In[14]:


# Initialize new classifiers
gnet_classifier = Classifier(in_features=googlenet.fc.in_features, out_features=N_CLASSES, 
                             hidden_layers=H_LAYERS)
mobilenet_classifier = Classifier(in_features=mobilenetv2.classifier[1].in_features, 
                                  out_features=N_CLASSES, hidden_layers=H_LAYERS)
resnet_classifier = Classifier(in_features=resnet34.fc.in_features, out_features=N_CLASSES, 
                               hidden_layers=H_LAYERS)
print("GoogLeNet",gnet_classifier)
print("\nMobileNet",mobilenet_classifier)
print("\nResNet",resnet_classifier)


# ## 3.2 Update CNN Architectures

# In[15]:


# Set models and model names as lists
MODEL_NAMES = ["GoogLeNet", "MobileNet v2", "ResNet-34"]
MODELS = [googlenet, mobilenetv2, resnet34]


# In[16]:


# Freeze architecture parameters to avoid backpropagating them
# Avoiding replacing pretrained weights
for model in MODELS:
    for param in model.parameters():
        param.requires_grad = False


# In[17]:


# Replace last FC layer for GoogLeNet with new classifier
googlenet.fc = gnet_classifier
googlenet


# In[18]:


# Replace classifier with new one
mobilenetv2.classifier = mobilenet_classifier
mobilenetv2


# In[19]:


resnet34.fc = resnet_classifier
resnet34


# In[20]:


# Total params for each model
for idx, model in enumerate(MODELS):
    print(f"{MODEL_NAMES[idx]}:")
    model.total_params = sum(p.numel() for p in model.parameters())
    print(f'{model.total_params:,} total parameters')
    model.trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{model.trainable_params:,} training parameters\n')


# ## 3.3 Initial Training of CNN Models

# In[21]:


# Set to GPU if available
device = utils.set_device()


# In[23]:


# Train GoogLeNet
# Reminder: only target fc parameters (rest are frozen)
googlenet.to(device) # move to GPU
SAVE_FILEPATH = "saved_models/GoogLeNet_64_512_256.pt"

# Set loss function and optimizer
criterion = nn.NLLLoss() # Negative Log Likelihood Loss
optimizer = torch.optim.Adam(googlenet.fc.parameters(), lr=LEARNING_RATE)

utils.train(googlenet, train_loader, valid_loader, criterion, 
            optimizer, SAVE_FILEPATH, EPOCHS)


# In[24]:


# Train MobileNet V2
# Reminder: only target classifier parameters (rest are frozen)
mobilenetv2.to(device) # move to GPU
SAVE_FILEPATH = "saved_models/MobileNet-V2_64_512_256.pt"

# Set loss function and optimizer
criterion = nn.NLLLoss() # Negative Log Likelihood Loss
optimizer = torch.optim.Adam(mobilenetv2.classifier.parameters(), lr=LEARNING_RATE)

utils.train(mobilenetv2, train_loader, valid_loader, criterion, 
            optimizer, SAVE_FILEPATH, EPOCHS)


# In[25]:


# Train ResNet-34
# Reminder: only target fc parameters (rest are frozen)
resnet34.to(device) # move to GPU
SAVE_FILEPATH = "saved_models/ResNet-34_64_512_256.pt"

# Set loss function and optimizer
criterion = nn.NLLLoss() # Negative Log Likelihood Loss
optimizer = torch.optim.Adam(resnet34.fc.parameters(), lr=LEARNING_RATE)

utils.train(resnet34, train_loader, valid_loader, criterion, 
            optimizer, SAVE_FILEPATH, EPOCHS)


# # 4. Performance Evaluation

# In[ ]:


# Load saved models - important for using train_losses and valid_losses
utils.load_model(googlenet, "saved_models/GoogLeNet_64_512_256.pt")
utils.load_model(mobilenetv2, "saved_models/MobileNet-V2_64_512_256.pt")
utils.load_model(resnet34, "saved_models/ResNet-34_64_512_256.pt")


# In[26]:


# Calculate predictions, test labels, and probabilities
model_preds, model_trues, model_probas = [], [], []
for i in range(len(MODELS)):
    y_pred, y_true, _, y_probas = utils.predict(MODELS[i], test_loader, N_PREDS,
                                                store_labels=True, 
                                                store_probas=True)
    model_preds.append(y_pred)
    model_trues.append(y_true)
    model_probas.append(y_probas)


# In[31]:


# Plot ROC curves
for i in range(len(MODELS)):
    vis.plot_roc(MODEL_NAMES[i], model_probas[i], 
                 model_trues[i], figsize=(18, 12))


# In[32]:


# Plot Confusion Matrices
for i in range(len(MODELS)):
    y_pred_cm, y_true_cm = utils.indices_to_labels(model_preds[i], model_trues[i], LABELS)
    vis.plot_cm(MODEL_NAMES[i], y_pred_cm, y_true_cm)


# In[33]:


# Plot loss comparsion
vis.create_plots(MODELS, MODEL_NAMES, figsize=(15, 4), plot_func=vis.plot_losses,
                 plot_name="valid_model_losses", save=False)


# # 5. Hyperparameter Tuning
# Please refer to the Jupyter Notebook `2. wep_classifier_tuning.ipynb` for this section.
