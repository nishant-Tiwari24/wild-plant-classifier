import os
import time
import numpy as np

from functions.model import Classifier
from functions.utils import Utilities

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets
import torchvision.models as models

class UnNormalize(object):
    """
    Used to convert a normalized torch tensor (image) into an unnormalized state. Used for plotting classification prediction images.
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Parameters:
            tensor (torch.Tensor): Tensor image of size (colour, height, width) to be normalized.
        Returns:
            torch.Tensor: Normalized image.
        """
        # Normalized state: t.sub_(m).div_(s) - simply perform opposite
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            
        return tensor

class Tuner:
    """
    Contains utility functions that are used within the hyperparameter tuning Notebook. Combines multiple components from the initial Notebook to condense the hyperparameter Notebook and focus on the tuning.
    """
    def __init__(self):
        self.utils = Utilities()
        self.device = self.utils.set_device()
    
    def set_data(self, filepath):
        """
        Sets the dataset using pre-defined transformations.
        
        Parameters:
            filepath (string) - filepath to the dataset
        """
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
        
        # Set transformations for batch data
        transform = transforms.Compose([
            transforms.Resize(224), # Resize images to 224
            transforms.CenterCrop(224), # Make images 224x224
            transforms.RandomHorizontalFlip(), # Randomly flip some samples (50% chance)
            transforms.RandomRotation(20), # Randomly rotate some samples
            transforms.ToTensor(), # Convert image to a tensor
            transforms.Normalize(mean=mean, std=std) # Normalize image values
        ])
        
        # Set dataset, labels and unnormalized object
        dataset = torchvision.datasets.ImageFolder(filepath, transform=transform)
        self.labels = np.array(list(dataset.class_to_idx), dtype=object)
        self.unnorm = UnNormalize(mean=mean, std=std)
        return dataset
    
    def set_initial_models(self, n_classes, h_layers):
        """
        Used to set the three models (GoogLeNet, MobileNet v2, and ResNet-34) with new classifiers.
        
        Parameters:
            n_classes (int) - number of classes for to output
            h_layers (list) - integers that represent each layers node count, can be 1 list or 3 lists
        """
        # Set class specific variables
        self.h_layers = h_layers
        self.n_classes = n_classes
        
        # Create instances of pretrained CNN architectures
        googlenet = models.googlenet(pretrained=True)
        mobilenetv2 = models.mobilenet_v2(pretrained=True)
        resnet34 = models.resnet34(pretrained=True)
        
        # Initialize new classifiers
        if isinstance(h_layers[0], list):
            gnet_classifier = Classifier(in_features=googlenet.fc.in_features, 
                                         out_features=n_classes, 
                                         hidden_layers=h_layers[0])
            mobilenet_classifier = Classifier(in_features=mobilenetv2.classifier[1].in_features, 
                                              out_features=n_classes, 
                                              hidden_layers=h_layers[1])
            resnet_classifier = Classifier(in_features=resnet34.fc.in_features, 
                                           out_features=n_classes, 
                                           hidden_layers=h_layers[2])
        else:
            gnet_classifier = Classifier(in_features=googlenet.fc.in_features, 
                                         out_features=n_classes, 
                                         hidden_layers=h_layers)
            mobilenet_classifier = Classifier(in_features=mobilenetv2.classifier[1].in_features, 
                                              out_features=n_classes, 
                                              hidden_layers=h_layers)
            resnet_classifier = Classifier(in_features=resnet34.fc.in_features, 
                                           out_features=n_classes, 
                                           hidden_layers=h_layers)
        
        cnn_models = [googlenet, mobilenetv2, resnet34]
        # Freeze architecture parameters to avoid backpropagating them
        # Avoiding replacing pretrained weights
        for model in cnn_models:
            for param in model.parameters():
                param.requires_grad = False
        
        # Replace last FC/classifier with new classifier
        googlenet.fc = gnet_classifier
        mobilenetv2.classifier = mobilenet_classifier
        resnet34.fc = resnet_classifier

        return cnn_models
    
    def calc_params(self, model_names, n_classes, h_layers):
        """
        Used to calculate the amount of trainable parameters vs total parameters for each model.
        
        Parameters:
            model_names (list) - a list of the model names
            n_classes (int) - number of output classes
            h_layers (list) - hidden node integers, one per layer
        """
        models = self.set_initial_models(n_classes, h_layers)
        
        # Total params for each model
        for idx, model in enumerate(models):
            print(f"{model_names[idx]}:")
            model.total_params = sum(p.numel() for p in model.parameters())
            print(f'{model.total_params:,} total parameters')
            model.trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f'{model.trainable_params:,} training parameters\n')
    
    def _set_filename(self, model_name, batch_size, h_layers):
        """
        Helper function used to set the saved models filename. Returns the filename.
        
        Parameters:
            model_name (string) - name of the model
            batch_size (int) - batch size of data loader
            h_layers (list) - hidden node integers, one per layer
        
        Format: [model_name]_[batch_size]_[hidden_sizes]
        """
        filename = f"{model_name}_{str(batch_size)}_"
        for layer in h_layers:
            filename += f"{str(layer)}_"
        return filename[:-1]
        
    
    def tune_model(self, model_names, batch_size, train_loader, valid_loader, 
                   n_classes, h_layers, lr, epochs=1000, iterations=2, patience=5):
        """
        Used to tune a model on the given training loader, evaluated against the validation loader. Iterates over a list of hidden layers, saving multiple model versions.
        
        Parameters:
            model_names (list) - a list of the model names as strings
            batch_size (int) - batch size of the train and validation loader
            train_loader (torch.DataLoader) - torch training dataset loader
            valid_loader (torch.DataLoader) - torch validation dataset loader
            n_classes (int) - number of output classes
            h_layers (list) - lists of a variety of hidden node sizes
            lr (float) - learning rate for training the model 
            epochs (int) - number of epochs for training (default: 1000)
            iterations (int) - iterations per number of epochs (default: 2)
            patience (int) - number of epochs to wait before early stopping (default: 5)
        """
        # Iterate over hidden layers
        for l in range(len(h_layers)):
            # Create instances of pretrained CNN architectures
            models = self.set_initial_models(n_classes, h_layers[l])

            # Iterate over models
            for m in range(len(models)):
                filename = self._set_filename(model_names[m], batch_size, h_layers[l])
                filepath = f"saved_models/{filename}.pt"
                
                # Skip model training if already has been trained
                if os.path.isfile(filepath):
                    print(f"{filename} already trained.")
                else:
                    print(f"\nTraining: {filename}")
                    criterion = nn.NLLLoss() # Negative Log Likelihood Loss

                    # Set optimizer
                    if m == 1: # MobileNetV2 specific
                        optimizer = torch.optim.Adam(models[m].classifier.parameters(), 
                                                 lr=lr)
                    else:
                        optimizer = torch.optim.Adam(models[m].fc.parameters(), 
                                                 lr=lr)

                    models[m].to(self.device) # move to GPU

                    # Train model
                    self.utils.train(models[m], train_loader, valid_loader, criterion, 
                                     optimizer, filepath, epochs, iterations, patience)
                    
    def set_model(self, model_paths, model, model_name):
        """
        Used to check what type of model needs to be set for testing. Returns the model and its name.
        
        Name format: [model]_[batch_size]_[hidden_size]_[hidden_size]
        
        Parameters:
            model_paths (list) - list of filepaths of saved models
            model (torchvision.models) - initial pretrained model
            model_name (string) - name of the model
        """
        # Set initial variables
        load_name = ""
        compare_parts = [model_name, self.utils.batch_size]
        compare_parts.extend(self.h_layers)
        
        # Iterate over each model
        for filepath in model_paths:
            compare_name = filepath.split('/')[-1].rstrip('.pt').split('_')
            valid = []
            
            # Check components match
            for item in range(len(compare_name)):
                if compare_name[item] == str(compare_parts[item]):
                    valid.append(True)
            
            # Load saved model
            if len(valid) == len(compare_name):
                load_name = filepath.split('/')[-1].rstrip('.pt')
                self.utils.load_model(model, f'saved_models/{filepath}')
                break
        
        return model, load_name
    
    def save_best_models(self, model_stats, model_names, n_preds):
        """
        Used to save the three best performing models based on the statistics of all model variations. Returns a list of the best models.
        
        Parameters:
            model_stats (pandas.DataFrame) - table of best model statistics
            model_names (list) - model names as strings
            n_preds (int) - number of additional predictions to store (e.g. top-5)
        """
        best_models = []
        n_models = len(model_names)
        count = 1
        start_time = time.time()
        
        # Iterate over each model
        for idx, item in enumerate(model_stats['Name']):
            name, batch = item.split('_')[:2]
            h_layers = list(map(int, item.split("_")[2:]))
            filepath = f'saved_models/{item}.pt'
            cnn_models = self.set_initial_models(self.n_classes, h_layers)

            # Check names match
            for cnn_name in model_names:
                if name == cnn_name:
                    # Load model and store it
                    model = cnn_models[idx]
                    self.utils.load_model(model, filepath)
                    best_models.append(model)
                    filename = cnn_name.replace('-', '').lower()
                    
                    # Set statistics
                    stats = model_stats.iloc[idx, 1:].to_dict()
                    
                    # Set additional model parameters
                    model.batch_size = int(batch)
                    model.h_layers = h_layers
                    model.stats = stats
                    
                    # Save model predictions
                    print(f"Calculating preds and stats for {name}...", end=" ")
                    _, _, test_loader = self.utils.split_data(self.utils.dataset, int(batch), 
                                                              self.utils.split_size, 
                                                              self.utils.seed)
                    self._save_predictions(model, test_loader, n_preds)
                    print(f"Complete ({count}/{n_models}).")
                    count += 1
                    
                    # Save as best model
                    print(f"Saving model...", end=" ")
                    self._save_model(model, filename)
                    print(f"Complete.")
        
        self.utils.time_taken(time.time() - start_time)
        return best_models
    
    def _save_predictions(self, model, test_loader, n_preds):
        """
        Helper function used to save the best models predictions, labels and probabilities for plotting.
        
        Parameters:
            model (torchvision.models) - models predictions to save
            valid_loader (torch.DataLoader) - torch test dataset loader
            n_preds (int) - number of additional predictions to store (e.g. top-5)
        """
        # Calculate predictions, labels and probabilities for best models
        y_pred, y_true, all_n_preds, y_probas = self.utils.predict(model, test_loader, 
                                                                   n_preds,
                                                                   store_labels=True,
                                                                   store_probas=True)
        # Store data
        model.y_pred = y_pred
        model.y_true = y_true
        model.n_preds = all_n_preds
        model.y_probas = y_probas
    
    def _save_model(self, model, filename):
        """
        Helper function used to save the best models.
        
        Parameters:
            model (torchvision.models) - model to save
            filename (string) - filename of model to save
        """
        torch.save({'parameters': model.state_dict(),
                    'train_losses': model.train_losses,
                    'valid_losses': model.valid_losses,
                    'batch_size': model.batch_size,
                    'h_layers': model.h_layers,
                    'stats': model.stats,
                    'y_pred': model.y_pred,
                    'y_true': model.y_true,
                    'n_preds': model.n_preds,
                    'y_probas': model.y_probas,
                    }, f'saved_models/best_{filename}.pt')
        
    def load_best_models(self, models, filenames):
        """
        Used to load the three best models.
        
        Parameters:
            model (list) - torchvision.models to load
            filenames (list) - filenames of saved models to load within saved_models folder
        """
        # Set a checkpoint
        for idx, model in enumerate(models):
            checkpoint = torch.load(f"saved_models/{filenames[idx]}.pt")

            # Store utility variables
            model.train_losses = checkpoint['train_losses']
            model.valid_losses = checkpoint['valid_losses']
            model.batch_size = checkpoint['batch_size']
            model.h_layers = checkpoint['h_layers']
            model.stats = checkpoint['stats']
            model.y_pred = checkpoint['y_pred']
            model.y_true = checkpoint['y_true']
            model.n_preds = checkpoint['n_preds']
            model.y_probas = checkpoint['y_probas']

            # load model parameters
            model.load_state_dict(checkpoint['parameters'])
        
        print("Models loaded. Utility variables available:")
        print("\ttrain_losses, valid_losses, batch_size, h_layers, stats,\n")
        print("\ty_pred, y_true, n_preds, y_probas.")