import numpy as np
import matplotlib.pyplot as plt

from functions.tuning import Tuner

from scikitplot.metrics import plot_confusion_matrix, plot_roc

class Plotter():
    """
    Used to create plots and visualise the given dataset.
    
    Parameters:
        labels (numpy.array) - list of plant class labels as strings
    """
    def __init__(self, labels):
        self.class_labels = labels
        self.class_labels_idx = np.array(range(len(labels)))
        self.tune = Tuner()
    
    def imshow(self, img):
        """
        Used to un-normalize and display an image.
        """
        img = img / 2 + 0.5 # Un-normalize
        plt.imshow(np.transpose(img, (1, 2, 0))) # Convert from tensor image

    def visualize_imgs(self, imgs, labels, figsize=(25, 15), num_rows=5, num_cols=35):
        """
        Creates subplots of a sample of images.
        
        Parameters:
            imgs (numpy.array) - batch of images
            labels (numpy.array) - batch of image labels
            figsize (tuple) - subplot figure size
            num_rows (int) - number of rows in subplots
            num_cols (int) - number of columns in subplots
        """
        fig = plt.figure(figsize=figsize)
        
        for idx in np.arange(num_cols):
            ax = fig.add_subplot(num_rows, int(np.ceil(num_cols/num_rows)), idx+1, 
                                 xticks=[], yticks=[])
            self.imshow(imgs[idx])
            ax.set_title(self.class_labels[labels[idx]])
    
    def create_plots(self, models, model_names, figsize, plot_func, plot_name=None, save=False):
        """
        Dynamically creates the correct amount of plots depending on number of models
        passed in.
        
        Parameters:
            models (list) - one or more torchvision.models
            model_names (list) - name of models as strings
            figsize (tuple) - size of each subplot figure
            plot_func (function) - type of plot to create
            plot_name (string) - plot name for saving
            save (boolean) - when true saves plot to plot folder
        """
        # Create subplots
        if isinstance(models, list):
            fig = plt.figure(figsize=figsize)
            fig.subplots_adjust(wspace=0.25)
            num_cols = len(models)
            # Create individual plot
            for idx in np.arange(num_cols):
                fig.add_subplot(1, num_cols, idx+1)
                plot_func(models[idx], model_names[idx])
            plt.show()
            
        # Create single plot
        else:
            fig = plt.figure()
            plot_func(models, model_names)
            plt.show()
        
        # Save plot
        if plot_name is not None and save:
            fig.savefig(f"plots/{plot_name}.png")
    
    def plot_losses(self, model, model_name):
        """
        Creates a plot of the models training loss and validation loss against
        the amount of epoch iterations. Takes in a model as input.
        
        Parameters:
            model (torchvision.models) - model for plotting
            model_name (string) - name of model
        """        
        # Create plot
        epochs = range(len(model.train_losses))
        line1 = plt.plot(epochs, model.train_losses, label='training loss')
        line2 = plt.plot(epochs, model.valid_losses, label='validation loss')
        plt.xlabel("Iterations")
        plt.ylabel("Losses")
        plt.legend(loc="upper right")
        plt.title(f"{model_name} Loss Comparison")
    
    def plot_cm(self, model_name, y_pred, y_true, save=False):
        """
        Creates a confusion matrix for the given model.
        
        Parameters:
            model (torchvision.models) - models to evaluate
            model_name (string) - name of model
            y_pred (torch.Tensor) - test or validation loader predictions
            y_true (torch.Tensor) - dataloader labels
        """
        # Plot confusion matrix
        plot_confusion_matrix(y_true, y_pred, figsize=(25, 15), labels=self.class_labels,
                              x_tick_rotation=90, title=f"{model_name} Confusion Matrix")
        if save:
            plt.savefig(f'plots/{model_name}_cm.png')

    def plot_roc(self, model_name, y_probas, y_true, figsize=(25, 15), save=False):
        """
        Creates multiple subplots of the ROC curve for each classes using the given model.
        
        Parameters:
            model (torchvision.models) - models to evaluate
            model_name (string) - name of model
            y_probas (torch.Tensor) - test or validation loader probabilities
            y_true (torch.Tensor) - dataloader labels
            figsize (tuple) - subplot figure size
        """
        plot_roc(y_true, y_probas, figsize=figsize, title=f"{model_name} ROC Plots")
        
        if save:
            plt.savefig(f'plots/{model_name}_roc.png')
    
    def plot_stats(self, model_stats):
        """
        Displays a table of the given models statistics.
        
        Parameters:
            model_stats (list/dictionary) - a list or single dict of statistics of trained models
        """
        # Set initial variables
        headers = list(model_stats[0].keys())
        table = pd.DataFrame(columns=headers)
        
        # Add rows if list
        if isinstance(model_stats, list):
            for i in range(len(model_stats)):
                table = table.append(model_stats[i], ignore_index=True)
        # Add row if single dict
        else:
            table = table.append(model_stats, ignore_index=True)
        
        return table
    
    def plot_model_predictions(self, models, model_names, batch_sizes, filepath, split_size,
                               seed, save=False, n_rows=2, n_cols=20):
        """
        Used to plot each models image classification predictions. Images are labelled with the prediction and the true label in brackets. A green name means the prediction is correct, otherwise it is red.
        Parameters:
            models (list) - best torchvision.models
            model_names (list) - a list of the model names as strings
            batch_sizes (list) - integers of images per batch for each model
            filepath (string) - filepath to dataset of images
            split_size (float) - size of split for both the test and validation sets
            seed (int) - number for recreating previous instances
            save (boolean) - If true, saves the plots created to plots folder
            n_rows (int) - number of rows in subplots
            n_cols (int) - number of columns in subplots (images to show)
        """
        dataset = self.tune.set_data(filepath)
        temp = seed
        
        # Iterate over models and batches
        for m, model in enumerate(models):
            for batch in batch_sizes:
                # Check batch size matches
                if model.batch_size == batch:
                    _, _, test_loader = self.tune.utils.split_data(dataset, batch, 
                                                                   split_size, temp)
                    # Obtain one batch of test images
                    dataiter = iter(test_loader)
                    imgs, lbls = dataiter.next()

                    # Get predictions
                    model.cpu()
                    preds = torch.exp(model.forward(imgs)).max(dim=1)[1].numpy()
                    imgs = imgs[:n_cols]

                    # Plot n_cols images in the batch, along with predicted and true labels
                    fig = plt.figure(figsize=(25, 4))
                    fig.suptitle(f"{model_names[m]} Predictions vs True Labels")
                    for idx in np.arange(n_cols):
                        ax = fig.add_subplot(n_rows, int(np.ceil(n_cols/n_rows)), idx+1, 
                                             xticks=[], yticks=[])
                        ax.imshow(self.tune.unnorm(imgs[idx]).permute(1, 2, 0).numpy())
                        ax.set_title(f"{self.class_labels[preds[idx]]}\n" \
                                     f"({self.class_labels[lbls[idx]]})",
                                     color=("green" if preds[idx] == lbls[idx] else "red"))
                    fig.subplots_adjust(top=0.8, hspace=0.55)
            
                    if save:
                        plt.savefig(f'plots/{model_names[m]}_preds.png')

            temp += 1 # Obtain different batch for plotting next model