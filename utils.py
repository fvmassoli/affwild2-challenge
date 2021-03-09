import os
import argparse
import prettytable
import numpy as np
from os.path import join, exists


import sklearn.metrics

import torch

import torchvision
from torchvision.transforms import Compose, Resize, RandomGrayscale, RandomCrop, CenterCrop, ToTensor, Lambda


def create_experiment_folder(args: argparse.Namespace, dataset_name: str) -> str:
    """Create the current output experiment folder.

    Parameters
    ----------
        args : 
            Argument given by the user
        dataset_name : str
            Name of the dataset

    Returns
    ------
    exp_folder : str
        Path to current experiment output folder

    """
    output_main_folder = args.output_folder_path
    
    if not exists(output_main_folder):
        os.makedirs(output_main_folder)

    tmp = '-'.join([str(args.multi_res_training), str(args.lr), str(args.weight_decay), str(args.momentum), str(args.nesterov), args.train_mode, dataset_name, str(args.step_size), str(args.batch_accumulation), str(args.batch_size)])
    exp_folder = join(output_main_folder, tmp)

    if not exists(exp_folder):
        os.makedirs(exp_folder)

    return output_main_folder, exp_folder


def create_output_test_folder(output_folder_path, dataset_name):
    """Create the test output folder.

    Parameters
    ----------
        output_folder_path : str
            Where to create the folder
        dataset_name : str
            Name of the dataset

    Returns
    ------
    exp_folder : str
        Path to current experiment output folder

    """
    outf = join(output_folder_path, dataset_name)
    if not exists(outf):
        os.makedirs(outf)
    return outf 


def get_transforms(mode: str, resize: int = 256, grayed_prob: float = 0.2, crop_size: int = 224) -> torchvision.transforms.Compose:
    """Returns composition of augmentation transformation.

    Parameters
    ----------
    mode : str
        Training mode, i.e., train or valid/test
    resize: int
        Side dimension to which resize the images
    grayed_prob : float
        Probability to gray an image
    crop_size : int
        Size of the image crop

    Returns
    -------
    composed treansformations : Compose
        Composition of transformation

    """
    def subtract_mean(x: torch.Tensor) -> torch.Tensor:
        """Subtract the mean for each channel.
        These values are given in the original VGGFace2 (Cao et al.) model.
        
        Parameters
        ----------
        x : torch.Tensor
            The input image tensor

        Returns
        -------
        x : torch.Tensor
            Nomralized tensor image

        """
        mean_vector = [91.4953, 103.8827, 131.0912]
        x *= 255.
        if x.shape[0] == 1:  
            x = x.repeat(3, 1, 1)
        x[0] -= mean_vector[0]
        x[1] -= mean_vector[1]
        x[2] -= mean_vector[2]
        return x
    if mode=='train':
        return Compose([
                    Resize(resize),
                    RandomGrayscale(p=grayed_prob),
                    RandomCrop(crop_size),
                    ToTensor(),
                    Lambda(lambda x: subtract_mean(x))
                ])
    else:
        return Compose([
                    Resize(resize),
                    CenterCrop(crop_size),
                    ToTensor(),
                    Lambda(lambda x: subtract_mean(x))
                ])


def eval_metrics(labels: torch.tensor, predictions: torch.tensor) -> [float, float, sklearn.metrics]:
    """Eval training metrics.
    
    Parameters
    ----------
    labels : torch.tensor
        Ground truth predictions
    predictions : torch.tensor
        Model predictions

    Returns
    ------
    accuracy : float
        Accuracy of the model
    f1_score : float 
        F1 score of the model
    confusion_matrix : sklearn.metrics
        Confusion matrix among all 7 expression classes

    """
    # Eval F1 score (macro-averaged)
    f1_score = sklearn.metrics.f1_score(labels, predictions, average='macro', zero_division=1)
    
    # Eval F1 score for each class
    classes_score = sklearn.metrics.f1_score(labels, predictions, average=None, zero_division=1)
    print(f'Acc classes: {classes_score}')

    # Eval accuracy
    accuracy = sklearn.metrics.accuracy_score(labels, predictions)
    
    # Eval the confusion matrix
    confusion_matrix = sklearn.metrics.confusion_matrix(labels, predictions)

    return accuracy, f1_score, confusion_matrix


def eval_score_for_competition(f1_score: float, accuracy: float) -> float:
    """Evaluate the statistics required from the AffWild2 competition.

    Parameters
    ----------
    f1_score : float
        F1 score of the classifier
    accuracy : float
        Accuracy of the classifier 

    Returns
    ------
    stat : float
        Competition statistics

    """
    return (0.33*accuracy) + (0.67*f1_score)
    