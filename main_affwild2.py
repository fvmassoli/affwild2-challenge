import os
import sys
import logging
import argparse
import numpy as np
import pandas as pd
from os.path import join, exists
from prettytable import PrettyTable

import torch
from torch.optim import SGD, Adam
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from models import ModelLoader
from trainer import train, test
from datasets import AffWild2Dataset
from utils import create_experiment_folder, get_transforms


def main(args):
    ## Init seed 
    if args.seed != -1:
        # cudnn.benchmark = True
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    # Set the device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if not args.test:
        # Create output folders
        output_main_folder, exp_folder = create_experiment_folder(args=args, dataset_name='affwild2')
        
        logging.basicConfig(
                        level=logging.INFO,
                        format="%(asctime)s | %(message)s",
                        handlers=[
                            logging.FileHandler(os.path.join(exp_folder, 'training.log')),
                            logging.StreamHandler()
                        ])
        # logger = logging.getLogger()
        
        # Init tensorboard writer
        tb_writer = SummaryWriter(join(output_main_folder, 'tb_run', exp_folder.split('/')[-1]))

    ## Load the model 
    model_loader = ModelLoader(
                            model_base_path=args.model_base_path, 
                            num_classes=args.num_classes, 
                        )

    if args.model_checkpoint is not None:
        model_loader.load_model_checkpoint(args.model_checkpoint)

    #########################
    #### Train the model ####
    #########################
    if args.train:

        if args.train_mode == 'transfer':
            model_loader.freeze_params()

        # Get the model
        model = model_loader.get_model()

        ## Init the optimizer
        optim_kwargs = {
                    'weight_decay': args.weight_decay
                }
        if args.optimizer == 'sgd':
            opt_fn = SGD
            optim_kwargs.update({
                'momentum': args.momentum,
                'nesterov': args.nesterov
            })
        else:
            opt_fn = Adam
            optim_kwargs.update({
                'betas':(0.9, 0.999)
            })
        
        if args.train_mode == 'finetune':
            # Train the entire model 
            optimizer = opt_fn([
                            {'params': torch.nn.Sequential(*(list(model.children())[:-1])).parameters(), 'lr': args.lr*0.01},
                            {'params': model.classifier_1.parameters()}
                        ],  
                        lr=args.lr, 
                        **optim_kwargs
                    )
        else:
            # ONLY train the classifier
            optimizer = opt_fn(params=model.classifier_1.parameters(), lr=args.lr, **optim_kwargs)

        ## Load the data
        tr_dataset = AffWild2Dataset(
                            dataset_folder=args.dataset_folder, 
                            output_folder=join(args.dataset_folder, 'database'), 
                            multi_res=args.multi_res_training,
                            valid_resolution=args.valid_resolution if args.multi_res_training else -1,
                            transforms=get_transforms(mode='train')
                        )
        
        tr_dataset.set_mode(train=True)
        train_loader = DataLoader(
                            dataset=tr_dataset, 
                            batch_size=args.batch_size,
                            shuffle=True, 
                            num_workers=8, 
                            pin_memory=device=='cuda'
                        )
        
        # Get classes weigths to weight the training loss to take care of the classes unbalance 
        # The call to this method must follow the one to the set_mode() for the training dataset
        # otherwise the weights are evaluated on the whole dataset and not on training images only
        training_classes_weights = tr_dataset.get_training_classes_weights()
        
        vl_dataset = AffWild2Dataset(
                            dataset_folder=args.dataset_folder, 
                            output_folder=join(args.dataset_folder, 'database'), 
                            multi_res=args.multi_res_training,
                            valid_resolution=args.valid_resolution if args.multi_res_training else -1,
                            test=True, # It simply disables some print statements
                            transforms=get_transforms(mode='valid')
                        )

        vl_dataset.set_mode(train=False)
        valid_loader = DataLoader(
                            dataset=vl_dataset, 
                            batch_size=args.batch_size,
                            num_workers=8, 
                            pin_memory=device=='cuda'
                        )

        ## Start training
        train(
            model=model, 
            loader=train_loader, 
            valid_loader=valid_loader, 
            optimizer=optimizer, 
            training_classes_weights=training_classes_weights,
            step_size=args.step_size, 
            epochs=args.epochs, 
            train_iters=args.train_iters, 
            batch_accumulation=args.batch_accumulation, 
            tb_writer=tb_writer, 
            log_freq=args.log_freq, 
            output_folder_path=exp_folder, 
            device=device
        )

    ########################
    #### Test the model ####
    ########################
    if args.test:

        model_checkpoint = []
        training_mode = []
        resolution = []
        accuracy = []
        f1_score = []
        challenge_score = []
        
        # -1 means that the test is performed on the images at the original reaolution 
        valid_res_list = [7, 14, 28, 56] if args.multi_res_test else [-1] 

        for valid_res in valid_res_list:
            
            ## Load the data
            tt_dataset = AffWild2Dataset(
                            dataset_folder=args.dataset_folder, 
                            output_folder=join(args.dataset_folder, 'database'), 
                            multi_res=valid_res!=-1,
                            valid_resolution=valid_res,
                            test=True, # It simply disables some print statements
                            transforms=get_transforms(mode='valid')
                        )

            tt_dataset.set_mode(train=False)
            test_loader = DataLoader(
                                dataset=tt_dataset, 
                                batch_size=args.batch_size,
                                num_workers=8, 
                                pin_memory=device=='cuda'
                            )   
            
            ## Start testing
            accuracy_, f1_score_, challenge_score_ = test(model=model_loader.get_model(), loader=test_loader, device=device)

            model_checkpoint.append(args.model_checkpoint)
            training_mode.append(args.model_checkpoint.split('/')[-2].split('-')[6])
            resolution.append(valid_res)
            accuracy.append(accuracy_)
            f1_score.append(f1_score_)
            challenge_score.append(challenge_score_)
            
        df = pd.DataFrame(
                    data=dict(
                            model_checkpoint=np.asarray(model_checkpoint),
                            training_mode=np.asarray(training_mode),
                            resolution=np.asarray(resolution),
                            accuracy=np.asarray(accuracy),
                            f1_score=np.asarray(f1_score),
                            challenge_score=np.asarray(challenge_score)
                        ),
                    index=None
                )

        output_csv_path = './test_results.csv'
        df.to_csv(output_csv_path, mode='a', header=not exists(output_csv_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('CRFER')

    # General
    parser.add_argument('-s', '--seed', type=int, default=-1)
    
    # Optimizer
    parser.add_argument('-o', '--optimizer', choices=('adam', 'sgd'), default='adam', help='Type of optimizer (default: adam)')
    # parser.add_argument('-lr', '--lr', type=float, nargs='+', default=[])
    parser.add_argument('-lr', '--lr', type=float, default=1.e-2, help='Learning rate (default: 1.e-2)')
    parser.add_argument('-wd', '--weight_decay', type=float, default=1.e-4, help='Weight decay (default: 1.e-4)')
    parser.add_argument('-m', '--momentum', type=float, default=0.9, help='Momentum of the SGD optimizer (default: 0.9)')
    parser.add_argument('-nt', '--nesterov', action='store_true', help='Use Nesterov with the SGD optimizer (default: False)')
    
    # Model selection
    parser.add_argument('-ck', '--model_checkpoint', help='Path to model checkpoint')
    parser.add_argument('-bp', '--model_base_path', help='Path to base model checkpoint')
    parser.add_argument('-nc', '--num_classes', type=int, default=7, help='Number of training classes (default: 7)')
    parser.add_argument('-tm', '--train_mode', choices=('finetune', 'transfer'), default='transfer', help='Set the training mode (default: transfer)')

    # Training
    parser.add_argument('-df', '--dataset_folder', help='Path to main data folder')
    parser.add_argument('-st', '--step_size', type=int, default=10, help='Number of valid iterations before to drop the learning rate (default: 10)')
    parser.add_argument('-e', '--epochs', type=int, default=1, help='Training epochs (default: 1)')
    parser.add_argument('-ti', '--train_iters', type=int, default=10, help='Number of training iterations before a validation run (default: 10)')
    parser.add_argument('-ba', '--batch_accumulation', type=int, default=4, help='Number of batch accumulation iterations (default: 4)')
    parser.add_argument('-bs', '--batch_size', type=int, default=32, help='Batch size (default: 32)')
    parser.add_argument('-lf', '--log_freq', type=int, default=100, help='Log frequency (default: 100)')
    parser.add_argument('-op', '--output_folder_path', default='./training_output', help='Path to output folder (default: ./training_output)')
    parser.add_argument('-tr', '--train', action='store_true', help='Train the model (default: False)')
    parser.add_argument('-mtr', '--multi_res_training', action='store_true', help='Multi-resolution training (default: False)') 
    parser.add_argument('-vr', '--valid_resolution', type=int, default=-1, help='Resolution of validation images in the multi-resolution training (default: -1)')
    parser.add_argument('-tt', '--test', action='store_true', help='Test the model (default: False)') 
    parser.add_argument('-mtt', '--multi_res_test', action='store_true', help='Multi-resolutoin test (default: False)') 
    
    args = parser.parse_args()

    main(args)
