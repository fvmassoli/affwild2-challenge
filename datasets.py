import os
import PIL
import cv2
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from os import listdir
from os.path import join, exists
from prettytable import PrettyTable, from_csv

import torchvision

import torch
from torch.utils.data import Dataset

from utils import get_transforms


def get_dataset(dsetname: str, dataset_folder: str, output_folder: str, mode: str, transforms: torchvision.transforms = None) -> torch.utils.data.Dataset:
    """Init the dataset class for the required dataset.
    
    Parameters
    ----------
    dsetname : str
        Name fo the dataset
    dataset_folder : str
        Path to where the dataset is stored.
        Depending on the dataset the dataset_folder 
        can be the path to a folder or to a .csv file
    output_folder : str
        Path to save dataset database
    mode : str
        Some datasets requires to know the mode, i.e., train/validation/test
    transforms : torchvision.transforms
        Composition of transformations

    Returns
    -------
    dset : torch.utils.data.Dataset
        Dataset

    """
    if self.dsetname == 'fer2013':
        dset = FER2013(database_path=dataset_folder, transforms=transforms)
        dset.set_usage(mode)
    if self.dsetname == 'expw':
        dset = ExpW(dataset_folder=dataset_folder, output_folder=output_folder, transforms=transforms)
    if self.dsetname == 'rafdb':
        dset = RAFdb(dataset_folder=dataset_folder, output_folder=output_folder, transforms=transforms)
    if self.dsetname == 'oulucasia':
        dset = OuluCasia(dataset_folder=dataset_folder, output_folder=output_folder, transforms=transforms)
    else:
        raise ValueError(f'The dataset -- {dsetname} -- does not exists!')
    return dset


class BaseDataset(Dataset):
    """Base class for all datasets.

    """
    def __init__(self, dataset_folder: str, output_folder: str, transforms: torchvision.transforms = None):
        """Init the base dataset class.

        Parameters
        ----------
        dataset_folder : str
            Path to folder cotaining the dataset
        output_folder : str
            Path to output folder where to save the database 
        transforms : torchvision.transforms
            Data augmentation transforms
        
        """
        super(BaseDataset, self).__init__()
        
        self.dataset_folder = dataset_folder
        self.output_folder = output_folder
        self.transforms = transforms
        
        self.loader = self.__get_loader
        self.logger = logging.getLogger()
        
    def get_emotion_from_idx(self, label: int) -> str:
        """Returns the expression corresponding to the given label.
        
        Parameters
        ----------
        label : int
            Label of the required expression

        Returns
        ------
        emotion : str 
            Name of the expression that correspond to the given label 

        """
        return self.idx_to_expression[label]
    
    def get_training_classes_weights(self) -> np.array:
        """
        
        Returns
        ------
        weights : np.array
            Numpy array contatining the normalized weight for each training class

        """
        expressions = self.curr_db.expression.unique()
        expressions.sort()
        weigths = np.asarray([len(self.curr_db[self.curr_db['expression']==expr]) for expr in expressions])
        normed_weights = np.asarray([1 - (w / sum(weigths)) for w in weigths])
        
        # Print weights for each class
        stats_table = PrettyTable()
        stats_table.title = 'Weights for each expression (Training)'
        stats_table.field_names = [self.idx_to_emotion[int(label)] for label in expressions]
        stats_table.add_row([nw for nw in normed_weights])
        print(stats_table)
        
        return normed_weights

    @staticmethod
    def __get_loader(path: str) -> PIL.Image:
        """Set the image loader... 
        WARNING: the images are loaded as BGR due to the use of cv2.imread()
        
        Returns
        ------
        image : PIL.Image
            Image loaded from array

        """
        return Image.fromarray(cv2.imread(path))

    def __len__(self) -> int:
        """Returns the length of the database in the current mode.
        
        Returns
        ------
        int : int
            The number of samples in the current mode

        """
        return len(self.db)


class AffWild2Dataset(BaseDataset):
    """Class to handle AffWild2 data.
    The Affwild2 dataset is used to train/valid the model since there is not any test set available yet
    
    """
    def __init__(self, dataset_folder: str, output_folder: str, multi_res: bool = False, valid_resolution: int = -1, test: bool = False, transforms: torchvision.transforms = None) -> None:
        """Init the dataset class.

        Parameters
        ----------
        dataset_folder : str
            Path to folder cotaining the dataset
        output_folder : str
            Path to output folder where to save the database 
        multi_res : bool
            If True, the model is trained with multi-resolution images
        valid_resolution : int
            Resolution at which downsample images while testing the model
        test : bool
            True if the model is under test
        transforms : torchvision.transforms
            Data augmentation transforms

        """
        super(AffWild2Dataset, self).__init__(dataset_folder=dataset_folder, output_folder=output_folder, transforms=transforms)

        self.multi_res = multi_res
        self.valid_resolution = valid_resolution
        self.test = test
        
        self.train = None
        self.curr_db = None

        self.idx_to_emotion = {
                            0: 'Neutral',
                            1: 'Anger',
                            2: 'Disgust',
                            3: 'Fear',
                            4: 'Happiness',
                            5: 'Sadness',
                            6: 'Surprise',
                            -1: 'None',
                        }

        # Create the database if it does not exists otherwise load it from .csv file
        db_fname = join(self.output_folder, 'affwild2_database.csv')
        if not exists(db_fname):
            if not exists(self.output_folder):
                os.makedirs(self.output_folder)
            self.logger.info('Databse not found... creating it')
            self.__create_database(db_fname=db_fname)
        
        self.db = pd.read_csv(db_fname)
        
        # Remove all the entries which have expression label equals to -1
        self.db = self.db[self.db['expression'] != -1]

        # Print data statistics
        self.__dataset_statistics()
        
    def __create_database(self, db_fname: str):
        """Create the database and save it in a .csv file.
        Following the instruction by the affwild2 team, the name of each frame corresponds
        to the position of the label in the corresponding file, i.e., the frame named video_id/00123.jpg
        correspond to the label in the position 123 of the annotation file video_id.txt.
        At the position 0 of each annotation file there is a string reporting the name of the expressions,
        that is why the frame always start from the number 00001.jpg
        
        Parameters
        ----------
        db_fname : str
            Path to output database file

        """
        train_frames_path = []
        train_video_id = []
        train_expr = []

        valid_frames_path = []
        valid_video_id = []
        valid_expr = []

        frames_base_path = join(self.dataset_folder, 'cropped_images_aligned/cropped_aligned/')

        for set_split in ['Training_Set', 'Validation_Set']:
            annotations_base_dir = join(self.dataset_folder, 'annotations/EXPR_Set', set_split)
            annotations_files = listdir(annotations_base_dir)

            for ff in tqdm(annotations_files, total=len(annotations_files), desc=f"Working on {set_split} set", leave=False):
                # read the annotation for each frame in the ff video
                annot = open(join(annotations_base_dir, ff), 'r')
                lines = np.asarray(list(map(lambda x: int(x.rstrip()) if 'Neutral' not in x else x.rstrip(), annot.readlines())))
                annot.close()

                # Convert the frame name into integers so to use them as indices to get the proper expression
                frames = list(map(lambda x: x.split('.')[0], filter(lambda x: '.jpg' in x, listdir(join(frames_base_path, ff.split('.')[0])))))
                frames_int = list(map(lambda x: int(x.split('.')[0]), filter(lambda x: '.jpg' in x, listdir(join(frames_base_path, ff.split('.')[0])))))
                
                if set_split == 'Training_Set':
                    train_video_id.extend([ff.split('.')[0] for _ in frames])
                    train_frames_path.extend([join(frames_base_path, ff.split('.')[0], frame+'.jpg') for frame in frames])
                    # use the indices into the 'frames' list to get the annotations for each face
                    train_expr.extend(lines[frames_int])
                else:
                    valid_video_id.extend([ff.split('.')[0] for _ in frames])
                    valid_frames_path.extend([join(frames_base_path, ff.split('.')[0], frame+'.jpg') for frame in frames])
                    # use the indices into the 'frames' list to get the annotations for each face
                    valid_expr.extend(lines[frames_int])

        train_expr = np.asarray(train_expr)
        train_video_id = np.asarray(train_video_id)
        valid_expr = np.asarray(valid_expr)
        valid_video_id = np.asarray(valid_video_id)

        # Create the dataframe and save it into a .csv file
        df = pd.DataFrame(
                    data=dict(
                            path=np.hstack([train_frames_path, valid_frames_path]),
                            video_id=np.hstack([train_video_id, valid_video_id]),
                            expression=np.hstack([train_expr, valid_expr]),
                            train=np.hstack([np.ones_like(train_expr, dtype=np.int), np.zeros_like(valid_expr, dtype=np.int)])
                        ),
                    index=None
                )
        df.to_csv(db_fname)
        self.logger.info(f'Databse saved at: {db_fname}')

    def __dataset_statistics(self) -> None:
        """Use PrettyTable package to print dataset info.
        
        """
        # Print the expressions
        print(f'\nAvailable expression: {self.idx_to_emotion.values()}\n')
        
        info_table = PrettyTable()
        info_table.title = 'Multi-resolution training info'
        info_table.field_names = ['Multi-resolution', 'Validation Resolution']
        info_table.add_row([self.multi_res, self.valid_resolution])
        print(info_table)

        # Print subsample from dataframe to show its structure
        stats_table = PrettyTable()
        stats_table.title = 'Database structure'
        stats_table.field_names = self.db.columns[1:] # We don't need the index
        for idx, row in self.db.iterrows():
            stats_table.add_row([row.path, row.video_id, row.expression, row.train])
            if idx == 3: break # idx starts at 1
        if not self.test: print(stats_table)
        
        # Print dataset cardinalities
        stats_table = PrettyTable()
        stats_table.title = 'Sample for training and validation'
        stats_table.field_names = ['Train samples', 'Valid samples', 'Number of expressions']
        stats_table.add_row([len(self.db[self.db['train'] == 1]), len(self.db[self.db['train'] == 0]), len(self.db.expression.unique())])
        print(stats_table)
        
        def eval_class_cardinality(self, expressions: list, train: bool) -> [list, list]:
            cardinality = [len(self.db[(self.db['expression']==expr) & (self.db['train']==int(train))]) for expr in expressions]
            frac = [len(self.db[(self.db['expression']==expr) & (self.db['train']==int(train))])/len(self.db[self.db['train']==int(train)]) for expr in expressions]
            return cardinality, frac

        expressions = self.db.expression.unique()
        expressions.sort()
        
        # Print expression statistics --> Training split
        stats_table = PrettyTable()
        stats_table.title = 'Samples for each expression (Training)'
        stats_table.field_names = [self.idx_to_emotion[int(label)] for label in expressions]
        cardinality, frac = eval_class_cardinality(expressio=expressions, train=True)
        stats_table.add_row(cardinality)
        stats_table.add_row(frac)
        
        # Print expression statistics --> Validation split
        stats_table2 = PrettyTable()
        stats_table2.title = 'Samples for each expression (Validation)'
        stats_table2.field_names = [self.idx_to_emotion[int(label)] for label in expressions]
        cardinality, frac = eval_class_cardinality(expressions=expressions, train=False)
        stats_table2.add_row(cardinality)
        stats_table2.add_row(frac)

        if not self.test: 
            print(stats_table)
            print(stats_table2)

    def __lower_resolution(self, img: PIL.Image) -> PIL.Image:
        """Resize image to a random resolution.
        
        Parameters
        ----------
        img : PIL.Image
            Image to be resized

        Returns
        ------
        img : PIL.Image
            Resized image
        
        """
        w_i, h_i = img.size
        r = h_i/float(w_i)
        if self.train:
            res = torch.rand(1).item()
            res = 3 + 5*res
            res = 2**int(res)
        else:
            res = self.valid_resolution
        if res >= w_i or res >= h_i:
            return img
        if h_i < w_i:
            h_n = res
            w_n = h_n/float(r)
        else:
            w_n = res
            h_n = w_n*float(r)
        img2 = img.resize((int(w_n), int(h_n)), Image.BILINEAR)
        img2 = img2.resize((w_i, h_i), Image.BILINEAR)
        return img2
        
    def set_mode(self, train: bool) -> None:
        """Set the database to train or valid.

        Parameters
        ----------
        train : bool
            Set train or valid mode to load data

        """
        self.train = train
        self.curr_db = self.db[self.db['train'] == int(train)]

    def __len__(self) -> int:
        """Returns the length of the database in the current mode.
        
        Returns
        ------
        int : int
            The number of samples in the current mode

        """
        return len(self.curr_db)

    def __getitem__(self, idx: int) -> [torch.Tensor, int]:
        """Load data. 
        
        Parameters:
        -----------
        idx : int
            Index of the sample to load

        Rerturn
        -------
        image : torch.Tensor
            The loaded tensor
        expression : int
            The label correspoding to the face expression
        train : int
            Integer that indicates if dataset is in train mode (1) or not (0)

        """
        if self.train is None:
            raise ValueError('You need to first call the set_mode() method')

        path = self.curr_db.path.iloc[idx]
        expression = self.curr_db.expression.iloc[idx]
        train = self.curr_db.train.iloc[idx]

        # Load the image as a PIL.Image object
        img = self.loader(path)
        
        # Downsample the image if multi-resolution training
        if self.multi_res:
            img = self.__lower_resolution(img=img)
        
        # Apply data augmentation transformation
        img = img if self.transforms is None else self.transforms(img)
        
        return img, expression, train


class FER2013(BaseDataset):
    """Class to handle FER2013 data.
    The FER2013 dataset is used ONLY to TEST the models
    
    """
    def __init__(self, database_path: str, transforms: torchvision.transforms = None):
        """Init the dataset class.

        Parameters
        ----------
        database_path : str
            Path to the csv file containing the whole database
        transforms : torchvision.transforms
            Data augmentation transforms

        """
        super(FER2013, self).__init__(dataset_folder=None, output_folder=None, transforms=transforms)

        self.database_path = database_path
        self.transforms = transforms
        self.db = pd.read_csv(self.database_path)
        # Rename columns so to have always same across datasets
        self.db = self.db.rename(columns={'emotion': 'expression', self.db.columns[1]: 'Usage', self.db.columns[-1]: 'pixels'})

        self.idx_to_expression = {
                                0: 'Angry', 
                                1: 'Disgust', 
                                2: 'Fear', 
                                3: 'Happy', 
                                4: 'Sad', 
                                5: 'Surprise', 
                                6: 'Neutral'
                            }

        self.__dataset_statistics()
    
    def __dataset_statistics(self) -> None:
        """Use PrettyTable package to print dataset info.
        
        """
        # Print the expressions
        print(f'Available expressions: {self.idx_to_expression.values()}')
        
        # Print dataset cardinalities
        stats_table = PrettyTable()
        stats_table.field_names = ['Train samples', 'PublicTest samples', 'PrivateTest samples', 'Number of expressions']
        stats_table.add_row([len(self.db[self.db['Usage'] == 'Training']), len(self.db[self.db['Usage'] == 'PublicTest']), len(self.db[self.db['Usage'] == 'PrivateTest']), len(self.db.expression.unique().tolist())])
        print(stats_table)
        
        def eval_class_cardinality(self, expressions: list, train: str) -> [list, list]:
            cardinality = [len(self.db[(self.db['expression']==expr) & (self.db['Usage']==train)]) for expr in expressions]
            frac = [len(self.db[(self.db['expression']==expr) & (self.db['Usage']==train)])/len(self.db[self.db['Usage']==train)]) for expr in expressions]
            return cardinality, frac

        expressions = self.db.expression.unique()
        expressions.sort()
        
        # Print expressions statistics --> Training 
        stats_table = PrettyTable()
        stats_table.field_names = [self.idx_to_expression[label] for label in expressions]
        cardinality, frac = eval_class_cardinality(expressions=expressions, train='Training')
        stats_table.add_row(cardinality)
        stats_table.add_row(frac)

        # Print expressions statistics --> Validation 
        stats_table2 = PrettyTable()
        stats_table2.field_names = [self.idx_to_expression[label] for label in expressions]
        cardinality, frac = eval_class_cardinality(expressions=expressions, train='PublicTest')
        stats_table2.add_row(cardinality)
        stats_table2.add_row(frac)

        # Print expressions statistics --> Test 
        stats_table3 = PrettyTable()
        stats_table3.field_names = [self.idx_to_expression[label] for label in expressions]
        cardinality, frac = eval_class_cardinality(expressions=expressions, train='PrivateTest')
        stats_table3.add_row(cardinality)
        stats_table3.add_row(frac)
        
        if not self.test: 
            print(stats_table)
            print(stats_table2)
        print(stats_table3)

    def set_usage(self, usage: str) -> None:
        """Select only a specific type of images."""
        self.db = self.db[self.db['Usage'] == usage]
        print(f'Selected {usage} usage.')

    def __getitem__(self, idx: int) -> [torch.Tensor, int]:
        """Load data. 
        
        Parameters:
        -----------
        idx : int
            Index of the sample to load

        Rerturn
        -------
        image : torch.Tensor
            The loaded tensor
        expression : int
            The label correspoding to the face expression

        """
        # Convert string of pixles to list of integers and then wrap it with a torch.Tensor
        img = np.asarray(list(map(int, self.db.iloc[0].pixels.split(' '))))
        
        # Reshape the list of pixels to a 48x48 image
        img = img.reshape(48, 48)
        
        # Convert the image to a PIL.Image object
        img = Image.fromarray(img/255.)
        
        if self.transforms is not None:
            img = self.transforms(img)
        
        return img, self.db.iloc[idx].expression


class ExpW(BaseDataset):
    """Class to handle ExpW data.
    The ExpW dataset is used ONLY to TEST the models
    
    Each line in the lable.lst file has the folling structure:
    image_name; face_id_in_image; face_box_top; face_box_left; face_box_right; face_box_bottom; face_box_cofidence; expression_label;

    """
    def __init__(self, dataset_folder: str, output_folder: str, transforms: torchvision.transforms = None):
        """Init the dataset class.

        Parameters
        ----------
        dataset_folder : str
            Path to folder cotaining the dataset
        output_folder : str
            Path to output folder where to save the database 
        transforms : torchvision.transforms
            Data augmentation transforms

        """
        super(ExpW, self).__init__(dataset_folder=dataset_folder, output_folder=output_folder, transforms=transforms)

        self.idx_to_emotion={
                        0: 'Angry',
                        1: 'Disgust',
                        2: 'Fear',
                        3: 'Happy',
                        4: 'Sad',
                        5: 'Surprise',
                        6: 'Neutral'
                    }

        # Create the database if it does not exists otherwise load it from .csv file
        db_fname = join(self.output_folder, 'expw_database.csv')
        if not exists(db_fname):
            if not exists(self.output_folder):
                os.makedirs(self.output_folder)
            self.logger.info('Databse not found... creating it')
            self.db = self.__create_database(db_fname=db_fname)
        else:
            self.db = pd.read_csv(db_fname)
        
        # Print data statistics
        self.__dataset_statistics()
        

    def __create_database(self, db_fname: str) -> pd.DataFrame:
        """Create the database and save it in a .csv file.
        
        Parameters
        ----------
        db_fname : str
            Path to output database file

        Returns
        ------
        dataframe : pd.DataFrame
            The database containing dataset infos

        """
        # Read the annotations for each image
        f = open(join(self.dataset_folder, "label/label.lst"), "r")
        annotations = f.readlines()
        f.close()

        # Read info from each annotation
        def get_infos(idx: int, dtype: np.dtype) -> np.array:
            return np.asarray([ant.split(' ')[idx] for ant in annotations], dtype=dtype)
        
        # Create the database
        df = pd.DataFrame(
                    data=dict(
                            image_path=np.asarray([join(self.dataset_folder, 'image_unzipped/origin_001', ant.split(' ')[0]) for ant in annotations]), 
                            face_id_in_image=get_infos(idx=1, dtype=np.int), 
                            face_box_top=get_infos(idx=2, dtype=np.int), 
                            face_box_left=get_infos(idx=3, dtype=np.int),
                            face_box_right=get_infos(idx=4, dtype=np.int), 
                            face_box_bottom=get_infos(idx=5, dtype=np.int), 
                            face_box_cofidence=get_infos(idx=6, dtype=np.float), 
                            expression_label=get_infos(idx=7, dtype=np.int)
                        ),
                    index=None
                )

        df.to_csv(db_fname)
        self.logger.info(f'Databse saved at: {db_fname}')

        return df
    
    def __dataset_statistics(self) -> None:
        """Use PrettyTable package to print dataset info.
        
        """
        # Print the expressions
        print(f'Available expression: {self.idx_to_emotion.values()}')
        
        # Print dataset cardinalities
        stats_table = PrettyTable()
        stats_table.field_names = ['Images', 'Number of expressions']
        stats_table.add_row([len(self.db), len(self.db.expression_label.unique())])
        print(stats_table)
        
        # Print expression statistics
        stats_table = PrettyTable()
        expressions = self.db.expression_label.unique()
        expressions.sort()
        stats_table.field_names = [self.idx_to_emotion[label] for label in expressions]
        stats_table.add_row([len(self.db[self.db['expression_label']==expr]) for expr in expressions])
        print(stats_table)

    def __getitem__(self, idx: int) -> [torch.Tensor, int]:
        """Load data. 
        
        Parameters:
        -----------
        idx : int
            Index of the sample to load

        Rerturn
        -------
        image : torch.Tensor
            The loaded tensor
        expression : int
            The label correspoding to the face expression

        """
        img = self.loader(self.db.iloc[idx].image_path)
        
        if self.transforms is not None:
            img = self.transforms(img)
        
        emotion = self.db.iloc[idx].expression_label
        
        return img, emotion


class RAFdb(BaseDataset):
    """Class to handle RAFdb data.
    The RAFdb dataset is used ONLY to TEST the models
    
    """
    def __init__(self, dataset_folder: str, output_folder: str, transforms: torchvision.transforms = None):
        """Init the dataset class.

        Parameters
        ----------
        dataset_folder : str
            Path to folder cotaining the dataset
        output_folder : str
            Path to output folder where to save the database 
        transforms : torchvision.transforms
            Data augmentation transforms
        
        """
        super(RAFdb, self).__init__(dataset_folder=dataset_folder, output_folder=output_folder, transforms=transforms)

        self.idx_to_emotion = {
                            1: 'Surprise',
                            2: 'Fear',
                            3: 'Disgust',
                            4: 'Happiness',
                            5: 'Sadness',
                            6: 'Anger',
                            7: 'Neutral'
                        }

        # Create the database if it does not exists otherwise load it from .csv file
        db_fname = join(self.output_folder, 'rafdb_database.csv')
        if not exists(db_fname):
            if not exists(self.output_folder):
                os.makedirs(self.output_folder)
            self.logger.info('Databse not found... creating it')
            self.db = self.__create_database(db_fname=db_fname)
        else:
            self.db = pd.read_csv(db_fname)
        
        # Print data statistics
        self.__dataset_statistics()

    def __create_database(self, db_fname: str) -> pd.DataFrame:
        """Create the database and save it in a .csv file.
        
        Parameters
        ----------
        db_fname : str
            Path to output database file

        Returns
        ------
        dataframe : pd.DataFrame
            The database containing dataset infos

        """
        # Read the emotion files containing the list of images with the corresponding emotion label
        f = open(join(self.dataset_folder, 'EmoLabel/list_patition_label.txt'), 'r')
        lines = list(map(str.rstrip, f.readlines()))
        f.close()

        # Create the database and save it to a .csv file
        df = pd.DataFrame(
                    data=dict(
                        image_path=list(map(lambda x : join(self.dataset_folder, 'Image/aligned', x.split(' ')[0]), lines)),
                        emotion_label=list(map(lambda x : x.split(' ')[1], lines))
                    ),
                    index=None     
                )
        
        df.to_csv(db_fname)
        self.logger.info(f'Databse saved at: {db_fname}')

        return df
    
    def __dataset_statistics(self) -> None:
        """Use PrettyTable package to print dataset info.
        
        """
        # Print the expressions
        print(f'Available expression: {self.idx_to_emotion.values()}')
        
        # Print dataset cardinalities
        stats_table = PrettyTable()
        stats_table.field_names = ['Images', 'Number of expressions']
        stats_table.add_row([len(self.db), len(self.db.emotion_label.unique())])
        print(stats_table)
        
        # Print expression statistics
        stats_table = PrettyTable()
        expressions = self.db.emotion_label.unique()
        expressions.sort()
        stats_table.field_names = [self.idx_to_emotion[label] for label in expressions]
        stats_table.add_row([len(self.db[self.db['emotion_label']==expr]) for expr in expressions])
        print(stats_table)

    def __getitem__(self, idx: int) -> [torch.Tensor, int]:
        """Load data. 
        
        Parameters:
        -----------
        idx : int
            Index of the sample to load

        Rerturn
        -------
        image : torch.Tensor
            The loaded tensor
        expression : int
            The label correspoding to the face expression

        """
        img = self.loader(self.db.iloc[idx].image_path)

        if self.transforms is not None:
            img = self.transforms(img)
        
        emotion = self.db.iloc[idx].emotion_label
        
        return img, emotion


class OuluCasia(BaseDataset):
    """Class to handle OuluCasia data.
    The OuluCasia dataset is used ONLY to TEST the models
    
    """
    def __init__(self, dataset_folder: str, output_folder: str, transforms: torchvision.transforms):
        """Init the dataset class.

        Parameters
        ----------
        dataset_folder : str
            Path to folder cotaining the dataset
        output_folder : str
            Path to output folder where to save the database 
        transforms : torchvision.transforms
            Data augmentation transforms
        
        """
        super(OuluCasia, self).__init__(dataset_folder=dataset_folder, output_folder=output_folder, transforms=transforms)

        self.idx_to_emotion = {
                            
                        }
        
        # Create the database if it does not exists otherwise load it from .csv file
        db_fname = join(self.output_folder, 'oulucasia_database.csv')
        if not exists(db_fname):
            if not exists(self.output_folder):
                os.makedirs(self.output_folder)
            self.logger.info('Databse not found... creating it')
            self.db = self.__create_database(db_fname=db_fname)
        else:
            self.db = pd.read_csv(db_fname)
        
        # Print data statistics
        self.__dataset_statistics()

    def __create_database(self, db_fname: str) -> pd.DataFrame:
        """Create the database and save it in a .csv file.
        
        Parameters
        ----------
        db_fname : str
            Path to output database file

        Returns
        ------
        dataframe : pd.DataFrame
            The database containing dataset infos

        """
        
        
        df.to_csv(db_fname)
        self.logger.info(f'Databse saved at: {db_fname}')

        return df
    
    def __dataset_statistics(self) -> None:
        """Use PrettyTable package to print dataset info.
        
        """
        # Print the expressions
        print(f'Available expression: {self.idx_to_emotion.values()}')
        
        # Print dataset cardinalities
        stats_table = PrettyTable()
        stats_table.field_names = ['Images', 'Number of expressions']
        stats_table.add_row([len(self.db), len(self.db.emotion_label.unique())])
        print(stats_table)
        
        # Print expression statistics
        stats_table = PrettyTable()
        expressions = self.db.emotion_label.unique()
        expressions.sort()
        stats_table.field_names = [self.idx_to_emotion[label] for label in expressions]
        stats_table.add_row([len(self.db[self.db['emotion_label']==expr]) for expr in expressions])
        print(stats_table)

    def __getitem__(self, idx: int) -> [torch.Tensor, int]:
        """Load data. 
        
        Parameters:
        -----------
        idx : int
            Index of the sample to load

        Rerturn
        -------
        image : torch.Tensor
            The loaded tensor
        expression : int
            The label correspoding to the face expression

        """
        img = self.loader(self.db.iloc[idx].image_path)

        if self.transforms is not None:
            img = self.transforms(img)
        
        emotion = self.db.iloc[idx].emotion_label
        
        return img, emotion
