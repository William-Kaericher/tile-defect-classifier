import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import datasets, models, transforms
from PIL import Image
import os
from tqdm import tqdm
import pandas as pd
from torchvision.models import ResNet18_Weights, VGG16_Weights, DenseNet121_Weights
import difflib

class CustomImageDataset(Dataset):
    """
    Custom Dataset for loading images from a flat directory with labels provided via a CSV file.
    """
    def __init__(self, img_dir, labels_csv, transform=None):
        """
        Args:
            img_dir (str): Path to the image directory.
            labels_csv (str): Path to the CSV file with image filenames and labels.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.img_dir = img_dir
        self.labels_df = pd.read_csv(labels_csv)
        self.transform = transform
        self.classes = sorted(self.labels_df['label'].unique())
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.labels_df['label_idx'] = self.labels_df['label'].map(self.class_to_idx)

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        img_name = self.labels_df.iloc[idx]['filename']
        label = self.labels_df.iloc[idx]['label_idx']
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

class MachineVisionModel:
    SUPPORTED_MODELS = ['resnet18', 'vgg16', 'densenet121']

    def __init__(self, 
                 model_name='resnet18', 
                 num_classes=2, 
                 pretrained=True, 
                 freeze_layers=True, 
                 num_frozen_layers=None,  # New Parameter
                 data_dir='data', 
                 input_size= 224,
                 batch_size=32, 
                 learning_rate=1e-3, 
                 num_epochs=5, 
                 device=None,
                 validation_split=0.1,
                 shuffle=True,
                 random_seed=42):
        """
        Initializes the MachineVisionModel with specified parameters.

        Parameters:
        - model_name (str): Name of the pretrained model to use (e.g., 'resnet18', 'vgg16', 'densenet121').
        - num_classes (int): Number of output classes.
        - pretrained (bool): Whether to use pretrained weights.
        - freeze_layers (bool): Whether to freeze pretrained layers.
        - num_frozen_layers (int, optional): Number of initial layers (modules) to freeze. Overrides freeze_layers if set.
        - data_dir (str or None): Path to the data directory. Set to None when using custom dataloaders.
        - input_size (int): Input image size.
        - batch_size (int): Batch size for training.
        - learning_rate (float): Learning rate for the optimizer.
        - num_epochs (int): Number of training epochs.
        - device (str or torch.device): Device to run the model on.
        - validation_split (float): Proportion of training data to use for validation if 'val' folder is absent.
        - shuffle (bool): Whether to shuffle the dataset before splitting.
        - random_seed (int): Random seed for reproducibility.
        """
        self.model_name = model_name.lower()
        
        if self.model_name not in self.SUPPORTED_MODELS:
            # Find close matches
            close_matches = difflib.get_close_matches(self.model_name, self.SUPPORTED_MODELS, n=1, cutoff=0.6)
            suggestion = f" Did you mean '{close_matches[0]}'?" if close_matches else ""
            raise ValueError(f"Model '{self.model_name}' is not supported. Supported models are: {', '.join(self.SUPPORTED_MODELS)}.{suggestion}")
        
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.freeze_layers = freeze_layers
        self.num_frozen_layers = num_frozen_layers
        self.data_dir = data_dir
        self.input_size = input_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.validation_split = validation_split
        self.shuffle = shuffle
        self.random_seed = random_seed
        
        # Initialize the model with updated weights parameter
        self.model = self._initialize_model()
        self.model = self.model.to(self.device)
        
        # Define loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        # Only optimize parameters that require gradients (i.e., unfrozen layers)
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.learning_rate)
        
        # Initialize data loaders if data_dir is provided
        if self.data_dir is not None:
            self.dataloaders = self._initialize_dataloaders()
        else:
            self.dataloaders = {}
            self.dataset_sizes = {}
        
    def _get_weights(self):
        """
        Maps the model name and pretrained flag to the appropriate weights enumeration.
        
        Returns:
            weights (WeightsEnum or None): The weights to use for the model.
        """
        if self.model_name == 'resnet18':
            if self.pretrained:
                return ResNet18_Weights.DEFAULT  # Latest pretrained weights
            else:
                return None
        elif self.model_name == 'vgg16':
            if self.pretrained:
                return VGG16_Weights.DEFAULT
            else:
                return None
        elif self.model_name == 'densenet121':
            if self.pretrained:
                return DenseNet121_Weights.DEFAULT
            else:
                return None
        else:
            raise ValueError(f"Model '{self.model_name}' not supported for weight mapping.")
    
    def _initialize_model(self):
        """
        Loads a pretrained model with updated 'weights' parameter and modifies the final layer for the desired number of classes.
        
        Returns:
            model (nn.Module): The initialized model.
        """
        weights = self._get_weights()
        
        if self.model_name == 'resnet18':
            model = models.resnet18(weights=weights)
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, self.num_classes)
        elif self.model_name == 'vgg16':
            model = models.vgg16(weights=weights)
            in_features = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(in_features, self.num_classes)
        elif self.model_name == 'densenet121':
            model = models.densenet121(weights=weights)
            in_features = model.classifier.in_features
            model.classifier = nn.Linear(in_features, self.num_classes)
        else:
            raise ValueError(f"Model '{self.model_name}' not supported.")
        
        # Freezing layers based on num_frozen_layers or freeze_layers
        if self.num_frozen_layers is not None:
            self._freeze_specific_layers(model)
        elif self.freeze_layers:
            self._freeze_all_but_classification(model)
        
        return model
    
    def _freeze_specific_layers(self, model):
        """
        Freezes the first 'n' convolutional layers of the model based on the specified number of frozen layers.
        
        Parameters:
            model (nn.Module): The model whose layers are to be frozen.
        """
        conv_layers = [module for module in model.modules() if isinstance(module, nn.Conv2d)]
        
        if not conv_layers:
            print(f"No Conv2d layers found in {self.model_name}.")
            return
        
        num_layers_to_freeze = min(self.num_frozen_layers, len(conv_layers))
        
        for i in range(num_layers_to_freeze):
            for param in conv_layers[i].parameters():
                param.requires_grad = False
            print(f"Conv layer {i+1} frozen: {conv_layers[i]}")
        
        # Ensure the final classification layer is trainable
        if self.model_name == 'resnet18':
            for param in model.fc.parameters():
                param.requires_grad = True
        elif self.model_name == 'vgg16':
            for param in model.classifier[6].parameters():
                param.requires_grad = True
        elif self.model_name == 'densenet121':
            for param in model.classifier.parameters():
                param.requires_grad = True
    
    def _freeze_all_but_classification(self, model):
        """
        Freezes all layers/modules of the model except the final classification layer.
        
        Parameters:
            model (nn.Module): The model whose layers are to be frozen.
        """
        if self.model_name == 'resnet18':
            for name, param in model.named_parameters():
                if 'fc' not in name:
                    param.requires_grad = False
            print("All layers except 'fc' have been frozen.")
        elif self.model_name == 'vgg16':
            for name, param in model.named_parameters():
                if 'classifier.6' not in name:
                    param.requires_grad = False
            print("All layers except 'classifier.6' have been frozen.")
        elif self.model_name == 'densenet121':
            for name, param in model.named_parameters():
                if 'classifier' not in name:
                    param.requires_grad = False
            print("All layers except 'classifier' have been frozen.")
    
    def _initialize_dataloaders(self):
        """
        Sets up data transformations and DataLoaders for training, validation, and testing.
        Detects if 'val' folder exists. If not, splits a portion of 'train' data for validation.
        """
        # Define transforms
        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize((self.input_size, self.input_size)),  # Ensure fixed size
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], 
                                     [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize((self.input_size, self.input_size)),  # Ensure fixed size
                transforms.CenterCrop(self.input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], 
                                     [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize((self.input_size, self.input_size)),  # Ensure fixed size
                transforms.CenterCrop(self.input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], 
                                     [0.229, 0.224, 0.225])
            ]),
        }
        
        # Check if 'val' directory exists
        val_dir = os.path.join(self.data_dir, 'val')
        has_val = os.path.isdir(val_dir)
        
        if has_val:
            print("Validation directory found. Using separate 'train', 'val', and 'test' splits.")
            image_datasets = {x: datasets.ImageFolder(os.path.join(self.data_dir, x),
                                                      data_transforms[x])
                              for x in ['train', 'val', 'test']}
        else:
            print("Validation directory not found. Splitting 'train' into 'train' and 'val'.")
            train_dir = os.path.join(self.data_dir, 'train')
            train_dataset = datasets.ImageFolder(train_dir, data_transforms['train'])
            
            # Calculate sizes
            total_train = len(train_dataset)
            val_size = int(self.validation_split * total_train)
            actual_train_size = total_train - val_size
            
            # Split the dataset
            generator = torch.Generator().manual_seed(self.random_seed)
            train_subset, val_subset = random_split(train_dataset, [actual_train_size, val_size], generator=generator)
            
            # Create datasets dictionary
            image_datasets = {
                'train': train_subset,
                'val': val_subset,
                'test': datasets.ImageFolder(os.path.join(self.data_dir, 'test'), data_transforms['test'])
            }
        
        # Create DataLoaders
        dataloaders = {
            'train': DataLoader(image_datasets['train'], batch_size=self.batch_size,
                                shuffle=True, num_workers=4),
            'val': DataLoader(image_datasets['val'], batch_size=self.batch_size,
                              shuffle=False, num_workers=4),
            'test': DataLoader(image_datasets['test'], batch_size=self.batch_size,
                               shuffle=False, num_workers=4)
        }
        
        # Dataset sizes and class names
        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
        if isinstance(image_datasets['train'], torch.utils.data.Subset):
            # Extract class names from the original dataset
            self.class_names = image_datasets['train'].dataset.classes
        else:
            self.class_names = image_datasets['train'].classes
        
        self.dataset_sizes = dataset_sizes
        
        return dataloaders
    
    def initialize_custom_dataloaders(self, img_dir, labels_csv, test_size=0.0, val_size=None):
        """
        Initializes data loaders for a custom dataset where images are in a flat directory
        and labels are provided via a CSV file.

        Parameters:
        - img_dir (str): Path to the image directory.
        - labels_csv (str): Path to the CSV file containing image filenames and labels.
        - test_size (float): Proportion of the dataset to include in the test split. Set to 0.0 if a separate test set exists.
        - val_size (float or None): Proportion of the dataset to include in the validation split. If None, uses class's validation_split.
        """

        # Define transforms
        train_transform = transforms.Compose([
            transforms.Resize((self.input_size, self.input_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                                [0.229, 0.224, 0.225])
        ])
        
        val_test_transform = transforms.Compose([
            transforms.Resize((self.input_size, self.input_size)),
            transforms.CenterCrop(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                                [0.229, 0.224, 0.225])
        ])
        
        # Create the full dataset
        full_dataset = CustomImageDataset(img_dir=img_dir, labels_csv=labels_csv, transform=train_transform)
        
        # Initialize variables
        test_loader = None
        test_dataset = None
        
        # Handle test split if test_size > 0 and no separate test set
        if test_size > 0:
            train_val_size = 1 - test_size
            train_size = int(train_val_size * len(full_dataset))
            test_size_actual = len(full_dataset) - train_size
            generator = torch.Generator().manual_seed(self.random_seed)
            train_val_subset, test_subset = random_split(full_dataset, [train_size, test_size_actual], generator=generator)
            
            # Further split train_val_subset into train and val
            if val_size is None:
                val_size = self.validation_split
            actual_val_size = int(val_size * len(full_dataset))
            actual_train_size = len(train_val_subset) - actual_val_size
            train_subset, val_subset = random_split(train_val_subset, [actual_train_size, actual_val_size], generator=generator)
            
            # Create DataLoaders
            self.dataloaders['train'] = DataLoader(train_subset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=4)
            self.dataloaders['val'] = DataLoader(val_subset, batch_size=self.batch_size, shuffle=False, num_workers=4)
            self.dataloaders['test'] = DataLoader(test_subset, batch_size=self.batch_size, shuffle=False, num_workers=4)
            self.dataset_sizes = {
                'train': len(train_subset),
                'val': len(val_subset),
                'test': len(test_subset)
            }
        else:
            # No test split, assume separate test set exists
            # Split into train and val
            if val_size is None:
                val_size = self.validation_split
            actual_val_size = int(val_size * len(full_dataset))
            actual_train_size = len(full_dataset) - actual_val_size
            generator = torch.Generator().manual_seed(self.random_seed)
            train_subset, val_subset = random_split(full_dataset, [actual_train_size, actual_val_size], generator=generator)
            
            # Create DataLoaders
            self.dataloaders['train'] = DataLoader(train_subset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=4)
            self.dataloaders['val'] = DataLoader(val_subset, batch_size=self.batch_size, shuffle=False, num_workers=4)
            self.dataloaders['test'] = None
            self.dataset_sizes = {
                'train': len(train_subset),
                'val': len(val_subset),
                'test': 0
            }
        # After creating DataLoaders
        if test_size > 0:
            self.class_names = full_dataset.classes
        else:
            self.class_names = full_dataset.classes
        
        print("Custom DataLoaders for training and validation have been initialized.")

        
    def initialize_test_loader(self, test_image_dir, test_labels_csv):
        """
        Initializes the test DataLoader for a separate test set.
    
        Parameters:
        - test_image_dir (str): Path to the test images directory.
        - test_labels_csv (str): Path to the test labels CSV file.
        """
    
        # Define the transformation for the test set
        test_transform = transforms.Compose([
            transforms.Resize((self.input_size, self.input_size)),
            transforms.CenterCrop(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                                 [0.229, 0.224, 0.225])
        ])
    
        # Create the test dataset
        test_dataset = CustomImageDataset(
            img_dir=test_image_dir,
            labels_csv=test_labels_csv,
            transform=test_transform
        )
    
        # Create the test DataLoader
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,        # No need to shuffle test data
            num_workers=4         # Adjust based on your system
        )
    
        # Assign the test DataLoader to the MachineVisionModel
        self.dataloaders['test'] = test_loader
        self.dataset_sizes['test'] = len(test_dataset)
    
        # Set class_names if not already set
        if not hasattr(self, 'class_names') or not self.class_names:
            self.class_names = test_dataset.classes
            print("class_names has been set from the test dataset.")
        else:
            print("class_names already exists and will not be overwritten.")
    
        print("Test DataLoader has been initialized.")


    
    def train_model(self):
        """
        Trains the model and evaluates on validation set after each epoch.
        """
        best_model_wts = self.model.state_dict()
        best_acc = 0.0
        
        for epoch in range(self.num_epochs):
            print(f'Epoch {epoch+1}/{self.num_epochs}')
            print('-' * 10)
            
            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()   # Set model to evaluate mode
                
                running_loss = 0.0
                running_corrects = 0
                
                # Iterate over data
                for inputs, labels in tqdm(self.dataloaders[phase], desc=f'{phase}'):
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    
                    # Zero the parameter gradients
                    self.optimizer.zero_grad()
                    
                    # Forward
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = self.criterion(outputs, labels)
                        
                        # Backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()
                    
                    # Statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                
                epoch_loss = running_loss / self.dataset_sizes[phase]
                epoch_acc = running_corrects.double() / self.dataset_sizes[phase]
                
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
                
                # Deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = self.model.state_dict()
            
            print()
        
        print(f'Best val Acc: {best_acc:.4f}')
        
        # Load best model weights
        self.model.load_state_dict(best_model_wts)
        torch.save(self.model.state_dict(), 'best_model.pth')
        print("Training complete. Best model saved as 'best_model.pth'.")
    
    def evaluate_model(self, phase='test'):
        """
        Evaluates the model on the specified dataset (default is test set).
        """
        if phase not in self.dataloaders or self.dataloaders[phase] is None:
            raise ValueError(f"Phase '{phase}' not found in dataloaders or dataloader is not initialized.")
        
        self.model.eval()
        running_corrects = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(self.dataloaders[phase], desc=f'Evaluating {phase}'):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                
                running_corrects += torch.sum(preds == labels.data)
        
        acc = running_corrects.double() / self.dataset_sizes[phase]
        print(f'{phase} Accuracy: {acc:.4f}')
        return acc
    
    def predict(self, image_path):
        """
        Makes a prediction on a single image.
        
        Parameters:
        - image_path (str): Path to the image file.
        
        Returns:
        - predicted_class (str): Predicted class label.
        - confidence (float): Confidence score of the prediction.
        """
        # Define the same transforms as validation
        transform = transforms.Compose([
            transforms.Resize((self.input_size, self.input_size)),  # Ensure fixed size
            transforms.CenterCrop(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                                 [0.229, 0.224, 0.225])
        ])
        
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0)  # Add batch dimension
        image = image.to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(image)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, preds = torch.max(probabilities, 1)
        
        predicted_class = self.class_names[preds]
        return predicted_class, confidence.item()
    
    def save_model(self, save_path='final_model.pth'):
        """
        Saves the trained model to the specified path.
        """
        torch.save(self.model.state_dict(), save_path)
        print(f'Model saved to {save_path}')
    
    def load_model(self, load_path='best_model.pth'):
        """
        Loads a saved model from the specified path.
        """
        self.model.load_state_dict(torch.load(load_path, map_location=self.device))
        self.model.to(self.device)
        print(f'Model loaded from {load_path}')
