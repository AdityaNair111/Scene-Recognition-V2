import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from sklearn.preprocessing import StandardScaler
import os.path as osp
import utils
import torch


def create_datasets(data_path, input_size, rgb=False):
  """
  Args:
  - data_path: (string) Path to the directory that contains the 'test' and
      'train' data directories.
  - input_size: (w, h) Size of input image. The images will be resized to
      this size.
  - rgb: (boolean) Flag indicating if input images are RGB or grayscale. If
      False, images will be converted to grayscale.

  Returns:
  - train_dataloader: Dataloader for the training dataset.
  - test_dataloader: Dataloader for the testing/validation dataset.
  """
  train_data_path = osp.join(data_path, 'train')
  test_data_path = osp.join(data_path, 'test')
  train_mean, train_std = utils.get_mean_std(train_data_path, input_size, rgb)
  test_mean, test_std = utils.get_mean_std(test_data_path, input_size, rgb)


  """ TRAIN DATA TRANSFORMS """
  train_data_tforms = []
  train_data_tforms.append(transforms.Resize(size=max(input_size)))
  train_data_tforms.append(transforms.CenterCrop(size=input_size))
  if not rgb:
    train_data_tforms.append(transforms.Grayscale())
  train_data_tforms.append(transforms.RandomHorizontalFlip(p=0.5))
  train_data_tforms.append(transforms.ToTensor())
  train_data_tforms.append(transforms.Normalize(train_mean, train_std))
  train_data_tforms = transforms.Compose(train_data_tforms)

  """ TEST/VALIDATION DATA TRANSFORMS """

  test_data_tforms = []
  test_data_tforms.append(transforms.Resize(size=max(input_size)))
  test_data_tforms.append(transforms.CenterCrop(size=input_size))
  if not rgb:
    test_data_tforms.append(transforms.Grayscale())
  test_data_tforms.append(transforms.ToTensor())
  test_data_tforms.append(transforms.Normalize(test_mean, test_std))

  test_data_tforms = transforms.Compose(test_data_tforms)


  """ DATASET LOADERS """
  # Creating dataset loaders using the tranformations specified above.
  train_dset = datasets.ImageFolder(root=osp.join(data_path, 'train'),
                                    transform=train_data_tforms)
  test_dset = datasets.ImageFolder(root=osp.join(data_path, 'test'),
                                   transform=test_data_tforms)
  return train_dset, test_dset


class SimpleNet(nn.Module):
  def __init__(self, num_classes, droprate=0.5, rgb=False, verbose=False):
    """
    Args:
    - num_classes: (int) Number of output classes.
    - droprate: (float) Droprate of the network (used for droppout).
    - rgb: (boolean) Flag indicating if input images are RGB or grayscale, used
      to set the number of input channels.
    - verbose: (boolean) If True a hook is registered to print the size of input
      to classifier everytime the forward function is called.
    """
    super(SimpleNet, self).__init__() # initialize the parent class, a must
    in_channels = 3 if rgb else 1

    """ NETWORK SETUP """
    self.features = nn.Sequential(
      nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=9,
        stride=1, padding=0, bias=False),
      nn.BatchNorm2d(num_features=32),
      nn.MaxPool2d(kernel_size=7, stride=7, padding=0), 
      nn.ReLU(),
      nn.Dropout(p=0.5,inplace=False),
      nn.Conv2d(in_channels=32, out_channels=128, kernel_size=6,
        stride=1, padding=0, bias=False),
      nn.BatchNorm2d(num_features=128),
      nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
      nn.ReLU(),
      nn.Dropout(p=0.5,inplace=False)
    )

    self.classifier = nn.Linear(in_features=128, out_features=num_classes,bias=True)


    """ NETWORK INITIALIZATION """
    for name, m in self.named_modules():
      if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 1)
        m.weight.data.mul_(1e-2)
        if m.bias is not None:
          nn.init.constant_(m.bias.data, 0)
      elif isinstance(m, nn.BatchNorm2d):
        
        m.weight.data.uniform_()
        if m.bias is not None:
          nn.init.constant_(m.bias.data, 0)
        
    if verbose:
      # Hook that prints the size of input to classifier everytime the forward
      # function is called.
      self.classifier.register_forward_hook(utils.print_input_size_hook)

  def forward(self, x):
    """
    Forward step of the network.

    Args:
    - x: input data.

    Returns:
    - x: output of the classifier.
    """
    x = self.features(x)
    x=torch.squeeze(x)
    x = self.classifier(x)
    return x.squeeze()

def custom_part1_trainer(model):
    # return a dict that contains your customized learning settings.
    pass
    return None
    
def create_part2_model(model, num_classes):
  """
  Take the passed in model and prepare it for finetuning by following the
  instructions.

  Args:
  - model: The model you need to prepare for finetuning. For the purposes of
    this project, you will pass in AlexNet.
  - num_classes: number of classes the model should output.

  Returns:
  - model: The model ready to be fine tuned.
  """
  # Getting all layers from the input model's classifier.
  new_classifier = list(model.classifier.children())
  new_classifier = new_classifier[:-1]
  obj=nn.Linear(in_features=4096,out_features=num_classes,bias=True)
  obj.weight.data.normal_(0,1)
  obj.weight.data.mul_(1e-2)
  if obj.bias is not None:
  	nn.init.constant_(obj.bias.data, 0)
  new_classifier.append(obj)
  # Connecting all layers to form a new classifier.
  model.classifier = nn.Sequential(*new_classifier)

  return model
  
def custom_part2_trainer(model):
    # return a dict that contains your customized learning settings.
    pass
    return None
