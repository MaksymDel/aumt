import os

import torch
from torch.autograd import Variable


def to_var(x):
    """Converts numpy to variable."""
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def to_data(x):
    """Converts variable to numpy."""
    if torch.cuda.is_available():
        x = x.cpu()
    return x.data.numpy()


def create_dir(directory):
    """Creates a directory if it does not already exist.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def trim_tensors(smaller_tensor, tensor_to_trim_1, tensor_to_trim_2):
    trimlen = smaller_tensor.size()[1]
    trimmed_1 = tensor_to_trim_1[:, :trimlen, :]
    trimmed_2 = tensor_to_trim_1[:, :trimlen, :]

    return trimmed_1, trimmed_2
