import shutil
import os

import torch


def find_val(tensor, value):
    """Finds the index of the first element equal to a given value in a 1D PyTorch tensor.
       -- gen my Colab AI

    Args:
      tensor: The 1D PyTorch tensor.
      value: The value to search for.

    Returns:
      The index of the first element equal to the given value, or -1 if the value
      is not found.
    """
    indices = (tensor == value).nonzero(as_tuple=True)[0]
    if len(indices) > 0:
        return indices[0].item()
    else:
        return -1


def delete_directory_contents(directory_path):
    # written by colab AI
    """Deletes the contents of a directory.

    Args:
      directory_path: The path to the directory.
    """
    try:
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))
    except FileNotFoundError:
        print(f"Directory not found: {directory_path}")


class Saver:
    def __init__(self):
        self.save_ctr = 0
        self.basedir = 'temp'
        delete_directory_contents(self.basedir)

    def save_tensors(self, tensors, filename):
        """Saves multiple PyTorch tensors to a single file.

        Args:
            tensors: A dictionary of PyTorch tensors.
            filename: The name of the file to save to.
        """
        file = os.path.join(self.basedir, filename +
                            str(self.save_ctr) + '.pt')
        torch.save(tensors, file)
        self.save_ctr += 1


def main():
    saver = Saver()
    a = torch.randn(3, 4)
    b = torch.randint(0, 10, (2, 5))
    c = torch.ones(1, 2)
    tensors_to_save = {
        'a': a,
        'b': b,
        'c': c,
    }
    outfile = 'out'
    saver.save_tensors(tensors_to_save, outfile)
    saver.save_tensors(tensors_to_save, outfile)


if __name__ == "__main__":
    main()
