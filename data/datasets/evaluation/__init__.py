from ModelCore.data import datasets
from .tct import tct_evaluation

def evaluate(dataset, predictions, output_folder, **kwargs):

    args = dict(dataset=dataset, prediction=predictions, output_folder=output_folder, **kwargs)
    if isinstance(dataset, datasets.tctDataset):
        return tct_evaluation(**args)
    else:
        dataset_name = dataset.__class__.__name__
        raise NotImplementedError("Unsupported dataset type {}".format(dataset_name))