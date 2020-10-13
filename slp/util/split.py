import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from torch.utils.data import SubsetRandomSampler


def dataloaders_from_indices(dataset, train_indices, val_indices, create_dataloader):
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    train_loader = create_dataloader(dataset, train_sampler)
    val_loader = create_dataloader(dataset, val_sampler)
    return train_loader, val_loader


def train_test_split(
    dataset, create_dataloader, test_size=0.2, shuffle=True, seed=None
):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    test_split = int(np.floor(test_size * dataset_size))
    if shuffle:
        if seed is not None:
            np.random.seed(seed)
        np.random.shuffle(indices)

    train_indices = indices[test_split:]
    val_indices = indices[:test_split]
    return dataloaders_from_indices(
        dataset, train_indices, val_indices, create_dataloader
    )


def kfold_split(dataset, create_dataloader, k=5, shuffle=True, stratified=False, seed=None):
    kf_cls = StratifiedKFold if stratified else KFold
    kfold = kf_cls(n_splits=k, shuffle=shuffle, random_state=seed)
    for train_indices, val_indices in kfold.split(dataset, dataset.y):
        yield dataloaders_from_indices(
            dataset, train_indices, val_indices, create_dataloader
        )
