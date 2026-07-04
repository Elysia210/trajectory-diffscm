import numpy as np
import random
import torch

from diff_scm.datasets.load_brats import BrainDataset
from diff_scm.datasets.load_mnist import MNIST_dataset
from diff_scm.datasets.trajectory_dataset import TrajectoryDataset


def seed_worker(worker_id):
    np.random.seed(worker_id)
    random.seed(0)

g = torch.Generator()
g.manual_seed(0)


def get_data_loader(dataset, config, split_set, generator = True):
    if dataset == "mnist":
        loader = get_data_loader_mnist(config.data.path, config.sampling.batch_size, 
                                    split_set=split_set, which_label=config.classifier.label)
    elif dataset == "brats":
        loader = get_data_loader_brats(config.data.path, config.sampling.batch_size, split_set=split_set,
                                            sequence_translation = config.data.sequence_translation)
    elif dataset == "trajectory":
        loader = get_data_loader_trajectory(config, split_set=split_set)
    else:
        raise Exception("Dataset does exit")
    
    return get_generator_from_loader(loader) if generator else loader

def get_data_loader_mnist(path, batch_size, split_set: str = 'train', which_label: str = "class"):
    assert split_set in ["train", "val", "test"]
    default_kwargs = {"shuffle": True, "num_workers": 1, "drop_last": True, "batch_size": batch_size}
    dataset = MNIST_dataset(root_dir=path, train=split_set != "test")

    if split_set != "test":
        val_ratio = 0.1
        split = torch.utils.data.random_split(dataset,
                                              [int(len(dataset) * (1 - val_ratio)), int(len(dataset) * val_ratio)],
                                              generator=torch.Generator().manual_seed(42))
        dataset = split[0] if split_set == "train" else split[1]

    return torch.utils.data.DataLoader(dataset, **default_kwargs)


def get_data_loader_brats(path, batch_size, split_set: str = 'train',
                             sequence_translation : bool = False, 
                             healthy_data_percentage : float = 1.0):

    assert split_set in ["train", "val", "test"]
    default_kwargs = {"drop_last": True, "batch_size": batch_size, "pin_memory" : True, "num_workers": 8,
                    "prefetch_factor" : 8, "worker_init_fn" : seed_worker, "generator": g,}

    if split_set == "test":
        default_kwargs["shuffle"] = False
        default_kwargs["num_workers"] = 1
        dataset = BrainDataset(path, n_tumour_patients = None,
                               n_healthy_patients = 0, split = split_set, 
                               sequence_translation = sequence_translation,
                               )
        return torch.utils.data.DataLoader(dataset, **default_kwargs)
    else:
        default_kwargs["shuffle"] = True
        default_kwargs["num_workers"] = 8
        dataset_healthy = BrainDataset(path, split = split_set,
                n_tumour_patients=0, n_healthy_patients=None,
                skip_healthy_s_in_tumour=True,skip_tumour_s_in_healthy=True,
                )
        dataset_unhealthy = BrainDataset(path, split = split_set,
                n_tumour_patients=None, n_healthy_patients=0,
                skip_healthy_s_in_tumour=True,skip_tumour_s_in_healthy=True,
                )
        if healthy_data_percentage is not None:
            healthy_size = int(len(dataset_healthy)*healthy_data_percentage)
            unhealthy_size = len(dataset_unhealthy)
            total_size = healthy_size + unhealthy_size
            samples_weight = torch.cat([torch.ones(healthy_size)   * total_size / healthy_size,
                                        torch.ones(unhealthy_size) * total_size / unhealthy_size]
                                        ).double()
            sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight))
            default_kwargs["sampler"] = sampler
            default_kwargs.pop('shuffle', None) # shuffle and sampler are mutually exclusive

            dataset = torch.utils.data.dataset.ConcatDataset([torch.utils.data.Subset(dataset_healthy, range(0, int(len(dataset_healthy)*healthy_data_percentage))),
                                 dataset_unhealthy])
        else:
            dataset = dataset_healthy
        
    print(f"dataset lenght: {len(dataset)}")
    return torch.utils.data.DataLoader(dataset, **default_kwargs)


def get_data_loader_trajectory(config, split_set: str = "train"):
    assert split_set in ["train", "val", "test"]

    dataset = TrajectoryDataset(
        data_path=config.data.path,
        expected_timesteps=config.data.expected_timesteps,
        require_labels=config.data.require_labels,
        recursive=config.data.recursive,
        cache_in_memory=config.data.cache_in_memory,
        label_candidates=config.data.label_candidates,
        label_map_path=config.data.label_map_path,
    )

    train_ratio = float(config.data.train_ratio)
    val_ratio = float(config.data.val_ratio)
    test_ratio = float(config.data.test_ratio)
    ratio_sum = train_ratio + val_ratio + test_ratio
    if ratio_sum <= 0:
        raise ValueError("Trajectory split ratios must sum to a positive number.")

    normalized_ratios = [train_ratio / ratio_sum, val_ratio / ratio_sum, test_ratio / ratio_sum]
    total_size = len(dataset)
    train_size = int(total_size * normalized_ratios[0])
    val_size = int(total_size * normalized_ratios[1])
    test_size = total_size - train_size - val_size
    if total_size > 0 and train_size == 0:
        train_size = max(1, total_size - val_size - test_size)
    if train_size + val_size + test_size != total_size:
        test_size = total_size - train_size - val_size

    subsets = torch.utils.data.random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(config.seed),
    )
    subset_map = {"train": subsets[0], "val": subsets[1], "test": subsets[2]}
    selected_dataset = subset_map[split_set]

    default_kwargs = {
        "batch_size": config.classifier.training.batch_size,
        "drop_last": split_set == "train",
        "num_workers": config.data.num_workers,
        "pin_memory": True,
        "worker_init_fn": seed_worker,
        "generator": g,
    }
    default_kwargs["shuffle"] = split_set == "train"

    return torch.utils.data.DataLoader(selected_dataset, **default_kwargs)


def get_generator_from_loader(loader):
    while True:
        yield from loader
