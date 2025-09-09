import copy
from turtle import shape
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import ConcatDataset, Dataset, TensorDataset
import torch
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# specify available data-sets.
AVAILABLE_DATASETS = {
    'mnist': datasets.MNIST,
    'usps' : datasets.USPS, 
}

# specify available transforms.
AVAILABLE_TRANSFORMS = {
    'mnist': [
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ],
    'usps' : [
        transforms.Resize((224, 224)),  # Resize to match ViT input
        transforms.Grayscale(num_output_channels=3),  # Ensure 3 channels
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

    ],
}

# specify configurations of available data-sets.
DATASET_CONFIGS = {
    'mnist': {'size': 224, 'channels': 3, 'classes': 10},
    'usps' : {'size': 224, 'channels': 3, 'classes': 10},
}

class SubDataset(Dataset):
    '''To sub-sample a dataset, taking only those samples with label in [sub_labels].

    After this selection of samples has been made, it is possible to transform the target-labels,
    which can be useful when doing continual learning with fixed number of output units.'''

    def __init__(self, original_dataset, sub_labels, target_transform=None):
        super().__init__()
        self.dataset = original_dataset
        self.sub_indeces = []
        for index in range(len(self.dataset)):
            if hasattr(original_dataset, "train_labels"):
                if self.dataset.target_transform is None:
                    label = self.dataset.train_labels[index]
                else:
                    label = self.dataset.target_transform(self.dataset.train_labels[index])
            elif hasattr(self.dataset, "test_labels"):
                if self.dataset.target_transform is None:
                    label = self.dataset.test_labels[index]
                else:
                    label = self.dataset.target_transform(self.dataset.test_labels[index])
            else:
                label = self.dataset[index][1]
            if label in sub_labels:
                self.sub_indeces.append(index)    ## only add the index from the input dataset that its label in sub_labels
        self.target_transform = target_transform

    def __len__(self):
        return len(self.sub_indeces)

    def __getitem__(self, index):
        sample = self.dataset[self.sub_indeces[index]]
        target = torch.tensor(sample[1])
        sample = (sample[0], target)
        if self.target_transform:
            target = self.target_transform(sample[1])
            sample = (sample[0], target)
        return sample


def get_dataset(name, type='train', download=True, capacity=None, permutation=None, dir='./datasets',
                verbose=False, target_transform=None):
    '''Create [train|valid|test]-dataset.'''

    data_name = 'mnist' if name=='mnist' else 'usps'
    dataset_class = AVAILABLE_DATASETS[data_name]

    # specify image-transformations to be applied
    dataset_transform = transforms.Compose([
        *AVAILABLE_TRANSFORMS[name],
    ])

    # load data-set
    dataset = dataset_class('{dir}/{name}'.format(dir=dir, name=data_name), train=False if type=='test' else True,
                            download=download, transform=dataset_transform, target_transform=target_transform)

    # print information about dataset on the screen
    if verbose:
        print("  --> {}: '{}'-dataset consisting of {} samples".format(name, type, len(dataset)))

    # if dataset is (possibly) not large enough, create copies until it is.
    if capacity is not None and len(dataset) < capacity:
        dataset = ConcatDataset([copy.deepcopy(dataset) for _ in range(int(np.ceil(capacity / len(dataset))))])

    return dataset

def get_multitask_experiment(dataset_name, scenario, tasks, data_dir="/newEra/furqon/data", verbose=False, exception=False, permutation=None):
    '''Load, organize and return train- and test-dataset for requested experiment.

    [exception]:    <bool>; if True, for visualization no permutation is applied to first task (permMNIST) or digits
                            are not shuffled before being distributed over the tasks (splitMNIST)'''
    transform = None
    if dataset_name == 'MNIST':
        # check for number of tasks
        if tasks>10:
            raise ValueError("Experiment 'splitMNIST' cannot have more than 10 tasks!")
        # configurations
        config = DATASET_CONFIGS['mnist']
        classes_per_task = int(np.floor(10 / tasks))

        # prepare permutation to shuffle label-ids (to create different class batches for each random seed)
        if permutation is None:
            permutation = np.array(list(range(10))) if exception else np.random.permutation(list(range(10)))
        target_transform = transforms.Lambda(lambda y, p=permutation: int(p[y]))
        # prepare train and test datasets with all classes
        mnist_train = get_dataset('mnist', type="train", dir=data_dir, target_transform=target_transform,
                                  verbose=verbose)
        mnist_test = get_dataset('mnist', type="test", dir=data_dir, target_transform=target_transform,
                                 verbose=verbose)      

        # generate labels-per-task
        labels_per_task = [
            list(np.array(range(classes_per_task)) + classes_per_task * task_id) for task_id in range(tasks)
        ]

        print(f'labels_per_task:{labels_per_task}')
        # split them up into sub-tasks
        train_datasets = []
        test_datasets = []
        for labels in labels_per_task:
            target_transform = transforms.Lambda(
                lambda y, x=labels[0]: y - x
            ) if scenario=='domain' else None
            train_datasets.append(SubDataset(mnist_train, labels, target_transform=target_transform))
            test_datasets.append(SubDataset(mnist_test, labels, target_transform=target_transform))

    elif dataset_name == 'USPS':
        # check for number of tasks
        if tasks>10:
            raise ValueError("Experiment 'splitMNIST' cannot have more than 10 tasks!")
        # configurations
        config = DATASET_CONFIGS['usps']
        classes_per_task = int(np.floor(10 / tasks))
        
        # prepare permutation to shuffle label-ids (to create different class batches for each random seed)
        if permutation is None:
            permutation = np.array(list(range(10))) if exception else np.random.permutation(list(range(10)))
        target_transform = transforms.Lambda(lambda y, p=permutation: int(p[y]))
        # prepare train and test datasets with all classes
        usps_train = get_dataset('usps', type="train", dir=data_dir, target_transform=target_transform,
                                  verbose=verbose)
        usps_test = get_dataset('usps', type="test", dir=data_dir, target_transform=target_transform,
                                 verbose=verbose)      

        # generate labels-per-task
        labels_per_task = [
            list(np.array(range(classes_per_task)) + classes_per_task * task_id) for task_id in range(tasks)
        ]

        print(f'labels_per_task:{labels_per_task}')
        # split them up into sub-tasks
        train_datasets = []
        test_datasets = []
        for labels in labels_per_task:
            target_transform = transforms.Lambda(
                lambda y, x=labels[0]: y - x
            ) if scenario=='domain' else None
            train_datasets.append(SubDataset(usps_train, labels, target_transform=target_transform))
            test_datasets.append(SubDataset(usps_test, labels, target_transform=target_transform))

    else:
        raise RuntimeError('Given undefined experiment: {}'.format(name))

    # If needed, update number of (total) classes in the config-dictionary
    config['classes'] = classes_per_task if scenario=='domain' else classes_per_task*tasks

    # Return tuple of train-, validation- and test-dataset, config-dictionary and number of classes per task
    return ((train_datasets, test_datasets), config, classes_per_task, transform, permutation)


class get_subset_office31(object):
    def __init__(self, name_dataset, scenario):
        # root dir (local pc or colab)
        self.root_dir = "/newEra/furqon/data/office/%s/images" % name_dataset

        __datasets__ = ["amazon", "dslr", "webcam"]

        if name_dataset not in __datasets__:
            raise ValueError("must introduce one of the three datasets in office")

        self.data_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        self.scenario = scenario
        self.task_info = {}
        self.get_tasks() # split to train and test subset

    def get_tasks(self):
        self.task_info[1] = {'n_elem': 6,
                            'classes_list': [0, 1, 2, 3, 4, 5]}
        self.task_info[2] = {'n_elem': 6,
                            'classes_list': [6, 7, 8, 9, 10, 11]}
        self.task_info[3] = {'n_elem': 6,
                            'classes_list': [12, 13, 14, 15, 16, 17]}
        self.task_info[4] = {'n_elem': 6,
                            'classes_list': [18, 19, 20, 21, 22, 23]}
        self.task_info[5] = {'n_elem':6,
                            'classes_list': [24, 25, 26, 27, 28, 29, 30]}
        
        # generate labels-per-task
        for i in range(5):
            if i == 0 :
                labels_per_task = [self.task_info[i+1]['classes_list']]
            else:
                labels_per_task = labels_per_task + [self.task_info[i+1]['classes_list']]
                
        # split them up into sub-tasks
        train_datasets = []
        test_datasets = []
        for labels in labels_per_task:
            target_transform = transforms.Lambda(
                lambda y, x=labels[0]: y - x
            ) if self.scenario=='domain' else None
            train_datasets.append(self.subset(labels, target_transform=target_transform)[0])
            test_datasets.append(self.subset(labels, target_transform=target_transform)[1])
        self.train_datasets = train_datasets
        self.test_datasets = test_datasets

    def subset(self, labels, target_transform, split=0.9):
        dataset = datasets.ImageFolder(root=self.root_dir,
                                    transform=self.data_transforms, 
                                    target_transform=transforms.Lambda(lambda y : torch.tensor(np.array(y).astype(np.int64))))
        targets = np.asarray(dataset.targets)
        for i, class_i in enumerate(labels):
            idx = (targets==class_i).nonzero()[0]
            if i==0:
                idxs = idx
            else:
                idxs = np.concatenate((idxs, idx),axis=0)
        
        train_idx = []
        test_idx = []
        for cls in np.unique(np.asarray(dataset.targets)[idxs]):
            idx = (dataset.targets==cls).nonzero()[0]
            idx_0 = np.random.choice(idx,int(split*len(idx)),replace=False).tolist()
            [train_idx.append(x) for x in idx_0]
            [test_idx.append(x) for x in idx if x not in set(idx_0)]

        train_Set = copy.deepcopy(dataset)
        train_Set.samples = np.asarray(dataset.samples)[train_idx]
        train_Set.targets = np.asarray(dataset.targets)[train_idx]
        train_Set.index = train_idx
        if target_transform is not None:
            train_Set.targets = train_Set.targets - labels[0]

        test_set = copy.deepcopy(dataset)
        test_set.samples = np.asarray(dataset.samples)[test_idx]
        test_set.targets = np.asarray(dataset.targets)[test_idx]
        test_set.index = test_idx
        if target_transform is not None:
            test_set.targets = test_set.targets - labels[0]

        return train_Set, test_set

#----------------------------------------------------------------------------------------------------------#

class get_subset_officeHome(object):
    def __init__(self, name_dataset, scenario):
        # root dir (local pc or colab)
        self.root_dir = "/newEra/furqon/data/OfficeHome/%s" % name_dataset

        __datasets__ = ["Art", "Clipart", "Product", "Real World"]

        if name_dataset not in __datasets__:
            raise ValueError("must introduce one of the three datasets in office")

        self.data_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        self.scenario = scenario
        self.task_info = {}
        self.get_tasks()
        
    def get_tasks(self):
        self.task_info[1] = {'n_elem': 5,
                                'classes_list': [0, 1, 2, 3, 4]}
        self.task_info[2] = {'n_elem': 5,
                                'classes_list': [5, 6, 7, 8, 9]}
        self.task_info[3] = {'n_elem': 5,
                                'classes_list': [10, 11, 12, 13, 14]}
        self.task_info[4] = {'n_elem': 5,
                                'classes_list': [15, 16, 17, 18, 19]}
        self.task_info[5] = {'n_elem': 5,
                                'classes_list': [20, 21, 22, 23, 24]}
        self.task_info[6] = {'n_elem': 5,
                                'classes_list': [25, 26, 27, 28, 29]}
        self.task_info[7] = {'n_elem': 5,
                                'classes_list': [30, 31, 32, 33, 34]}
        self.task_info[8] = {'n_elem': 5,
                                'classes_list': [35, 36, 37, 38, 39]}
        self.task_info[9] = {'n_elem': 5,
                                'classes_list': [40, 41, 42, 43, 44]}
        self.task_info[10] = {'n_elem': 5,
                                'classes_list': [45, 46, 47, 48, 49]}
        self.task_info[11] = {'n_elem': 5,
                                'classes_list': [50, 51, 52, 53, 54]}
        self.task_info[12] = {'n_elem': 5,
                                'classes_list': [55, 56, 57, 58, 59]}
        self.task_info[13] = {'n_elem': 5,
                                'classes_list': [60, 61, 62, 63, 64]}
        
        # generate labels-per-task
        for i in range(13):
            if i == 0 :
                labels_per_task = [self.task_info[i+1]['classes_list']]
            else:
                labels_per_task = labels_per_task + [self.task_info[i+1]['classes_list']]
                
        # split them up into sub-tasks
        train_datasets = []
        test_datasets = []
        for labels in labels_per_task:
            target_transform = transforms.Lambda(
                lambda y, x=labels[0]: y - x
            ) if self.scenario=='domain' else None
            train_datasets.append(self.subset(labels, target_transform=target_transform)[0])
            test_datasets.append(self.subset(labels, target_transform=target_transform)[1])
        self.train_datasets = train_datasets
        self.test_datasets = test_datasets
    
    def subset(self, labels, target_transform, split=0.9):
        dataset = datasets.ImageFolder(root=self.root_dir,
                                    transform=self.data_transforms, 
                                    target_transform=transforms.Lambda(lambda y : torch.tensor(np.array(y).astype(np.int64))))
        targets = np.asarray(dataset.targets)
        for i, class_i in enumerate(labels):
            idx = (targets==class_i).nonzero()[0]
            if i==0:
                idxs = idx
            else:
                idxs = np.concatenate((idxs, idx),axis=0)
        
        train_idx = []
        test_idx = []
        for cls in np.unique(np.asarray(dataset.targets)[idxs]):
            idx = (dataset.targets==cls).nonzero()[0]
            idx_0 = np.random.choice(idx,int(split*len(idx)),replace=False).tolist()
            [train_idx.append(x) for x in idx_0]
            [test_idx.append(x) for x in idx if x not in set(idx_0)]

        train_Set = copy.deepcopy(dataset)
        train_Set.samples = np.asarray(dataset.samples)[train_idx]
        train_Set.targets = np.asarray(dataset.targets)[train_idx]
        train_Set.index = train_idx
        if target_transform is not None:
            train_Set.targets = train_Set.targets - labels[0]

        test_set = copy.deepcopy(dataset)
        test_set.samples = np.asarray(dataset.samples)[test_idx]
        test_set.targets = np.asarray(dataset.targets)[test_idx]
        test_set.index = test_idx
        if target_transform is not None:
            test_set.targets = test_set.targets - labels[0]

        return train_Set, test_set

#----------------------------------------------------------------------------------------------------------#

class get_subset_visda(object):
    def __init__(self, name_dataset, scenario):
        self.root_dir = f"/newEra/furqon/data/visda/{name_dataset}"
        self.data_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.scenario = scenario
        self.get_tasks()

    def get_tasks(self):
        self.task_info = {
            1: {'n_elem': 3, 'classes_list': [0, 1, 2]},
            2: {'n_elem': 3, 'classes_list': [3, 4, 5]},
            3: {'n_elem': 3, 'classes_list': [6, 7, 8]},
            4: {'n_elem': 3, 'classes_list': [9, 10, 11]},
        }
        labels_per_task = [self.task_info[i + 1]['classes_list'] for i in range(4)]

        print(f'labels_per_task:{labels_per_task}')

        train_datasets, test_datasets = [], []
        for labels in labels_per_task:
            target_transform = transforms.Lambda(
                lambda y, x=labels[0]: y - x
            ) if self.scenario == 'domain' else None
            train_datasets.append(self.subset(labels, target_transform=target_transform)[0])
            test_datasets.append(self.subset(labels, target_transform=target_transform)[1])
        self.train_datasets = train_datasets
        self.test_datasets = test_datasets

    def subset(self, labels, target_transform, split=0.9):
        dataset = datasets.ImageFolder(root=self.root_dir,
            transform=self.data_transforms,
            target_transform=transforms.Lambda(lambda y: torch.tensor(np.array(y).astype(np.int64))))
        
        targets = np.asarray(dataset.targets)
        idxs = np.concatenate([np.where(targets == c)[0] for c in labels])
        train_idx, test_idx = [], []
        for c in labels:
            idx = np.where(targets == c)[0]
            np.random.shuffle(idx)
            n_train = int(split * len(idx))
            train_idx.extend(idx[:n_train])
            test_idx.extend(idx[n_train:])

        train_set = copy.deepcopy(dataset)
        train_set.samples = np.array(dataset.samples)[train_idx]
        train_set.targets = np.array(dataset.targets)[train_idx]
        train_set.index = train_idx
        if target_transform: train_set.targets = train_set.targets - labels[0]

        test_set = copy.deepcopy(dataset)
        test_set.samples = np.array(dataset.samples)[test_idx]
        test_set.targets = np.array(dataset.targets)[test_idx]
        test_set.index = test_idx
        if target_transform: test_set.targets = test_set.targets - labels[0]

        return train_set, test_set

#----------------------------------------------------------------------------------------------------------#

class get_subset_domainnet(object):
    def __init__(self, name_dataset, scenario):
        self.root_dir = f"/newEra/furqon/DomainNet/{name_dataset}"
        self.data_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.scenario = scenario
        self.total_classes = 345
        self.classes_per_task = 23
        self.num_tasks = self.total_classes // self.classes_per_task
        self.get_tasks()

    def get_tasks(self):
        labels_per_task = [
            list(range(i * self.classes_per_task, (i + 1) * self.classes_per_task))
            for i in range(self.num_tasks)
        ]

        print(f'labels_per_task:{labels_per_task}')

        train_datasets, test_datasets = [], []
        for labels in labels_per_task:
            target_transform = transforms.Lambda(
                lambda y, x=labels[0]: y - x
            ) if self.scenario == 'domain' else None
            train_datasets.append(self.subset(labels, target_transform=target_transform)[0])
            test_datasets.append(self.subset(labels, target_transform=target_transform)[1])
        self.train_datasets = train_datasets
        self.test_datasets = test_datasets

    def subset(self, labels, target_transform, split=0.9):
        dataset = datasets.ImageFolder(root=self.root_dir,
            transform=self.data_transforms,
            target_transform=transforms.Lambda(lambda y: torch.tensor(np.array(y).astype(np.int64)))
        )

        targets = np.asarray(dataset.targets)
        idxs = np.concatenate([np.where(targets == c)[0] for c in labels])

        train_idx, test_idx = [], []
        for c in labels:
            idx = np.where(targets == c)[0]
            np.random.shuffle(idx)
            n_train = int(split * len(idx))
            train_idx.extend(idx[:n_train])
            test_idx.extend(idx[n_train:])

        train_set = copy.deepcopy(dataset)
        train_set.samples = np.array(dataset.samples)[train_idx]
        train_set.targets = np.array(dataset.targets)[train_idx]
        train_set.index = train_idx
        if target_transform:
            train_set.targets = train_set.targets - labels[0]

        test_set = copy.deepcopy(dataset)
        test_set.samples = np.array(dataset.samples)[test_idx]
        test_set.targets = np.array(dataset.targets)[test_idx]
        test_set.index = test_idx
        if target_transform:
            test_set.targets = test_set.targets - labels[0]

        return train_set, test_set
