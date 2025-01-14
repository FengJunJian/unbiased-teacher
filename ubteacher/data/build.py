# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import numpy as np
import operator
import json
import torch.utils.data
from detectron2.utils.comm import get_world_size
from detectron2.data.common import (
    DatasetFromList,
    MapDataset,
)
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data.samplers import (
    InferenceSampler,
    RepeatFactorTrainingSampler,
    TrainingSampler,
)
from detectron2.data.build import (
    trivial_batch_collator,
    worker_init_reset_seed,
    get_detection_dataset_dicts,
    build_batch_data_loader,
)
from ubteacher.data.common import (
    AspectRatioGroupedSemiSupDatasetTwoCrop,
)
import os
import json
def subsample_idx(num_all):
    np.random.seed(0)
    SupPercent = [0.01, 0.1,0.5, 1.0, 2.0, 5.0, 10.0]#20.0
    run_times = 10
    dict_all = {}
    for sup_p in SupPercent:
        dict_all[sup_p] = {}
        for run_i in range(run_times):
            num_label = int(sup_p / 100. * num_all)
            labeled_idx = np.random.choice(range(num_all), size=num_label, replace=False)
            dict_all[sup_p][run_i] = labeled_idx.tolist()
    return dict_all

"""
This file contains the default logic to build a dataloader for training or testing.
"""

def divide_label_unlabel(
    dataset_dicts, SupPercent, random_data_seed, random_data_seed_path
):
    num_all = len(dataset_dicts)
    num_label = int(SupPercent / 100.0 * num_all)

    # read from pre-generated data seed
    if not os.path.exists(random_data_seed_path):
        semidict_all=subsample_idx(num_all)
        with open(random_data_seed_path,"w") as f:
            json.dump(semidict_all,f)
    with open(random_data_seed_path) as COCO_sup_file:
        coco_random_idx = json.load(COCO_sup_file)

    labeled_idx = np.array(coco_random_idx[str(SupPercent)][str(random_data_seed)])
    assert labeled_idx.shape[0] == num_label, "Number of READ_DATA is mismatched."

    label_dicts = []
    unlabel_dicts = []
    labeled_idx = set(labeled_idx)

    for i in range(len(dataset_dicts)):
        if i in labeled_idx:
            label_dicts.append(dataset_dicts[i])
        else:
            unlabel_dicts.append(dataset_dicts[i])

    return label_dicts, unlabel_dicts


# uesed by supervised-only baseline trainer
def build_detection_semisup_train_loader(cfg, mapper=None):

    dataset_dicts = get_detection_dataset_dicts(
        cfg.DATASETS.TRAIN,
        filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
        min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
        if cfg.MODEL.KEYPOINT_ON
        else 0,
        proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN
        if cfg.MODEL.LOAD_PROPOSALS
        else None,
    )

    # Divide into labeled and unlabeled sets according to supervision percentage
    label_dicts, unlabel_dicts = divide_label_unlabel(
        dataset_dicts,
        cfg.DATALOADER.SUP_PERCENT,
        cfg.DATALOADER.RANDOM_DATA_SEED,
        cfg.DATALOADER.RANDOM_DATA_SEED_PATH,
    )

    dataset = DatasetFromList(label_dicts, copy=False)

    if mapper is None:
        mapper = DatasetMapper(cfg, True)
    dataset = MapDataset(dataset, mapper)

    sampler_name = cfg.DATALOADER.SAMPLER_TRAIN
    logger = logging.getLogger(__name__)
    logger.info("Using training sampler {}".format(sampler_name))

    if sampler_name == "TrainingSampler":
        sampler = TrainingSampler(len(dataset))
    elif sampler_name == "RepeatFactorTrainingSampler":
        repeat_factors = (
            RepeatFactorTrainingSampler.repeat_factors_from_category_frequency(
                label_dicts, cfg.DATALOADER.REPEAT_THRESHOLD
            )
        )
        sampler = RepeatFactorTrainingSampler(repeat_factors)
    else:
        raise ValueError("Unknown training sampler: {}".format(sampler_name))

    # list num of labeled and unlabeled
    logger.info("Number of training samples " + str(len(dataset)))
    logger.info("Supervision percentage " + str(cfg.DATALOADER.SUP_PERCENT))

    return build_batch_data_loader(
        dataset,
        sampler,
        cfg.SOLVER.IMS_PER_BATCH,
        aspect_ratio_grouping=cfg.DATALOADER.ASPECT_RATIO_GROUPING,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
    )


# uesed by evaluation
def build_detection_test_loader(cfg, dataset_name, mapper=None):
    dataset_dicts = get_detection_dataset_dicts(
        [dataset_name],
        filter_empty=False,
        proposal_files=[
            cfg.DATASETS.PROPOSAL_FILES_TEST[
                list(cfg.DATASETS.TEST).index(dataset_name)
            ]
        ]
        if cfg.MODEL.LOAD_PROPOSALS
        else None,
    )
    dataset = DatasetFromList(dataset_dicts)
    if mapper is None:
        mapper = DatasetMapper(cfg, False)
    dataset = MapDataset(dataset, mapper)

    sampler = InferenceSampler(len(dataset))
    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, 1, drop_last=False)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        batch_sampler=batch_sampler,
        collate_fn=trivial_batch_collator,
    )
    return data_loader


# uesed by unbiased teacher trainer
def build_detection_semisup_train_loader_two_crops(cfg, mapper=None):
    if cfg.DATASETS.CROSS_DATASET:  # cross-dataset (e.g., coco-additional)
        label_dicts = get_detection_dataset_dicts(
            cfg.DATASETS.TRAIN_LABEL,
            filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
            min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
            if cfg.MODEL.KEYPOINT_ON
            else 0,
            proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN
            if cfg.MODEL.LOAD_PROPOSALS
            else None,
        )
        unlabel_dicts = get_detection_dataset_dicts(
            cfg.DATASETS.TRAIN_UNLABEL,
            filter_empty=False,
            min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
            if cfg.MODEL.KEYPOINT_ON
            else 0,
            proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN
            if cfg.MODEL.LOAD_PROPOSALS
            else None,
        )
    else:  # different degree of supervision (e.g., COCO-supervision)
        dataset_dicts = get_detection_dataset_dicts(
            cfg.DATASETS.TRAIN,
            filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
            min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
            if cfg.MODEL.KEYPOINT_ON
            else 0,
            proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN
            if cfg.MODEL.LOAD_PROPOSALS
            else None,
        )

        # Divide into labeled and unlabeled sets according to supervision percentage
        label_dicts, unlabel_dicts = divide_label_unlabel(
            dataset_dicts,
            cfg.DATALOADER.SUP_PERCENT,
            cfg.DATALOADER.RANDOM_DATA_SEED,
            cfg.DATALOADER.RANDOM_DATA_SEED_PATH,
        )


        if False:
            from detectron2.data.catalog import MetadataCatalog
            from detectron2.data.build import print_instances_class_histogram, check_metadata_consistency
            dataset_dicts_test = get_detection_dataset_dicts(
                cfg.DATASETS.TEST,
                filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
                min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
                if cfg.MODEL.KEYPOINT_ON
                else 0,
                proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN
                if cfg.MODEL.LOAD_PROPOSALS
                else None,
            )
            class_names = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes
            check_metadata_consistency("thing_classes", cfg.DATASETS.TRAIN)
            #data_count = {}
            dataA = print_instances_class_histogram(dataset_dicts, class_names)
            datal=print_instances_class_histogram(label_dicts, class_names)
            datau=print_instances_class_histogram(unlabel_dicts, class_names)
            datatest=print_instances_class_histogram(dataset_dicts_test, class_names)
            #list2dict=lambda d: dict(tuple(zip(d[::2], d[1::2])))
            extractNum=lambda d:d[1::2]
            import matplotlib.pyplot as plt
            import numpy as np
            a=np.array(extractNum(datal))
            b = np.array(extractNum(datau))
            c=np.array(extractNum(dataA))#68862
            d= np.array(extractNum(datatest))
            np.savez('dataClassDistribution.npz',labeled=a,unlabeled=b,training=c,test=d)

    label_dataset = DatasetFromList(label_dicts, copy=False)
    # exclude the labeled set from unlabeled dataset
    unlabel_dataset = DatasetFromList(unlabel_dicts, copy=False)
    # include the labeled set in unlabel dataset
    # unlabel_dataset = DatasetFromList(dataset_dicts, copy=False)

    if mapper is None:
        mapper = DatasetMapper(cfg, True)
    label_dataset = MapDataset(label_dataset, mapper)
    unlabel_dataset = MapDataset(unlabel_dataset, mapper)

    sampler_name = cfg.DATALOADER.SAMPLER_TRAIN
    logger = logging.getLogger(__name__)
    logger.info("Using training sampler {}".format(sampler_name))
    if sampler_name == "TrainingSampler":
        label_sampler = TrainingSampler(len(label_dataset))
        unlabel_sampler = TrainingSampler(len(unlabel_dataset))
    elif sampler_name == "RepeatFactorTrainingSampler":
        raise NotImplementedError("{} not yet supported.".format(sampler_name))
    else:
        raise ValueError("Unknown training sampler: {}".format(sampler_name))
    return build_semisup_batch_data_loader_two_crop(
        (label_dataset, unlabel_dataset),
        (label_sampler, unlabel_sampler),
        cfg.SOLVER.IMG_PER_BATCH_LABEL,
        cfg.SOLVER.IMG_PER_BATCH_UNLABEL,
        aspect_ratio_grouping=cfg.DATALOADER.ASPECT_RATIO_GROUPING,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
    )

def draw_class_distribution(file="dataClassDistribution.npz"):
    import matplotlib.pyplot as plt
    import numpy as np

    data=np.load(file)
    a = data['labeled']
    b = data['unlabeled']
    c = data['training']
    d = data['test']
    DataSum = [a.sum(), b.sum(), c.sum(), d.sum()]
    # a=np.array([384, 1758, 1213, 1729, 803, 1725, 258, 45233, 1291, 3642, 709, 4666, 507, 3374, 1570])  # 68862
    # b=np.array([35,158,142,188,86,148,22,4344,126,358,66,447,46,304,166])#6636
    # b =b/ b.sum()
    plt.figure(1)
    a = a / a.sum()
    b = b / b.sum()
    c = c / c.sum()
    d = d / d.sum()
    bar_width = 0.2
    x = np.arange(len(a))

    plt.bar(x, a, width=bar_width)
    # plt.figure(2)
    plt.bar(x + bar_width, b, width=bar_width)
    plt.bar(x + 2 * bar_width, c, width=bar_width)
    plt.bar(x + 3 * bar_width, d, width=bar_width)
    plt.legend(
        ['labeled data:' + str(DataSum[0]), 'unlabeled data:' + str(DataSum[1]), 'training data:' + str(DataSum[2]),
         'test data:' + str(DataSum[3])])
    plt.savefig("a.jpg", )  # bbox_inches = 'tight'
    # plt.hist(a, bins=len(a))
    # plt.hist(b)


# batch data loader
def build_semisup_batch_data_loader_two_crop(
    dataset,
    sampler,
    total_batch_size_label,
    total_batch_size_unlabel,
    *,
    aspect_ratio_grouping=False,
    num_workers=0
):
    world_size = get_world_size()
    assert (
        total_batch_size_label > 0 and total_batch_size_label % world_size == 0
    ), "Total label batch size ({}) must be divisible by the number of gpus ({}).".format(
        total_batch_size_label, world_size
    )

    assert (
        total_batch_size_unlabel > 0 and total_batch_size_unlabel % world_size == 0
    ), "Total unlabel batch size ({}) must be divisible by the number of gpus ({}).".format(
        total_batch_size_label, world_size
    )

    batch_size_label = total_batch_size_label // world_size
    batch_size_unlabel = total_batch_size_unlabel // world_size

    label_dataset, unlabel_dataset = dataset
    label_sampler, unlabel_sampler = sampler

    if aspect_ratio_grouping:
        label_data_loader = torch.utils.data.DataLoader(
            label_dataset,
            sampler=label_sampler,
            num_workers=num_workers,
            batch_sampler=None,
            collate_fn=operator.itemgetter(
                0
            ),  # don't batch, but yield individual elements
            worker_init_fn=worker_init_reset_seed,
        )  # yield individual mapped dict
        unlabel_data_loader = torch.utils.data.DataLoader(
            unlabel_dataset,
            sampler=unlabel_sampler,
            num_workers=num_workers,
            batch_sampler=None,
            collate_fn=operator.itemgetter(
                0
            ),  # don't batch, but yield individual elements
            worker_init_fn=worker_init_reset_seed,
        )  # yield individual mapped dict
        return AspectRatioGroupedSemiSupDatasetTwoCrop(
            (label_data_loader, unlabel_data_loader),
            (batch_size_label, batch_size_unlabel),
        )
    else:
        raise NotImplementedError("ASPECT_RATIO_GROUPING = False is not supported yet")