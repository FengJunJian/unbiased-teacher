# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import torchvision.transforms as transforms
from ubteacher.data.transforms.augmentation_impl import (
    GaussianBlur,
)

#import cv2

def build_strong_augmentation(cfg, is_train):
    """
    Create a list of :class:`Augmentation` from config.
    Now it includes resizing and flipping.

    Returns:
        list[Augmentation]
    """

    logger = logging.getLogger(__name__)
    augmentation = []
    p=0.1#0.1
    if is_train:
        # This is simialr to SimCLR https://arxiv.org/abs/2002.05709
        original_flag=True
        if original_flag:
            augmentation.append(
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=p)#0.8
            )
        augmentation.append(transforms.RandomGrayscale(p=0.1))#0.2
        augmentation.append(transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=p))#0.5
        #transforms.AutoAugment()
        if original_flag:
            randcrop_transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    #original
                    transforms.RandomErasing(
                        p=p-0.03, scale=(0.05, 0.2), ratio=(0.3, 3.3), value="random"# 0.7 0.07
                    ),
                    transforms.RandomErasing(
                        p=p-0.05, scale=(0.02, 0.2), ratio=(0.1, 6), value="random" #0.5 0.05
                    ),
                    transforms.RandomErasing(
                        p=p-0.07, scale=(0.02, 0.2), ratio=(0.05, 8), value="random" #0.3 0.03
                    ),
                    #fixed
                    # transforms.RandomErasing(
                    #     p=0.7, scale=(0.002, 0.005), ratio=(0.3, 3.3), value="random"# H: max scale-0.0083, W:max scale-0.0057
                    #     # H: max scale-0.0083, W:max scale-0.0057
                    # ),
                    # transforms.RandomErasing(
                    #     p=0.5, scale=(0.001, 0.005), ratio=(0.1, 6), value="random"
                    # ),
                    # transforms.RandomErasing(
                    #     p=0.3, scale=(0.001, 0.005), ratio=(0.05, 8), value="random"
                    # ),
                    transforms.ToPILImage(),
                ]
            )
            augmentation.append(randcrop_transform)

        logger.info("Augmentations used in training: " + str(augmentation))
    return transforms.Compose(augmentation)