#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.engine.defaults import DefaultPredictor
from ubteacher import add_ubteacher_config
from ubteacher.engine.trainer import UBTeacherTrainer, BaselineTrainer
from detectron2.utils.visualizer import Visualizer
from detectron2.data import build_detection_test_loader
from detectron2.data import MetadataCatalog
# hacky way to register
from ubteacher.modeling.meta_arch.rcnn import TwoStagePseudoLabGeneralizedRCNN
from ubteacher.modeling.proposal_generator.rpn import PseudoLabRPN
from ubteacher.modeling.roi_heads.roi_heads import StandardROIHeadsPseudoLab
from ubteacher.data.datasets import builtin
from predictor import PredictorCustom
import os
import matplotlib.pyplot as plt
import json
from sklearn.manifold import TSNE
from collections import Counter
import numpy as np
import pickle
import argparse
import sys
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()#from yacs.config import CfgNode
    add_ubteacher_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    #cfg.freeze()
    default_setup(cfg, args)
    return cfg

def t_sne_projection(x,y=None,dims=2):
    #sns.set(color_codes=True)
    #sns.set(rc={'figure.figsize': (11.7, 8.27)})
    #palette = sns.color_palette("bright", 80)
    tsne = TSNE(n_components=dims)
    x_embedded=tsne.fit_transform(x,y)  # 进行数据降维,降成两维

    return x_embedded,tsne#y
def drawDiffClassST(x,y,colors):#SS:o,SMD:^
    marker = ['o', '^']
    #s = 10 #* np.ones(x[indices].shape[0])
    s=None
    for i in sorted(set(y)):
        indices = np.where(
            np.isin(np.array(y), i)
        )[0]
        #m, c = divmod(i, 2)
        plt.scatter(x[indices, 0], x[indices, 1],s=s, c=colors[i], #marker=marker[m],#alpha=0.5,
                    edgecolors='k', )  # sns.color_palette(palettes[0]) alpha=0.5

    plt.xticks([])
    plt.yticks([])

def ncolors(num_color):
    #import seaborn
    import colorsys
    colors = []
    if False:
        cs=list(seaborn.xkcd_rgb.values())
        inv=1#int(len(cs)/num_color)
        for i in range(num_color):
            ind=i*inv
            r = int(cs[ind][1:3], 16)
            g = int(cs[ind][3:5], 16)
            b = int(cs[ind][5:7], 16)
            colors.append((r,g,b))
    else:
        hsv_tuples = [(x / num_color, 1.0, 1.0)
                      for x in range(num_color)]
        # hsv_tuples = [(x / num_color, 1.0, 1.0)
        #               for x in range(num_color)]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        # colors = list(
        #     map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
        #         colors))
        colors = list(
            map(lambda x: (x[0], x[1], x[2]),
                colors))
        colors = [c[::-1] for c in colors]

    return colors
def main(args):
    cfg = setup(args)
    # if cfg.SEMISUPNET.Trainer == "ubteacher":
    #     Trainer = UBTeacherTrainer
    # elif cfg.SEMISUPNET.Trainer == "baseline":
    #     Trainer = BaselineTrainer
    # else:
    #     raise ValueError("Trainer Name is not found.")
    extractFlag=True
    if  extractFlag:#extract feature
        predictor = PredictorCustom(cfg)
        featuresROI=predictor.feature_extract(True)
        featuresROI = predictor.feature_extract(False)
        # outputs = predictor.draw_batch()
        # for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
        #     results=res[dataset_name]['bbox']
        #     with open(os.path.join(output_folders[idx],'AP.txt'),'w') as f:
        #         json.dump(results,f)
        #     predictions = torch.load(os.path.join(output_folders[idx], 'instances_predictions.pth'))
        #     pre=COCO(predictions)
        # saveImgPath = os.path.join(output_folder, 'img')
        # Visualizer()
        #return
    #else:#visual feature dimensionality reduction
        #np.array(((0, 0, 255), (255, 0, 0), (0, 255, 255), (255, 192, 203))) / 255.0  # 蓝，红，暗蓝,暗红
        featureFiles=['E:/SSL/unbiased-teacher/Demooutput10/test_SS_SMD/featureROIweak1.pkl' ,
                     'E:/SSL/unbiased-teacher/Demooutput10/test_SS_SMD/featureROIstrong1.pkl']#featureROIstrong.pkl  featureROIweak
        colors = ncolors(15)
        for fi,featureFile in enumerate(featureFiles):
            with open(featureFile,"rb") as f:
                featureROI=pickle.load(f)

            labels=sorted(featureROI.keys())

            features=[]
            Clabel=[]
            for la in tqdm(labels):
                num=len(featureROI[la])
                # print(len(featureROI[la]))
                features.extend(featureROI[la])
                Clabel.extend([la]*num)
            features=np.array(features).squeeze()
            features=np.reshape(features,(features.shape[0],-1))
            indices={}
            Clabel = np.array(Clabel, np.int64)
            cy = Counter(Clabel)
            print(cy)

            minV=min(cy.values())*1.1#1
            x=np.empty((0,features.shape[1]),features.dtype)
            y=np.empty((0,),np.int64)
            for i in set(Clabel):#采样平衡
                indice = np.where(
                    np.isin(np.array(Clabel), i)
                )[0]
                if len(indice)>minV:
                    indice=np.random.permutation(indice)[:int(minV)]
                #indices.update({i:indice})
                x=np.concatenate([x,features[indice]],axis=0)
                y=np.concatenate([y,Clabel[indice]],axis=0)

            #################################################total
            print('processing tsne')
            x_e,tsne=t_sne_projection(x, y=y, dims=2)
            ax=plt.figure()
            drawDiffClassST(x_e, y, colors)
            string=list(map(lambda x:str(x),labels))
            # string=
            plt.legend(string)
            if fi==0:
                plt.savefig("visual_weak1.jpg")
            else:
                plt.savefig("visual_strong1.jpg")
        #plt.show()



def argument_parser():
    """
    Create a parser with some common arguments used by detectron2 users.

    Args:
        epilog (str): epilog passed to ArgumentParser describing the usage.

    Returns:
        argparse.ArgumentParser:
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument("--predict-only", action="store_true", help="perform evaluation only")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Whether to attempt to resume from the checkpoint directory. "
             "See documentation of `DefaultTrainer.resume_or_load()` for what it means.",
    )
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus *per machine*")
    parser.add_argument("--num-machines", type=int, default=1, help="total number of machines")
    parser.add_argument(
        "--machine-rank", type=int, default=0, help="the rank of this machine (unique per machine)"
    )

    # PyTorch still may leave orphan processes in multi-gpu training.
    # Therefore we use a deterministic way to obtain port,
    # so that users are aware of orphan processes by seeing the port occupied.
    port = 2 ** 15 + 2 ** 14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    parser.add_argument(
        "--dist-url",
        default="tcp://127.0.0.1:{}".format(port),
        help="initialization URL for pytorch distributed backend. See "
             "https://pytorch.org/docs/stable/distributed.html for details.",
    )
    parser.add_argument(
        "opts",
        help="""
Modify config options at the end of the command. For Yacs configs, use
space-separated "PATH.KEY VALUE" pairs.
For python-based LazyConfig, use "path.key=value".
        """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser

if __name__ == "__main__":
    # args = default_argument_parser().parse_args()
    args=argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
