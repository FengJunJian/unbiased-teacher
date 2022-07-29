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
from sklearn.decomposition import PCA
from collections import Counter
import numpy as np
import pickle
import argparse
import sys
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

CLASSES_NAMES=[
'passenger ship',
'ore carrier',
'general cargo ship',
'fishing boat',
'sail boat',
'kayak',
'flying bird',
'vessel',
'buoy',
'ferry',
'container ship',
'other',
'boat',
'speed boat',
'bulk cargo carrier',
]

Nratio = 1

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
def pca_projection(x,y=None,dims=2):
    pca = PCA(n_components=dims)
    x_embedded=pca.fit_transform(x,y)  # 进行数据降维,降成两维
    print(pca.explained_variance_ratio_)
    print(pca.explained_variance_)
    return x_embedded,pca#y

def drawDiffClassST(ax,x,y,colors,marker='o',s=50):#SS:o,SMD:^
    #marker = ['o', '^']
    #s = 10 #* np.ones(x[indices].shape[0])

    #s=[s]*len(x)
    if x.shape[1]==3:
        for i in sorted(set(y)):
            indices = np.where(
                np.isin(np.array(y), i)
            )[0]
            #m, c = divmod(i, 2)
            ax.scatter(x[indices, 0], x[indices, 1],x[indices, 2],s=s, c=colors[i], marker=marker,#alpha=0.5,
                        edgecolors='k', )  # sns.color_palette(palettes[0]) alpha=0.5
            ax.set_zticks([])
    else:
        for i in sorted(set(y)):
            indices = np.where(
                np.isin(np.array(y), i)
            )[0]
            #m, c = divmod(i, 2)
            ax.scatter(x[indices, 0], x[indices, 1],s=s, c=colors[i], marker=marker,#alpha=0.5,
                        edgecolors='k', )  # sns.color_palette(palettes[0]) alpha=0.5

    ax.set_xticks([])
    ax.set_yticks([])


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
def reduction_feature(featureFiles):
    # featureFiles = ['Demooutput10/test_SS_SMD/featureROIweak1.pkl',
    #                 'Demooutput10/test_SS_SMD/featureROIstrong1.pkl']  # featureROIstrong.pkl  featureROIweak
    colors = ncolors(15)
    for fi, featureFile in enumerate(featureFiles):
        with open(featureFile, "rb") as f:
            featureROI = pickle.load(f)

        labels = sorted(featureROI.keys())

        features = []
        Clabel = []
        for la in tqdm(labels):
            if la == 6:
                continue
            num = len(featureROI[la])
            # print(len(featureROI[la]))
            features.extend(featureROI[la])
            Clabel.extend([la] * num)
        features = np.array(features).squeeze()
        features = np.reshape(features, (features.shape[0], -1))
        indices = {}
        Clabel = np.array(Clabel, np.int64)
        labels = set(Clabel)
        cy = Counter(Clabel)
        print('before:', cy)

        minV = min(cy.values()) * Nratio  # 1.1
        x = np.empty((0, features.shape[1]), features.dtype)
        y = np.empty((0,), np.int64)
        for i in set(Clabel):  # 采样平衡
            indice = np.where(
                np.isin(np.array(Clabel), i)
            )[0]
            if len(indice) > minV:
                indice = np.random.permutation(indice)[:int(minV)]
            # indices.update({i:indice})
            x = np.concatenate([x, features[indice]], axis=0)
            y = np.concatenate([y, Clabel[indice]], axis=0)
        cy = Counter(y)
        print('after:', cy)
        #################################################total
        print('processing tsne')
        x_e, tsne = t_sne_projection(x, y=y, dims=2)
        # if fi==0:
        #     x_e,pca=pca_projection(x,y=y,dims=2)
        ax = plt.figure()
        if fi == 0:
            # drawDiffClassST(x_e[:featureNum], y[:featureNum], colors, marker='o', s=60)  # marker = ['o', '^']
            # # plt.legend(string)
            # drawDiffClassST(x_e[featureNum:], y[featureNum:], colors, marker='^', s=40)
            drawDiffClassST(x_e, y, colors, marker='o')
        elif fi == 1:
            drawDiffClassST(x_e, y, colors, marker='^')
        string = list(map(lambda x: CLASSES_NAMES[x], labels))
        # string=
        # plt.legend(string)
        folder=os.path.dirname(featureFile)
        basename=os.path.splitext(os.path.basename(featureFile))[0]
        #if 'weak' in featureFile:
        plt.savefig(os.path.join(folder,f"visual_{basename}_{Nratio}.jpg"))
        # else:
        #     plt.savefig(os.path.join(folder, f"visual_strong1_{Nratio}.jpg"))
            #plt.savefig(f"Demooutput10/test_SS_SMD/visual_strong1_{Nratio}.jpg")
def reduction_feature_intra_class(featureFiles,flag_3D=False,force_reload=False):
    # featureFiles = ['Demooutput10/test_SS_SMD/featureROIweak1.pkl',
    #                 'Demooutput10/test_SS_SMD/featureROIstrong1.pkl']  # featureROIstrong.pkl  featureROIweak
    reload=False
    numid=os.path.splitext(featureFiles[0])[0][-1]
    #for featureFile in featureFiles:
    #Nratio = 1
    basename = f"featureReduce_weak_strong_{numid}_{Nratio}"
    if flag_3D:
        basename=basename+'_3D'
    folder=os.path.dirname(featureFiles[0])
    if not os.path.exists(os.path.join(folder,basename+'.npz')):
        reload = True
    colors = ncolors(15)

    # f"Demooutput10/test_SS_SMD/featureReduce{Nratio}.npz"
    if not reload and not force_reload:
        data = np.load(os.path.join(folder,basename+'.npz'))  # x=x_e, y=y, N=featureNum
        x_e = data['x']
        y = data['y']
        featureNum = data['N']
        # x_e0=data['x0']
        # x_e1=data['x1']

    # np.savez("Demooutput10/test_SS_SMD/featureReduce1", x=x_e, y=y, N=featureNum)
    else:
        featureCols = [None] * len(featureFiles)
        ClabelCols = [None] * len(featureFiles)
        for fi, featureFile in enumerate(featureFiles):
            with open(featureFile, "rb") as f:
                featureROI = pickle.load(f)
            features = []
            Clabel = []
            labels = sorted(featureROI.keys())
            for la in tqdm(labels):
                if la == 6:
                    continue
                num = len(featureROI[la])
                # print(len(featureROI[la]))
                features.extend(featureROI[la])
                Clabel.extend([la] * num)

            features = np.array(features).squeeze()
            features = np.reshape(features, (features.shape[0], -1))
            Clabel = np.array(Clabel, np.int64)
            featureCols[fi] = features
            ClabelCols[fi] = Clabel

        assert len(featureCols[0]) == len(featureCols[1]) and len(ClabelCols[0]) == len(ClabelCols[1])

        # features = np.array(features).squeeze()
        # features = np.reshape(features, (features.shape[0], -1))
        # indices = {}

        features = featureCols[0]
        Clabel = ClabelCols[0]
        # for i,Clabel in enumerate(ClabelCols):
        labels = set(Clabel)
        cy = Counter(Clabel)
        print('before:', cy)
        minV = min(cy.values()) * Nratio  # 1.1

        x = np.empty((0, features.shape[1]), features.dtype)
        y = np.empty((0,), np.int64)
        indices = []
        for i in set(Clabel):  # 采样平衡
            indice = np.where(
                np.isin(np.array(Clabel), i)
            )[0]
            if len(indice) > minV:
                indice = np.random.permutation(indice)[:int(minV)]
            indices.append(indice)
            # indices.update({i:indice})
            x = np.concatenate([x, features[indice]], axis=0)
            y = np.concatenate([y, Clabel[indice]], axis=0)
        featureNum = x.shape[0]
        print('Number of features:', featureNum)
        cy = Counter(y)
        print('after0:', cy)
        features = featureCols[1]
        Clabel = ClabelCols[1]
        # for indice in indices:
        #     x = np.concatenate([x, features[indice]], axis=0)
        #     y = np.concatenate([y, Clabel[indice]], axis=0)
        for i in set(Clabel):
            indice = np.where(
                np.isin(np.array(Clabel), i)
            )[0]

            if len(indice) > minV:
                indice = np.random.permutation(indice)[:int(minV)]
            #indices.append(indice)
            # indices.update({i:indice})
            x = np.concatenate([x, features[indice]], axis=0)
            y = np.concatenate([y, Clabel[indice]], axis=0)

        cy = Counter(y)
        print('after1:', cy)
        #################################################total
        print('processing tsne')
        # if flag_3D:
        #     x_e, tsne = t_sne_projection(x, y=y, dims=3)
        #     #np.savez(os.path.join(folder, basename + '3D.npz'), x=x_e, y=y, N=featureNum)
        # else:
        # x_e0, _ = t_sne_projection(x[:featureNum], y=y[:featureNum], dims=2)
        # x_e1, _ = t_sne_projection(x[featureNum:], y=y[featureNum:], dims=2)
        x_e, tsne = t_sne_projection(x, y=y, dims=2)
        np.savez(os.path.join(folder, basename + '.npz'), x=x_e, y=y, N=featureNum)
        #np.savez(os.path.join(folder,basename+'.npz'), x=x_e, y=y, N=featureNum,x0=x_e0,x1=x_e1)
    # if fi==0:
    #     x_e,pca=pca_projection(x,y=y,dims=2)
    fig = plt.figure()
    ax = fig.add_subplot(131)
    drawDiffClassST(ax, x_e[:featureNum], y[:featureNum], colors, marker='o', s=70)  # marker = ['o', '^']
    ax = fig.add_subplot(132)
    drawDiffClassST(ax, x_e[:featureNum], y[:featureNum], colors, marker='^', s=70)  # marker = ['o', '^']

    ax = fig.add_subplot(133)
    # if flag_3D:
    #     ax = fig.gca(projection="3d")
    # else:
    #     ax = fig.gca()
    drawDiffClassST(ax,x_e[:featureNum], y[:featureNum], colors, marker='o', s=70)  # marker = ['o', '^']

    # ax.scatter(x_e[:, 0], x_e[:, 1], x_e[:, 2],
    #             edgecolors='k', )  # sns.color_palette(palettes[0]) alpha=0.5

    #ax.legend(string,loc=2,  borderaxespad=0., ncol=5)
    if False:#legend
        labels = set(y)
        string = list(map(lambda x: CLASSES_NAMES[x], labels))
        tt = ax.legend(string, loc=2, bbox_to_anchor=(1.2, -0.2), borderaxespad=0., ncol=7)
        fig = tt.figure
        fig.canvas.draw()
        bbox = tt.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig('legend1', dpi="figure", bbox_inches=bbox)

    drawDiffClassST(ax,x_e[featureNum:], y[featureNum:], colors, marker='^', s=40)
    #ax.legend(string+string, loc=2, borderaxespad=0., ncol=5)
    plt.savefig(os.path.join(folder,f"visual_{basename}.jpg"))
def reduction_feature_cross_class(featureFiles,flag_3D=False,force_reload=False):
    # featureFiles = ['Demooutput10/test_SS_SMD/featureROIweak1.pkl',
    #                 'Demooutput10/test_SS_SMD/featureROIstrong1.pkl']  # featureROIstrong.pkl  featureROIweak
    reload=False
    numid=os.path.splitext(featureFiles[0])[0][-1]
    #for featureFile in featureFiles:
    #Nratio = 1
    basename = f"featureReduce_weak_strong_PCA{numid}_{Nratio}"
    if flag_3D:
        basename=basename+'_3D'
    folder=os.path.dirname(featureFiles[0])
    if not os.path.exists(os.path.join(folder,basename+'.npz')):
        reload = True
    colors = ncolors(15)

    # f"Demooutput10/test_SS_SMD/featureReduce{Nratio}.npz"
    if not reload and not force_reload:
        data = np.load(os.path.join(folder,basename+'.npz'))  # x=x_e, y=y, N=featureNum
        x_e = data['x']
        y = data['y']
        featureNum = data['N']
    # np.savez("Demooutput10/test_SS_SMD/featureReduce1", x=x_e, y=y, N=featureNum)
    else:
        featureCols = [None] * len(featureFiles)
        ClabelCols = [None] * len(featureFiles)
        for fi, featureFile in enumerate(featureFiles):
            with open(featureFile, "rb") as f:
                featureROI = pickle.load(f)
            features = []
            Clabel = []
            labels = sorted(featureROI.keys())
            for la in tqdm(labels):
                if la == 6:
                    continue
                num = len(featureROI[la])
                # print(len(featureROI[la]))
                features.extend(featureROI[la])
                Clabel.extend([la] * num)

            features = np.array(features).squeeze()
            features = np.reshape(features, (features.shape[0], -1))
            Clabel = np.array(Clabel, np.int64)
            featureCols[fi] = features
            ClabelCols[fi] = Clabel

        assert len(featureCols[0]) == len(featureCols[1]) and len(ClabelCols[0]) == len(ClabelCols[1])

        # features = np.array(features).squeeze()
        # features = np.reshape(features, (features.shape[0], -1))
        # indices = {}

        features = featureCols[0]
        Clabel = ClabelCols[0]
        # for i,Clabel in enumerate(ClabelCols):
        labels = set(Clabel)
        cy = Counter(Clabel)
        print('before:', cy)
        minV = min(cy.values()) * Nratio  # 1.1

        x = np.empty((0, features.shape[1]), features.dtype)
        y = np.empty((0,), np.int64)
        indices = []
        for i in set(Clabel):  # 采样平衡
            indice = np.where(
                np.isin(np.array(Clabel), i)
            )[0]

            if len(indice) > minV:
                indice = np.random.permutation(indice)[:int(minV)]
            indices.append(indice)
            # indices.update({i:indice})
            x = np.concatenate([x, features[indice]], axis=0)
            y = np.concatenate([y, Clabel[indice]], axis=0)
        featureNum = x.shape[0]
        print('Number of features:', featureNum)
        cy = Counter(y)
        print('after0:', cy)
        features = featureCols[0]
        Clabel = ClabelCols[0]
        for indice in indices:
            x = np.concatenate([x, features[indice]], axis=0)
            y = np.concatenate([y, Clabel[indice]], axis=0)
        # for i in set(Clabel):
        #     indice = np.where(
        #         np.isin(np.array(Clabel), i)
        #     )[0]
        #
        #     if len(indice) > minV:
        #         indice = np.random.permutation(indice)[:int(minV)]
        #     indices.append(indice)
        #     # indices.update({i:indice})
        #     x = np.concatenate([x, features[indice]], axis=0)
        #     y = np.concatenate([y, Clabel[indice]], axis=0)

        cy = Counter(y)
        print('after1:', cy)
        #################################################total
        print('processing tsne')
        if flag_3D:
            x_e, pca = pca_projection(x[:featureNum], y=y[:featureNum], dims=3)
            #np.savez(os.path.join(folder, basename + '3D.npz'), x=x_e, y=y, N=featureNum)
        else:
            x_e, pca = pca_projection(x[:featureNum], y=y[:featureNum], dims=2)
        x_e=pca.transform(x)
        np.savez(os.path.join(folder,basename+'.npz'), x=x_e, y=y, N=featureNum)
    # if fi==0:
    #     x_e,pca=pca_projection(x,y=y,dims=2)
    labels = set(y)
    fig = plt.figure()
    if flag_3D:
        ax=fig.gca(projection="3d")
    else:
        ax=fig.gca()
    string = list(map(lambda x: CLASSES_NAMES[x], labels))

    drawDiffClassST(ax,x_e[:featureNum], y[:featureNum], colors, marker='o', s=70)  # marker = ['o', '^']

    # ax.scatter(x_e[:, 0], x_e[:, 1], x_e[:, 2],
    #             edgecolors='k', )  # sns.color_palette(palettes[0]) alpha=0.5

    #ax.legend(string,loc=2,  borderaxespad=0., ncol=5)
    tt = ax.legend(string, loc=2, bbox_to_anchor=(1.2, -0.2), borderaxespad=0., ncol=7)
    fig = tt.figure
    fig.canvas.draw()
    bbox = tt.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig('legend1', dpi="figure", bbox_inches=bbox)
    drawDiffClassST(ax,x_e[featureNum:], y[featureNum:], colors, marker='^', s=40)
    #ax.legend(string+string, loc=2, borderaxespad=0., ncol=5)
    plt.savefig(os.path.join(folder,f"visual_{basename}.jpg"))
def main(args):

    # if cfg.SEMISUPNET.Trainer == "ubteacher":
    #     Trainer = UBTeacherTrainer
    # elif cfg.SEMISUPNET.Trainer == "baseline":
    #     Trainer = BaselineTrainer
    # else:
    #     raise ValueError("Trainer Name is not found.")
    featureFiles = ['Demooutput10/test_SS_SMD/featureROIweak2.pkl',
                    'Demooutput10/test_SS_SMD/featureROIstrong2.pkl']
    extractFlag=2
    if  extractFlag==0:#extract feature
        cfg = setup(args)
        cfg.SEMISUPNET.Trainer="baseline"
        cfg.MODEL.WEIGHTS='soutput10/model_final.pth'
        predictor = PredictorCustom(cfg)
        #featuresROI = predictor.feature_extract(featureFiles[1], True)
        for featureFile in featureFiles:
            if "weak" in featureFile:
                featuresROI=predictor.feature_extract(featureFile,False)
            else:
                featuresROI = predictor.feature_extract(featureFile,True)
        reduction_feature(featureFiles)
        reduction_feature_intra_class(featureFiles, force_reload=False)
        #featuresROI = predictor.feature_extract(False)
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
    elif extractFlag==1:#visual feature dimensionality reduction 单独文件
        reduction_feature(featureFiles)
    elif extractFlag == 2:#tsne
        reduction_feature_intra_class(featureFiles,flag_3D=False,force_reload=True)
    # elif extractFlag == 3:#PCA
    #     reduction_feature_cross_class(featureFiles,flag_3D=False,force_reload=True)

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
