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
import ubteacher.data.datasets.builtin
from predictor import PredictorCustom
import os
import json
import torch
from ubteacher.modeling.meta_arch.ts_ensemble import EnsembleTSModel
from pycocotools.coco import COCO
import argparse
import sys
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


def main(args):
    cfg = setup(args)
    if cfg.SEMISUPNET.Trainer == "ubteacher":
        Trainer = UBTeacherTrainer
    elif cfg.SEMISUPNET.Trainer == "baseline":
        Trainer = BaselineTrainer
    else:
        raise ValueError("Trainer Name is not found.")
    if  args.predict_only:
        predictor = PredictorCustom(cfg)
        outputs = predictor.draw_batch()
        # for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
        #     results=res[dataset_name]['bbox']
        #     with open(os.path.join(output_folders[idx],'AP.txt'),'w') as f:
        #         json.dump(results,f)
        #     predictions = torch.load(os.path.join(output_folders[idx], 'instances_predictions.pth'))
        #     pre=COCO(predictions)
        # saveImgPath = os.path.join(output_folder, 'img')
        # Visualizer()
        return
    elif args.eval_only:
        output_folders=[]
        if cfg.SEMISUPNET.Trainer == "ubteacher":
            model = Trainer.build_model(cfg)
            model_teacher = Trainer.build_model(cfg)
            ensem_ts_model = EnsembleTSModel(model_teacher, model)

            DetectionCheckpointer(
                ensem_ts_model, save_dir=cfg.OUTPUT_DIR
            ).resume_or_load(os.path.join(cfg.OUTPUT_DIR,cfg.MODEL.WEIGHTS), resume=args.resume)

            #output_folder = os.path.join(cfg.OUTPUT_DIR, "inference_final")
            evaluators = []
            for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
                output_folder = os.path.join(cfg.OUTPUT_DIR, "inference_final", dataset_name)
                output_folders.append(output_folder)
                evaluators.append([Trainer.build_evaluator(cfg, dataset_name, output_folder=output_folder)])

            # self._last_eval_results_student = self.test(self.cfg, self.model, evaluators)
            res = Trainer.test(cfg, ensem_ts_model.modelTeacher,evaluators)

        else:
            model = Trainer.build_model(cfg)
            DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
                cfg.MODEL.WEIGHTS, resume=args.resume
            )
            evaluators = []
            for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
                output_folder = os.path.join(cfg.OUTPUT_DIR, "inference_final", dataset_name)
                output_folders.append(output_folder)
                evaluators.append([Trainer.build_evaluator(cfg, dataset_name, output_folder=output_folder)])
            res = Trainer.test(cfg, model, evaluators)

        # path='E:/DA1/SW/logSMDToSSship/inference2500/ship_test_SeaShips_cocostyle/bbox.json'
        # with open(path,'r') as f:
        #     pre=json.load(f)
        return res


    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)

    return trainer.train()


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
