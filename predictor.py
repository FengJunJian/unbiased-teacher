from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup
from detectron2.engine.defaults import DefaultPredictor
from detectron2.data import build_detection_test_loader
from detectron2.data.build import _test_loader_from_config,_train_loader_from_config
import detectron2.data.transforms as T
from detectron2.data.detection_utils import read_image
from detectron2.utils.visualizer import Visualizer,ColorMode
from detectron2.data import MetadataCatalog
from detectron2.structures import Boxes,Instances
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.modeling.poolers import ROIPooler
from ubteacher.modeling.meta_arch.ts_ensemble import EnsembleTSModel
from ubteacher.engine.trainer import UBTeacherTrainer, BaselineTrainer

#from ubteacher.data.detection_utils import build_strong_augmentation
# hacky way to register
from tqdm import tqdm
from collections import defaultdict
import os
import json
import numpy as np
import cv2
import torch
from torchvision.ops.poolers import MultiScaleRoIAlign
import pickle
import torchvision.transforms as transforms
from PIL import Image

def build_strong_augmentation(p=0.1):
    """
    Create a list of :class:`Augmentation` from config.
    Now it includes resizing and flipping.

    Returns:
        list[Augmentation]
    """


    augmentation = []
    #p=0.1#0.1
    # This is simialr to SimCLR https://arxiv.org/abs/2002.05709
    augmentation.append(
        transforms.RandomApply([transforms.ColorJitter(0.7, 0.7, 0.7, 0.45)], p=p)#0.8
    )
    augmentation.append(transforms.RandomGrayscale(p=1.0))#0.2
    augmentation.append(transforms.RandomApply([transforms.GaussianBlur((7,7),(0.1, 2.0))], p=p))#0.5
    # transforms.AutoAugment()

    randcrop_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            #original
            transforms.RandomErasing(
                p=p, scale=(0.05, 0.2), ratio=(0.3, 3.3), value="random"# 0.7 0.07
            ),
            transforms.RandomErasing(
                p=p, scale=(0.02, 0.2), ratio=(0.1, 6), value="random" #0.5 0.05
            ),
            transforms.RandomErasing(
                p=p, scale=(0.02, 0.2), ratio=(0.05, 8), value="random" #0.3 0.03
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

    return transforms.Compose(augmentation)


class PredictorCustom(DefaultPredictor):
    def __init__(self, cfg):
        self.cfg = cfg.clone()  # cfg can be modified by model
        if cfg.SEMISUPNET.Trainer == "ubteacher":
            Trainer = UBTeacherTrainer
            model = Trainer.build_model(cfg)
            model_teacher = Trainer.build_model(cfg)
            ensem_ts_model = EnsembleTSModel(model_teacher, model)

            DetectionCheckpointer(
                ensem_ts_model, save_dir=cfg.OUTPUT_DIR
            ).resume_or_load(cfg.MODEL.WEIGHTS, resume=False)
            self.model=ensem_ts_model.modelTeacher#model_teacher
        else:
            Trainer = BaselineTrainer
            model = Trainer.build_model(cfg)
            DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
                cfg.MODEL.WEIGHTS, resume=False
            )
            self.model=model
        self.model.eval()
        if len(cfg.DATASETS.TEST):
            self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

        self.aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )

        self.input_format = cfg.INPUT.FORMAT
        self.instance_mode=ColorMode.IMAGE
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def __call__(self, original_image):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = self.aug.get_transform(original_image).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

            inputs = {"image": image, "height": height, "width": width}
            predictions = self.model([inputs])[0]
            return predictions

    def draw_batch(self):
        outputs={}
        for idx, dataset_name in enumerate(self.cfg.DATASETS.TEST):
            data_loader=build_detection_test_loader(self.cfg, dataset_name)

            outputs[dataset_name]=[]
            savepath=os.path.join(self.cfg.OUTPUT_DIR, dataset_name,'save')
            if not os.path.exists(savepath):
                os.makedirs(savepath)
            for idx, inputs in enumerate(data_loader):
                predictions = self.model(inputs)
                outputs[dataset_name].append(predictions)
                for jdx,input in enumerate(inputs):
                    filename=inputs[jdx]['file_name']
                    basename=os.path.basename(filename)
                    # image = input['image'].permute((1, 2, 0)).numpy().copy()
                    image=read_image(inputs[jdx]['file_name'])
                    H,W,C=image.shape
                    visualizer = Visualizer(image, self.metadata, instance_mode=self.instance_mode)
                    prediction=predictions[jdx]
                    if "instances" in prediction:
                        instances = prediction["instances"].to("cpu")
                        instances.pred_boxes.clip((H,W))#.tensor.detach().numpy()
                        boxes=instances.pred_boxes.tensor.detach().numpy()
                        for box in boxes:
                            # cv2.rectangle(image, (int(1), int(2)), (int(5), int(8)), (255, 0, 0), 2)
                            cv2.rectangle(image,(int(box[0]),int(box[1])),(int(box[2]),int(box[3])),(255,0,0),2)
                        vis_output = visualizer.draw_instance_predictions(predictions=instances)

                        vis_output.save(os.path.join(savepath,basename))
                        #cv2.imshow('a',vis_output.get_image())
                        # cv2.imshow('b',image)
                        # cv2.waitKey(0)
        return outputs

    def feature_extract(self,saveFile,augFlag=True,):
        print("using strong aug:",augFlag)
        strong_aug=None
        if augFlag:
            strong_aug=build_strong_augmentation(0.01)
        #image_pil = Image.fromarray(image_weak_aug.astype("uint8"), "RGB")

        # image_strong_aug = np.array(strong_aug(image_pil))
        # dataset_dict["image"] = torch.as_tensor(
        #     np.ascontiguousarray(image_strong_aug.transpose(2, 0, 1))
        # )
        fea_dict = defaultdict(list)
        output_size = self.cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION#[8, 16]
        #pooler_resolution =
        for idx, dataset_name in enumerate(self.cfg.DATASETS.TEST):
            data_loader = build_detection_test_loader(self.cfg, dataset_name,mapper= DatasetMapper(self.cfg, True))
            savepath=os.path.dirname(saveFile)
            if not os.path.exists(savepath):
                os.makedirs(savepath)
            for idx, inputs in enumerate(tqdm(data_loader)):
                #predictions = self.model(inputs)
                import matplotlib.pyplot as plt
                if augFlag:
                    for data_dict in inputs:
                        original_image=data_dict['image'].permute((1, 2, 0)).numpy()[:,:,::-1]
                        #height, width = inputs[0]['height'],inputs[0]['width']
                        #image_pil = Image.fromarray(original_image.astype("uint8"), "RGB")
                        str_img=strong_aug(Image.fromarray(original_image.astype("uint8"), "RGB"))
                        #str_img.show()
                        image_strong_aug = np.array(str_img)
                        #image_pil = Image.fromarray(original_image.astype("uint8"), "RGB")
                        # inputs["image"] = torch.as_tensor(
                        #     np.ascontiguousarray(image_strong_aug.transpose(2, 0, 1))
                        # )
                        data_dict['image'] = torch.as_tensor(
                            np.ascontiguousarray(image_strong_aug.transpose(2, 0, 1))
                        )

                images = self.model.preprocess_image(inputs)
                image_sizes = images.image_sizes
                #if "instances" in inputs[0]:
                assert "instances" in inputs[0]
                gt_instances = None
                bboxes=[x["instances"].get('gt_boxes').tensor.to(self.model.device) for x in inputs]
                gt_classes=torch.cat([x["instances"].get('gt_classes') for x in inputs],dim=0).tolist()

                features = self.model.backbone(images.tensor)
                # proposals_rpn, proposal_losses = self.model.proposal_generator(
                #     images, features, gt_instances, compute_loss=False,compute_val_loss=False,
                # )
                # map_layer_to_index = {"p2":0,"p3": 1, "p4": 2, "p5": 3, "p6": 4,}
                featmap_names = self.model.roi_heads.box_in_features  # map_layer_to_index.keys()
                featureslist = [features[f] for f in featmap_names]
                pooledfeatures = self.model.roi_heads.box_pooler(featureslist,[Boxes(box) for box in bboxes] )#[x.proposal_boxes for x in proposals]
                # roi_head lower branch



                # RoIpooler = MultiScaleRoIAlign(featmap_names=featmap_names, output_size=output_size, sampling_ratio=0)
                # pooledfeatures = RoIpooler(features, bboxes, image_sizes)

                if len(pooledfeatures) == 0:
                    # return fea_list
                    #fea_list.append(None)
                    continue
                else:
                    pi = pooledfeatures.cpu().detach().numpy()  # for p0
                    # if len(pooledfeatures)>2:
                    #     print(pooledfeatures.shape)
                    # print('fg:',fg_num_boxes,'bg:',bg_num_boxes,'fea shape:',pi.shape)
                    assert pi.shape[0]==len(gt_classes)
                    for i in range(pi.shape[0]):
                        fea_dict[gt_classes[i]].append(pi[i])
                        #fea_dict[gt_classes[i]].append(pi[i].mean(0))
            #if augFlag:
        with open(saveFile,"wb") as f:
            pickle.dump(fea_dict, f)
            # else:
            #     with open(os.path.join(savepath,'featureROIweak1.pkl'),"wb") as f:
            #         pickle.dump(fea_dict,f)

            #np.save(os.path.join(savepath,'featureROI'),fea_dict)

def visual_img_torch(img):
    import cv2
    img_arr=img.permute((1,2,0)).numpy()
    cv2.imshow('a',img_arr)
    cv2.waitKey()


