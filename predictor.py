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

def ncolors(num_color):
    #import seaborn
    import colorsys
    # colors = []
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
            map(lambda x: (int(x[0]*255), int(x[1]*255), int(x[2]*255)),
                colors))
        colors = [c[::-1] for c in colors]

    return colors
def write_detection(im, dets, thiness=5,colors=None, class_name=None, GT_color=None, show_score=False):
    '''
    dets:xmin,ymin,xmax,ymax,score
    '''
    H, W, C = im.shape
    for i in range(len(dets)):
        rectangle_tmp = im.copy()
        bbox = dets[i, :4].astype(np.int32)
        #class_ind = int(dets[i, 4])
        # if class_ind==7:#ignore flying
        #     continue
        # if show_score:
        #     score = dets[i, -1]
        if GT_color:
            color = GT_color
        else:
            if colors is None:
                raise ValueError("colors is None!")
            color = colors[i]

        string = class_name[i]#[class_ind]
        if show_score:
            string += '%.2f' % (dets[i, -1])

        # string = '%s' % (CLASSES[class_ind])
        fontFace = cv2.FONT_HERSHEY_SIMPLEX#cv2.FONT_HERSHEY_COMPLEX

        fontScale = 1.5
        # thiness = 2
        Texthiness=2
        text_size, baseline = cv2.getTextSize(string, fontFace, fontScale, Texthiness)
        text_origin = (bbox[0] - 2, bbox[1]+4)  # - text_size[1]
        ###########################################putText
        p1 = [text_origin[0] - 1, text_origin[1] + 1] #(x,y)
        p2 = [text_origin[0] + text_size[0] + 1, text_origin[1] - text_size[1] - 2]
        if p2[0] > W:
            dw = p2[0] - W
            p2[0] -= dw
            p1[0] -= dw

        rectangle_tmp = cv2.rectangle(rectangle_tmp, (p1[0], p1[1]),
                                      (p2[0], p2[1]),
                                      color, cv2.FILLED)
        cv2.addWeighted(im, 0.7, rectangle_tmp, 0.3, 0, im)
        im = cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, thiness)
        # imt=im.copy()
        im=cv2.putText(im,string,(p1[0]+1,p1[1]-1), fontFace, fontScale, (0, 0, 0), Texthiness,lineType=-1)
        # im = cv2AddChineseText(im, string, (p1[0] + 1, p2[1] - 1), (0, 0, 0), )
        # cv2.imshow('a',imt)
        # cv2.waitKey()
        # im = cv2.putText(im, string, (p1[0]+1,p1[1]-1),
        #                  fontFace, fontScale, (0, 0, 0), thiness,lineType=-1)
    return im

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
        self.colors=ncolors(cfg.MODEL.ROI_HEADS.NUM_CLASSES)
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

    def draw_detection(self,savefolder,coco_instances_results,threshold=0.5):
        from pycocotools.coco import COCO
        ImgPath = 'E:/SeaShips_SMD/JPEGImages'
        annopath = 'E:/SeaShips_SMD/test_SS_SMD_cocostyle.json'
        #savepath = os.path.join(self.cfg.OUTPUT_DIR, dataset_name, 'saveImg' + comment)
        if not os.path.exists(savefolder):
            os.makedirs(savefolder)

        cocogt=COCO(annopath)
        cocodt=cocogt.loadRes(coco_instances_results)
        imgIds=cocogt.getImgIds()
        print("first image:", cocogt.loadImgs(1)[0]['file_name'])
        for imgId in tqdm(imgIds):
            imgr=cocogt.loadImgs(imgId)[0]
            filename=imgr['file_name']
            annIds=cocodt.getAnnIds(imgIds=imgr['id'],catIds=[],iscrowd=None)
            anns=cocodt.loadAnns(annIds)
            img=cv2.imread(os.path.join(ImgPath,filename))
            # for imgId in imgIds:
            class_name=[]
            colors=[]
            bboxes=np.empty((0,5),dtype=np.float32)
            for ann in anns:
                bbox=ann['bbox']
                score=ann['score']
                if score<threshold:
                    continue
                class_name.append(cocogt.loadCats(ann['category_id'])[0]['name'])
                #print(ann['category_id'])
                colors.append(self.colors[ann['category_id']-1])
                bboxes=np.concatenate([bboxes,np.array([[bbox[0],bbox[1],bbox[0]+bbox[2],bbox[1]+bbox[3],score]])],axis=0)
                #bboxes.append()
            bboxes=np.array(bboxes)
            im=write_detection(img,bboxes,class_name=class_name,colors=colors,show_score=True)
            # cv2.imshow('a',im)
            # cv2.waitKey(1)
            cv2.imwrite(os.path.join(savefolder,filename),im)

    def draw_GT(self,savefolder,):
        from pycocotools.coco import COCO

        #savepath = os.path.join(self.cfg.OUTPUT_DIR, dataset_name, 'saveImg' + comment)
        if not os.path.exists(savefolder):
            os.makedirs(savefolder)

        ImgPath='E:/SeaShips_SMD/JPEGImages'
        annopath='E:/SeaShips_SMD/test_SS_SMD_cocostyle.json'
        cocogt=COCO(annopath)

        imgIds=cocogt.getImgIds()
        print("first image:",cocogt.loadImgs(1)[0]['file_name'])
        for imgId in tqdm(imgIds):
            imgr=cocogt.loadImgs(imgId)[0]
            filename=imgr['file_name']
            annIds=cocogt.getAnnIds(imgIds=imgr['id'],catIds=[],iscrowd=None)
            anns=cocogt.loadAnns(annIds)
            img=cv2.imread(os.path.join(ImgPath,filename))
            # for imgId in imgIds:
            class_name=[]
            colors=[]
            bboxes=[]
            for ann in anns:
                bbox=ann['bbox']
                class_name.append(cocogt.loadCats(ann['category_id'])[0]['name'])
                #print(ann['category_id'])
                colors.append(self.colors[ann['category_id']-1])
                bboxes.append([bbox[0],bbox[1],bbox[0]+bbox[2],bbox[1]+bbox[3]])
            bboxes=np.array(bboxes)
            im=write_detection(img,bboxes,class_name=class_name,colors=colors,show_score=False)
            # cv2.imshow('a',im)
            # cv2.waitKey(1)
            cv2.imwrite(os.path.join(savefolder,filename),im)


    def feature_extract(self,saveFile,augFlag=True,):
        print("using strong aug:",augFlag)
        strong_aug=None
        if augFlag:
            strong_aug=build_strong_augmentation(0.8)
        #image_pil = Image.fromarray(image_weak_aug.astype("uint8"), "RGB")

        # image_strong_aug = np.array(strong_aug(image_pil))
        # dataset_dict["image"] = torch.as_tensor(
        #     np.ascontiguousarray(image_strong_aug.transpose(2, 0, 1))
        # )
        fea_dict = defaultdict(list)
        output_size = self.cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION#[8, 16]
        #pooler_resolution =
        for idx, dataset_name in enumerate(['train_SS_SMD0',]):#self.cfg.DATASETS.TRAIN
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


