from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup
from detectron2.engine.defaults import DefaultPredictor

from ubteacher.engine.trainer import UBTeacherTrainer, BaselineTrainer
from detectron2.utils.visualizer import Visualizer,ColorMode
from detectron2.data import MetadataCatalog
from detectron2.structures import Boxes
# hacky way to register
import ubteacher.data.datasets.builtin
import os
import json
import cv2
import torch
from ubteacher.modeling.meta_arch.ts_ensemble import EnsembleTSModel
from detectron2.data import build_detection_test_loader
import detectron2.data.transforms as T
from detectron2.data.detection_utils import read_image
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
            ).resume_or_load(os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.WEIGHTS), resume=False)
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