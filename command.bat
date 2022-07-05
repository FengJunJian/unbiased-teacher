rem python train_net.py --num-gpus 1 --config configs/semiship/faster_rcnn_R_50_FPN_sup1_run1.yaml SOLVER.IMG_PER_BATCH_LABEL 2 SOLVER.IMG_PER_BATCH_UNLABEL 2 DATALOADER.SUP_PERCENT 0.5 OUTPUT_DIR ./output05
rem python train_net.py --num-gpus 1 --config configs/semiship/faster_rcnn_R_50_FPN_sup1_run1.yaml SOLVER.IMG_PER_BATCH_LABEL 2 SOLVER.IMG_PER_BATCH_UNLABEL 2 DATALOADER.SUP_PERCENT 1.0 OUTPUT_DIR ./output1
rem python train_net.py --num-gpus 1 --config configs/semiship/faster_rcnn_R_50_FPN_sup1_run1.yaml SOLVER.IMG_PER_BATCH_LABEL 2 SOLVER.IMG_PER_BATCH_UNLABEL 2 DATALOADER.SUP_PERCENT 2.0 OUTPUT_DIR ./output2
python train_net.py --config configs/semiship/faster_rcnn_R_50_FPN_sup1_run1.yaml SOLVER.IMG_PER_BATCH_LABEL 2 SOLVER.IMG_PER_BATCH_UNLABEL 2 DATALOADER.SUP_PERCENT 5.0 OUTPUT_DIR ./output5
python train_net.py --config configs/semiship/faster_rcnn_R_50_FPN_sup1_run1.yaml SOLVER.IMG_PER_BATCH_LABEL 2 SOLVER.IMG_PER_BATCH_UNLABEL 2 DATALOADER.SUP_PERCENT 10.0 OUTPUT_DIR ./output10


rem test
rem python train_net.py --resume --config configs/semiship/faster_rcnn_R_50_FPN_sup1_run1.yaml SOLVER.IMG_PER_BATCH_LABEL 2 SOLVER.IMG_PER_BATCH_UNLABEL 2 DATALOADER.SUP_PERCENT 5.0 OUTPUT_DIR ./outputDemo MODEL.WEIGHTS output2/model_final.pth DATASETS.TEST ('test10',) SOLVER.MAX_ITER 21
