rem python train_net.py --config configs/semiship/faster_rcnn_R_50_FPN_sup1_run1.yaml DATALOADER.SUP_PERCENT 10.0 OUTPUT_DIR ./foutput10

python train_net.py --config configs/semiship/faster_rcnn_R_50_FPN_sup1_run1.yaml DATALOADER.SUP_PERCENT 10.0 OUTPUT_DIR ./fsoutput10_8_EMA0999 SEMISUPNET.BBOX_THRESHOLD 0.8

rem python train_net.py --config configs/semiship/faster_rcnn_R_50_FPN_sup1_run1.yaml DATALOADER.SUP_PERCENT 5.0 OUTPUT_DIR ./foutput5
rem python train_net.py --config configs/semiship/faster_rcnn_R_50_FPN_sup1_run1.yaml DATALOADER.SUP_PERCENT 0.5 OUTPUT_DIR ./ssoutput05
rem python train_net.py --config configs/semiship/faster_rcnn_R_50_FPN_sup1_run1.yaml DATALOADER.SUP_PERCENT 1.0 OUTPUT_DIR ./foutput1
rem python train_net.py --config configs/semiship/faster_rcnn_R_50_FPN_sup1_run1.yaml DATALOADER.SUP_PERCENT 2.0 OUTPUT_DIR ./ssoutput2


rem test
rem python train_net.py --config configs/semiship/faster_rcnn_R_50_FPN_sup1_run1.yaml DATALOADER.SUP_PERCENT 10.0 OUTPUT_DIR ./fwoutput10_EMA09_UNSUP01 SEMISUPNET.UNSUP_LOSS_WEIGHT 0.1 SEMISUPNET.EMA_KEEP_RATE 0.9
rem python train_net.py --config configs/semiship/faster_rcnn_R_50_FPN_sup1_run1.yaml DATALOADER.SUP_PERCENT 10.0 OUTPUT_DIR ./fwoutput10_EMA09_UNSUP001 SEMISUPNET.UNSUP_LOSS_WEIGHT 0.01 SEMISUPNET.EMA_KEEP_RATE 0.9
rem python train_net.py --config configs/semiship/faster_rcnn_R_50_FPN_sup1_run1.yaml DATALOADER.SUP_PERCENT 10.0 OUTPUT_DIR ./fwoutput10_EMA099_UNSUP01 SEMISUPNET.UNSUP_LOSS_WEIGHT 0.1 SEMISUPNET.EMA_KEEP_RATE 0.99

rem python train_net.py --config configs/semiship/faster_rcnn_R_50_FPN_sup1_run1.yaml DATALOADER.SUP_PERCENT 10.0 OUTPUT_DIR ./fwoutput10
rem python train_net.py --config configs/semiship/faster_rcnn_R_50_FPN_sup1_run1.yaml DATALOADER.SUP_PERCENT 10.0 OUTPUT_DIR ./fwoutput10_EMA09
rem python train_net.py --config configs/semiship/faster_rcnn_R_50_FPN_sup1_run1.yaml DATALOADER.SUP_PERCENT 10.0 OUTPUT_DIR ./fwoutput10_UNSUP01 SEMISUPNET.UNSUP_LOSS_WEIGHT 0.1
rem python train_net.py --config configs/semiship/faster_rcnn_R_50_FPN_sup1_run1.yaml DATALOADER.SUP_PERCENT 10.0 OUTPUT_DIR ./foutput10_5 SEMISUPNET.BBOX_THRESHOLD 0.5
rem python train_net.py --resume --config configs/semiship/faster_rcnn_R_50_FPN_sup1_run1.yaml SOLVER.IMG_PER_BATCH_LABEL 2 SOLVER.IMG_PER_BATCH_UNLABEL 2 DATALOADER.SUP_PERCENT 5.0 OUTPUT_DIR ./outputDemo MODEL.WEIGHTS output2/model_final.pth DATASETS.TEST ('test10',) SOLVER.MAX_ITER 21
rem python train_net.py --eval-only --config configs/semiship/faster_rcnn_R_50_FPN_sup1_run1.yaml MODEL.WEIGHTS model_final.pth DATASETS.TEST ('test10',) OUTPUT_DIR output10
rem --predict-only --eval-only --config configs/semiship/faster_rcnn_R_50_FPN_sup1_run1.yaml MODEL.WEIGHTS soutput10/model_final.pth DATASETS.TEST ('test10',) OUTPUT_DIR soutput10 SEMISUPNET.Trainer baseline
