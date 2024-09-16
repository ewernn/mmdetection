from mmdeploy.apis import torch2onnx
from mmdeploy.backend.sdk.export_info import export2SDK

work_dir = '/home/eawern/mmdetection/'
img = work_dir + 'EqNeckData/Im0.tif'
save_file = work_dir + 'vertebrae_obj_detection_1000x10000.onnx'

deploy_cfg = '../mmdeploy/configs/mmdet/detection/detection_onnxruntime_static.py'
model_cfg = 'configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py'
model_checkpoint = 'exps/exp1/epoch_83.pth'
device = 'cuda'

# 1. convert model to onnx
torch2onnx(img, work_dir, save_file, deploy_cfg, model_cfg,
           model_checkpoint, device)

# 2. extract pipeline info for inference by MMDeploy SDK
export2SDK(deploy_cfg, model_cfg, work_dir, pth=model_checkpoint,
           device=device)


# python \
# mmdeploy/tools/deploy.py \
# mmdeploy/configs/mmdet/detection/detection_onnxruntime_static.py \
# mmdetection/configs/faster_rcnn/EQUINE_faster-rcnn_r50_fpn_1x_coco.py \
# mmdetection/exps/exp1/epoch_83.pth \
# mmdetection/EqNeckData/Im0.tif \
# --work-dir john_model \
# --device cuda \
# --dump-info
#python mmdeploy/tools/deploy.py mmdeploy/configs/mmdet/detection/detection_onnxruntime_static.py mmdetection/configs/faster_rcnn/EQUINE_faster-rcnn_r50_fpn_1x_coco.py mmdetection/exps/exp1/epoch_83.pth mmdetection/EqNeckData/Im0.tif --work-dir john_model --device cuda --dump-info


# orig
# python mmdeploy/tools/deploy.py \
#     mmdeploy/configs/mmdet/detection/detection_tensorrt_dynamic-320x320-1344x1344.py \
#     mmdetection/configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py \
#     checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
#     mmdetection/demo/demo.jpg \
#     --work-dir mmdeploy_model/faster-rcnn \
#     --device cuda \
#     --dump-info