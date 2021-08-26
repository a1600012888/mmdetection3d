#bash tools/dist_test.sh configs/detr3d_cam/resnet101_dcn.py models/epoch_2.pth  4 --eval bbox
#bash tools/dist_test.sh configs/detr3d_cam/test_resnet101_dcn.py models/epoch_2.pth  4 --eval bbox
# normal test

# bash tools/dist_test.sh configs/detr3d_cam/new_model.py models/epoch_12.pth  4 --eval bbox

# over test
# bash tools/dist_test.sh configs/detr3d_cam/test_new_model.py models/epoch_12.pth  4 --eval bbox
CUDA_VISIBLE_DEVICES=0, python3 tools/train_tracker.py plugin/track/configs/test_0.3_0.2.py
CUDA_VISIBLE_DEVICES=1, python3 tools/train_tracker.py plugin/track/configs/test_0.4_0.3.py
CUDA_VISIBLE_DEVICES=2, python3 tools/train_tracker.py plugin/track/configs/test_0.5_0.4.py
CUDA_VISIBLE_DEVICES=3, python3 tools/train_tracker.py plugin/track/configs/test_0.6_0.5.py
CUDA_VISIBLE_DEVICES=4, python3 tools/train_tracker.py plugin/track/configs/test_0.7_0.6.py
