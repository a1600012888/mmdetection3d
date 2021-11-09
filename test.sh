#bash tools/dist_test.sh configs/detr3d_cam/resnet101_dcn.py models/epoch_2.pth  4 --eval bbox
#bash tools/dist_test.sh configs/detr3d_cam/test_resnet101_dcn.py models/epoch_2.pth  4 --eval bbox
# normal test

# bash tools/dist_test.sh configs/detr3d_cam/new_model.py models/epoch_12.pth  4 --eval bbox

# over test
# bash tools/dist_test.sh configs/detr3d_cam/test_new_model.py models/epoch_12.pth  4 --eval bbox
#CUDA_VISIBLE_DEVICES=4, python3 tools/train_tracker.py plugin/track/configs/test_0.7_0.6.py
CUDA_VISIBLE_DEVICES=0, python3 tools/test.py plugin/track/configs_test/baseline_3f_test0.15.py work_dirs/baseline_3f_24ep/epoch_24.pth --eval bbox
CUDA_VISIBLE_DEVICES=1, python3 tools/test.py plugin/track/configs_test/baseline_3f_test0.2.py work_dirs/baseline_3f_24ep/epoch_24.pth --eval bbox
CUDA_VISIBLE_DEVICES=2, python3 tools/test.py plugin/track/configs_test/baseline_3f_test0.3.py work_dirs/baseline_3f_24ep/epoch_24.pth --eval bbox
CUDA_VISIBLE_DEVICES=3, python3 tools/test.py plugin/track/configs_test/baseline_3f_test0.5.py work_dirs/baseline_3f_24ep/epoch_24.pth --eval bbox
CUDA_VISIBLE_DEVICES=4, python3 tools/test.py plugin/track/configs_test/baseline_3f_test0.6.py work_dirs/baseline_3f_24ep/epoch_24.pth --eval bbox
CUDA_VISIBLE_DEVICES=4, python3 tools/test.py plugin/track/configs_test/baseline_3f_test0.6.py work_dirs/baseline_3f_24ep/epoch_24.pth --eval bbox
CUDA_VISIBLE_DEVICES=5, python3 tools/test.py plugin/track/configs_test/baseline_3f_test0.1.py work_dirs/baseline_3f_24ep/epoch_24.pth --eval bbox
