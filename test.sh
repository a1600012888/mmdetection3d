#bash tools/dist_test.sh configs/detr3d_cam/resnet101_dcn.py models/epoch_2.pth  4 --eval bbox
#bash tools/dist_test.sh configs/detr3d_cam/test_resnet101_dcn.py models/epoch_2.pth  4 --eval bbox
# normal test
bash tools/dist_test.sh configs/detr3d_cam/new_model.py models/epoch_12.pth  4 --eval bbox

# over test
bash tools/dist_test.sh configs/detr3d_cam/test_new_model.py models/epoch_12.pth  4 --eval bbox
