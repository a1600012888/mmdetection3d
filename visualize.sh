#bash tools/dist_test.sh configs/detr3d_cam/resnet101_dcn.py models/epoch_2.pth  4 --eval bbox
#python3 tools/visualize.py configs/detr3d_cam/test_resnet101_dcn.py models/epoch_2.pth --out visualize/res101_epoch2.pkl  --eval bbox
#bash tools/dist_visual.sh configs/detr3d_cam/test_resnet101_dcn.py models/epoch_2.pth 1 --out visualize/res101_epoch2.pkl --show --show-dir visualize/res101_epoch2/test  --eval bbox
bash tools/dist_visual.sh configs/detr3d_cam/test_new_model.py models/epoch_12.pth 1 --out visualize/new_model.pkl --show --show-dir visualize/new_model/test  --eval bbox
