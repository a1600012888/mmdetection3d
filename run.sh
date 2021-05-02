#CUDA_VISIBLE_DEVICES=0,1 bash tools/dist_train.sh plugin/depth_lidar_supervision/config.py 2
#bash tools/dist_train.sh plugin/depth_lidar_supervision/res18_c1x_t1x_4m.py 4
#CUDA_VISIBLE_DEVICES=1 python3 tools/train.py plugin/depth_lidar_supervision/res18_c4x_t1x_1m.py
#CUDA_VISIBLE_DEVICES=0,1,2,3 bash tools/dist_train.sh plugin/depth_lidar_supervision/res18_c4x_t1x_1m_steplr.py 4
sleep 4h
CUDA_VISIBLE_DEVICES=0,1,2,3 bash tools/dist_train.sh plugin/depth_lidar_supervision/res18_c4x_t1x_1m_steplr_e2x.py 4
