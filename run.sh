#CUDA_VISIBLE_DEVICES=0,1 bash tools/dist_train.sh plugin/depth_lidar_supervision/config.py 2
#bash tools/dist_train.sh plugin/depth_lidar_supervision/res18_c1x_t1x_4m.py 4
#CUDA_VISIBLE_DEVICES=1 python3 tools/train.py plugin/depth_lidar_supervision/res18_c4x_t1x_1m.py
#CUDA_VISIBLE_DEVICES=0,1,2,3 bash tools/dist_train.sh plugin/depth_lidar_supervision/res18_c4x_t1x_1m_steplr.py 4
#CUDA_VISIBLE_DEVICES=0,1,2,3 bash tools/dist_train.sh plugin/depth_v2/res50_c4x_t1x_1m_steplr.py 4 --resume-from work_dirs/res50_c4x_t1x_1m_steplr/epoch_60.pth
#PORT=12345 CUDA_VISIBLE_DEVICES=0,1,2,3 bash tools/dist_train.sh plugin/radar/configs/depth_sup_radar_0.01.py 4 #--resume-from /home/zhangty/projects/mmdetection3d/work_dirs/res50_c4x_t1x_1m_steplr_crop/latest.pth
PORT=12345 CUDA_VISIBLE_DEVICES=0,1,2,3 bash tools/dist_train.sh plugin/radar/configs/unsup_radar.py 4
PORT=12345 CUDA_VISIBLE_DEVICES=0,1,2,3 bash tools/dist_train.sh plugin/radar/configs/unsup_radar_lr1e-3.py 4
PORT=12345 CUDA_VISIBLE_DEVICES=0,1,2,3 bash tools/dist_train.sh plugin/radar/configs/unsup_radar_base.py 4
