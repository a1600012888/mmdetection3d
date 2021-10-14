sleep 4000
bash tools/dist_train.sh plugin/add_radar/configs_radar_rectify/res50_sparse_static_mul4_1200_30_velo1.0_24ep.py 8 --work-dir=work_dirs/radar_det/res50_sparse_static_mul4_1200_30_velo1.0_24ep
#CUDA_VISIBLE_DEVICES=0,1,2,4,5,6 bash tools/dist_train.sh plugin/add_radar/configs_radar_rectify/res50_sparse_static_mul4_1200_30_velo1.0_24ep.py 6 --work-dir=work_dirs/radar_det/res50_sparse_static_mul4_1200_30_velo1.0_24ep
