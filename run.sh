sleep 21600
bash tools/dist_train_tracker.sh plugin/track/fp16/base_3f.py 8 --work-dir=work_dirs/track/v2/fp16/base3f_mem_24ep_test_truck
