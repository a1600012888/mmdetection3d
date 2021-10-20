python3 tools/visualize.py work_dirs/track_v2/pretrain_baseline_meminhead_1f_pre_3f_12ep/baseline_meminhead_3f.py work_dirs/track_v2/pretrain_baseline_meminhead_1f_pre_3f_12ep/latest.pth --out visualize/test_memin_head
mkdir -p work_dirs/visualization/test_track
python3 tools/visualization/draw_results.py --result_path visualize/test_memin_head/results_nusc.json --out work_dirs/visualization/test_track
