# packnet_sceneflow_net

## 5.13 更新说明
- 依赖文件在geometry和utils中
- 新增函数reduce_loss, 用于处理list of tensor, tensor.shape可以是(6,*,H,W),返回一个torch.tensor([1])
reduce_loss可用于处理最后所有photometric_loss的list，我直接用在scene_flow_consistency_loss中了

- 主要文件说明：
    - temporal_spatial_warp中有5个函数，分别是：
        - warp_ref_image_temporal, 时域warp
        - warp_ref_image_spatial, cross camera warp
        - warp_ref_image_temporal_spatial, 时空warp
        - calc_scene_flow_consistency_loss, 返回一个torch.tensor([1])
        - reduce_loss, 如上述
    - photometric_loss如题
        - PhotometricLoss类的文件中有计算SSIM的函数，也有一些默认参数
- 所有文件计算的正确性经过人脑检测，但未经过程序检测