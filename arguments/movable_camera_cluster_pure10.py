ModelHiddenParams = dict(
    kplanes_config={
        "grid_dimensions": 2,
        "input_coordinate_dim": 4,
        "output_coordinate_dim": 32,
        "resolution": [64, 64, 64, 10],
    },
    multires=[1, 2, 4],
    defor_depth=1,
    net_width=128,
    plane_tv_weight=0.0002,
    time_smoothness_weight=0.001,
    l1_time_planes=0.0001,
    no_do=False,
    no_dshs=False,
    no_ds=False,
    no_dr=False,
    no_dx=False,
    render_process=False,
)

OptimizationParams = dict(
    dataloader=True,
    iterations=30000,
    batch_size=2,
    coarse_iterations=3000,
    densify_until_iter=10000,
    opacity_reset_interval=30000,
    opacity_threshold_coarse=0.005,
    opacity_threshold_fine_init=0.005,
    opacity_threshold_fine_after=0.005,
)
