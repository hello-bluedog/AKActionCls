from slowfast.utils.parser import load_config, parse_args
from slowfast.config.defaults import assert_and_infer_cfg
from slowfast.datasets import utils
from slowfast.datasets import video_container as container
from slowfast.datasets import decoder as decoder
from slowfast.models import build_model
import slowfast.utils.checkpoint as cu
import torch
import numpy as np
import slowfast.utils.distributed as du
def getInput(video_path : str, cfg):
    min_scale, max_scale, crop_size = ([224] * 3)
    sampling_rate = utils.get_random_sampling_rate(
        cfg.MULTIGRID.LONG_CYCLE_SAMPLING_RATE,
        cfg.DATA.SAMPLING_RATE,
    )
    video_container = container.get_video_container(
        video_path,
        cfg.DATA_LOADER.ENABLE_MULTI_THREAD_DECODE,
        cfg.DATA.DECODING_BACKEND,
    )
    frames = decoder.decode(
        video_container,
        sampling_rate,
        cfg.DATA.NUM_FRAMES,
        0,
        cfg.TEST.NUM_ENSEMBLE_VIEWS,
        video_meta={},
        target_fps=cfg.DATA.TARGET_FPS,
        backend=cfg.DATA.DECODING_BACKEND,
        max_spatial_scale=min_scale,
        use_offset=cfg.DATA.USE_OFFSET_SAMPLING,
        sparse=True
    )
    frames = utils.tensor_normalize(
        frames, cfg.DATA.MEAN, cfg.DATA.STD
    )
    # T H W C -> C T H W.
    frames = frames.permute(3, 0, 1, 2)
    # Perform data augmentation.
    frames = utils.spatial_sampling(
        frames,
        spatial_idx=0,
        min_scale=min_scale,
        max_scale=max_scale,
        crop_size=crop_size,
        random_horizontal_flip=cfg.DATA.RANDOM_FLIP,
        inverse_uniform_sampling=cfg.DATA.INV_UNIFORM_SAMPLE,
    )
    return frames


if __name__ == "__main__":
    args = parse_args()
    print(args)
    cfg = load_config(args)
    cfg = assert_and_infer_cfg(cfg)
    tensor = getInput(args.video_path , cfg)
    du.init_distributed_training(cfg)
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    model = build_model(cfg)
    
    cu.load_test_checkpoint(cfg, model)
    #print(tensor.shape)

