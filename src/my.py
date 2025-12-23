import torch
from pathlib import Path

import einops
from PIL import Image
from kappautils.images.png import png_writer_viridis
from kappautils.images.points_to_image import coords_to_image

from datasets import dataset_from_kwargs
from models import model_from_kwargs
from providers.dataset_config_provider import DatasetConfigProvider
from providers.path_provider import PathProvider
from utils.data_container import DataContainer
from utils.kappaconfig.util import get_stage_hp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    stage_hp = get_stage_hp(
        hp_file="./src/yamls/cfd/e100_lr5e5_8M_lat512_fp32.yaml",
        template_path="./src/zztemplates"
    )
    
    path_provider = PathProvider(
        output_path=Path("./outputs"),
        model_path=None,
        stage_name="stage1",
        stage_id="sdfsfg",
        temp_path=Path("temp"),
    )
    
    datasets_dict = {}
    dataset_config_provider = DatasetConfigProvider(
        global_dataset_paths={"mesh_dataset": "/home/dell/datasets/transientflow2D"}
    )
    
    # 初始化数据集（参考 train_stage.py 的用法）
    for dataset_key, dataset_kwargs in stage_hp["datasets"].items():
        datasets_dict[dataset_key] = dataset_from_kwargs(
            dataset_config_provider=dataset_config_provider,
            path_provider=path_provider,
            num_supernodes=dataset_kwargs["collators"][0]["num_supernodes"],
            **dataset_kwargs,
        )
    
    data_container_kwargs = {}
    data_container = DataContainer(
        **datasets_dict,
        **data_container_kwargs,
        seed = 182376
    )

    # 只为方便调试数据，这里暂时不构建完整模型
    # 如果以后需要，可以参考 train_stage.py 中的用法来创建 trainer 和 model

    # 1) 看一下 train 数据集的基本信息
    raw_dataset = data_container.get_dataset("train")
    print(f"raw train dataset type: {type(raw_dataset)}  len={len(raw_dataset)}")

    # 2) 使用 ModeWrapper 取 'x'（输入）
    dataset_x, collator_x = data_container.get_dataset("train", mode="x")
    sample_x, ctx_x = dataset_x[0]
    print("\n=== sample x ===")
    print("x.shape:", sample_x.shape)
    # 打印前几个点的前几个通道
    print("x[0:5, 0:5]:\n", sample_x[:5, :5])

    # 3) 使用 ModeWrapper 取 'target'（要预测的下一步流场）
    dataset_y, collator_y = data_container.get_dataset("train", mode="target")
    sample_y, ctx_y = dataset_y[0]
    print("\n=== sample target ===")
    print("target.shape:", sample_y.shape)
    print("target[0:5, 0:5]:\n", sample_y[:5, :5])

    # 4) 取一下 mesh_pos（网格点坐标），看一下几何信息
    dataset_pos, _ = data_container.get_dataset("train", mode="mesh_pos")
    sample_pos, ctx_pos = dataset_pos[0]
    print("\n=== sample mesh_pos ===")
    print("mesh_pos.shape:", sample_pos.shape)
    print("mesh_pos[0:5]:\n", sample_pos[:5])

    model = model_from_kwargs(
                **stage_hp["model"],
                input_shape = (None, 6),
                output_shape = (None, 3),
                path_provider=path_provider,
                data_container=data_container,
            )
    
    def load_ckpt(ckpt_path: Path, submodel):
        print(f"\nLoading checkpoint from: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device)
        state_dict = checkpoint['state_dict']
        try:
            incompatible = submodel.load_state_dict(state_dict, strict=False)
            # 打印一下加载结果，方便检查是否存在未匹配的键
            missing = getattr(incompatible, "missing_keys", [])
            unexpected = getattr(incompatible, "unexpected_keys", [])
            print(f"Loaded checkpoint. missing_keys={len(missing)}, unexpected_keys={len(unexpected)}")
        except Exception as e:
            print(f"Failed to load state_dict from checkpoint: {e}")
    
    load_ckpt(
        ckpt_path=Path("./outputs/stage1/train1/checkpoints/cfd_simformer_model.conditioner cp=latest model.th"),
        submodel=model.conditioner,
    )
    load_ckpt(
        ckpt_path=Path("./outputs/stage1/train1/checkpoints/cfd_simformer_model.decoder cp=latest model.th"),
        submodel=model.decoder,
    )
    load_ckpt(
        ckpt_path=Path("./outputs/stage1/train1/checkpoints/cfd_simformer_model.encoder cp=latest model.th"),
        submodel=model.encoder,
    )
    load_ckpt(
        ckpt_path=Path("./outputs/stage1/train1/checkpoints/cfd_simformer_model.latent cp=latest model.th"),
        submodel=model.latent,
    )
    model.to(device)

    # ============================
    # 从 test_rollout 取一批数据并执行 model.rollout
    # ============================
    from torch.utils.data import DataLoader

    # 与 CfdSimformerTrainer.dataset_mode 保持一致
    dataset_mode = "x mesh_pos query_pos mesh_edges geometry2d timestep velocity target"

    rollout_dataset, rollout_collator = data_container.get_dataset("train_rollout", mode=dataset_mode)

    # 使用与训练中类似的 batch_size（YAML 中 max_num_sequences=32）
    rollout_loader = DataLoader(
        rollout_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=rollout_collator,
    )

    (batch_data, ctx) = next(iter(rollout_loader))
    x, mesh_pos, query_pos, mesh_edges, geometry2d, timestep, velocity, target = batch_data
    
    # 分辨率用 geometry2d 的 H, W
    resolution = (geometry2d.shape[1], geometry2d.shape[2])
    
    # 查询所有位置
    # query_pos = torch.meshgrid(
    #     torch.linspace(0, 1, steps=resolution[0]),
    #     torch.linspace(0, 1, steps=resolution[1]),
    #     indexing="ij",
    # )
    # query_pos = torch.stack(query_pos, dim=-1).view(1, -1, 2)  # (1, H*W, 2)
    # # 只保留 geometry2d 为 1 的查询位置
    # geom = geometry2d[0]
    # mask = geom == 0
    # mask = mask.reshape(-1)  # (H*W,)
    # query_pos = query_pos[:, mask, :]  # (1, N_valid, 2)
    # ctx["unbatch_idx"] = torch.zeros(query_pos.shape[1], dtype=torch.long)  # 全部属于同一 batch

    # 打印各个张量的 shape，和你给出的示例进行对比
    print("\n=== rollout batch (from test_rollout) ===")
    print("raw x.shape        ", x.shape)
    print("geometry2d.shape   ", geometry2d.shape)
    print("timestep.shape     ", timestep.shape)
    print("velocity.shape     ", velocity.shape)
    print("mesh_pos.shape     ", mesh_pos.shape)
    print("query_pos.shape    ", query_pos.shape)
    print("mesh_edges shape   ", None if mesh_edges is None else mesh_edges.shape)
    print("batch_idx.shape    ", ctx["batch_idx"].shape)
    print("unbatch_idx.shape  ", ctx["unbatch_idx"].shape)
    print("unbatch_select.shape", ctx["unbatch_select"].shape)

    # 将所有需要的张量搬到与模型相同的 device
    x = x.to(device)
    geometry2d = geometry2d.to(device)
    velocity = velocity.to(device)
    mesh_pos = mesh_pos.to(device)
    query_pos = query_pos.to(device)
    if mesh_edges is not None:
        mesh_edges = mesh_edges.to(device)
    batch_idx = ctx["batch_idx"].to(device)
    unbatch_idx = ctx["unbatch_idx"].to(device)
    unbatch_select = ctx["unbatch_select"].to(device)

    # roll-out 需要的输入 x 维度应与 model.input_shape 对齐：
    #   (num_points, num_input_timesteps * num_channels)
    # test_rollout 的 x 只包含 t0 的通道 -> 重复到所需的历史长度
    _, input_dim = model.input_shape
    num_channels = x.size(1)
    assert input_dim % num_channels == 0, f"input_dim={input_dim} 不能被 num_channels={num_channels} 整除"
    num_input_timesteps = input_dim // num_channels
    x0 = einops.repeat(x, "n c -> n (t c)", t=num_input_timesteps)

    print("\n=== x0 for rollout ===")
    print("x0.shape          ", x0.shape)

    model.eval()
    with torch.no_grad():
        preds = model.rollout(
            x=x0,
            geometry2d=geometry2d,
            velocity=velocity,
            mesh_pos=mesh_pos,
            query_pos=query_pos,
            mesh_edges=mesh_edges,
            batch_idx=batch_idx,
            unbatch_idx=unbatch_idx,
            unbatch_select=unbatch_select,
            mode="latent",
        )

    print("\n=== rollout output ===")
    print("preds.shape        ", preds.shape)

    # ============================
    # 只用 preds 做灰度 GIF，可视化 rollout
    # ============================

    # preds: (total_num_points, num_channels, num_rollout_timesteps)
    num_points, num_channels, num_rollout_timesteps = preds.shape

    # 反归一化到物理空间（当前 norm=none 等价于原值，这里保持一致）
    rollout_root_dataset = data_container.get_dataset("test_rollout")
    preds_denorm = rollout_root_dataset.denormalize(preds.clone().cpu(), inplace=False)

    # 使用除最后一维以外的通道计算“强度”（例如速度模长），形状: (total_num_points, num_rollout_timesteps)
    preds_intensity = preds_denorm[:, :-1, :].norm(dim=1)

    # 位置使用 query_pos（单 batch），形状 (num_points, 2)
    assert query_pos.dim() == 3 and query_pos.size(0) == 1
    pos = query_pos[0].cpu()

    import io

    def _tensor_to_pil(values_1xn, progress: float, pos_2d):
        """将 [num_points] 的数据和坐标转换成一张灰度 PIL 图像。"""
        # values_1xn: (num_points,)
        img = coords_to_image(
            coords=pos_2d,
            resolution=resolution,
            weights=values_1xn,
        )  # (H, W) float 或 (C, H, W)

        # 如果返回带通道维的张量 (C, H, W)，先在通道维上平均，得到 2D 灰度图
        if img.dim() == 3:
            img = img.mean(dim=0)

        # 归一化到 [0, 1]
        img_min = img.min()
        img = img - img_min
        img_max = img.max()
        img = img / (img_max + 1e-6)

        # 顶部进度条
        h, w = img.shape
        progress_row = torch.zeros((1, w), dtype=img.dtype)
        progress_row[:, : round(progress * w)] = 1
        img = torch.cat([progress_row, img], dim=0)  # (H+1, W)

        # 转为 [0,255] 的 uint8 并生成灰度图
        arr = (img.clamp(0, 1).numpy() * 255.0).astype("uint8")
        pil = Image.fromarray(arr, mode="L")
        return pil

    imgs = []
    for t in range(num_rollout_timesteps):
        frame_vals = preds_intensity[:, t]  # (num_points,)
        img = _tensor_to_pil(
            values_1xn=frame_vals,
            progress=t / max(1, (num_rollout_timesteps - 1)),
            pos_2d=pos,
        )
        imgs.append(img)

    out_dir = path_provider.stage_output_path / "rollout"
    out_dir.mkdir(parents=True, exist_ok=True)
    gif_uri = out_dir / "rollout_debug.gif"
    imgs[0].save(
        fp=str(gif_uri),
        format="GIF",
        append_images=imgs[1:],
        save_all=True,
        duration=100,
        loop=0,
    )
    print(f"saved rollout gif to: {gif_uri}")


if __name__ == "__main__":
    main()
