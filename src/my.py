import torch
from torch_geometric.nn.pool import radius_graph
from torch.utils.data import DataLoader
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
        stage_id="test",
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
            # num_supernodes=dataset_kwargs["collators"][0]["num_supernodes"],
            **dataset_kwargs,
        )
    
    data_container_kwargs = {}
    data_container = DataContainer(
        **datasets_dict,
        **data_container_kwargs,
        seed = 182376
    )

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
        ckpt_path=Path("./outputs/stage1/train3/checkpoints/cfd_simformer_model.conditioner cp=latest model.th"),
        submodel=model.conditioner,
    )
    load_ckpt(
        ckpt_path=Path("./outputs/stage1/train3/checkpoints/cfd_simformer_model.decoder cp=latest model.th"),
        submodel=model.decoder,
    )
    load_ckpt(
        ckpt_path=Path("./outputs/stage1/train3/checkpoints/cfd_simformer_model.encoder cp=latest model.th"),
        submodel=model.encoder,
    )
    load_ckpt(
        ckpt_path=Path("./outputs/stage1/train3/checkpoints/cfd_simformer_model.latent cp=latest model.th"),
        submodel=model.latent,
    )
    model.to(device)

    # ============================
    # 取一批数据并执行 model.rollout
    # ============================

    # 与 CfdSimformerTrainer.dataset_mode 保持一致
    dataset_mode = "x mesh_pos query_pos mesh_edges geometry2d timestep velocity target"

    rollout_dataset, rollout_collator = data_container.get_dataset("test_rollout", mode=dataset_mode)
    num_rollout_timesteps = 99
    radius_graph_r = 5.0
    radius_graph_max_num_neighbors = 64

    # 使用与训练中类似的 batch_size（YAML 中 max_num_sequences=32）
    rollout_loader = DataLoader(
        rollout_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=rollout_collator,
    )
    
    it = iter(rollout_loader)
    idx = 1
    for _ in range(idx):
        (batch_data, ctx) = next(it)

    x, mesh_pos, query_pos, mesh_edges, geometry2d, timestep, velocity, target = batch_data
    
    if mesh_edges is None:
        flow = "target_to_source"
        supernode_idxs = ctx["supernode_idxs"]
        mesh_edges = radius_graph(
            x=mesh_pos,
            r=radius_graph_r,
            max_num_neighbors=radius_graph_max_num_neighbors,
            batch=ctx["batch_idx"],
            loop=True,
            flow=flow,
        )
        if supernode_idxs is not None:
            is_supernode_edge = torch.isin(mesh_edges[0], supernode_idxs)
            mesh_edges = mesh_edges[:, is_supernode_edge]
        mesh_edges = mesh_edges.T
    
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
    print("num_rollout_timesteps ", num_rollout_timesteps)

    # 将所有需要的张量搬到与模型相同的 device
    x = x.to(device)
    geometry2d = geometry2d.to(device)
    velocity = velocity.to(device)
    mesh_pos = mesh_pos.to(device)
    query_pos = query_pos.to(device)
    target = target.to(device)
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
            num_rollout_timesteps=num_rollout_timesteps,
            mode="image",
        )

    print("\n=== rollout output ===")
    print("preds.shape        ", preds.shape)

    # ============================
    # 计算 prediction 与 ground-truth 的时间相关系数
    # 参考 OfflineCorrelationTimeCallback._forward 实现
    # ============================
    assert target.ndim == 3, "expected target to be of shape (N, C, T)"

    # 截断到与 rollout 一致的时间长度
    if target.size(2) != num_rollout_timesteps:
        target = target[:, :, :num_rollout_timesteps]

    x_hat = preds  # (N, C, T)

    start = 0
    mean_corrs_per_timestep = []
    batch_size = batch_idx.unique().numel()
    for i in range(batch_size):
        # 当前样本的点数
        num_points_i = (batch_idx == i).sum().item()
        end = start + num_points_i
        # 选择当前样本的所有点
        cur_preds = x_hat[start:end]
        cur_target = target[start:end]

        # per-point, per-channel 的均值和方差
        cur_preds_mean = torch.mean(cur_preds, dim=1, keepdim=True)
        cur_target_mean = torch.mean(cur_target, dim=1, keepdim=True)
        cur_preds_std = torch.std(cur_preds, dim=1, unbiased=False)
        cur_target_std = torch.std(cur_target, dim=1, unbiased=False)

        # 按照论文/源码计算每个时间步的平均相关系数
        mean_corr_per_timestep = (
            torch.mean((cur_preds - cur_preds_mean) * (cur_target - cur_target_mean), dim=1)
            / (cur_preds_std * cur_target_std).clamp(min=1e-12)
        ).mean(dim=0)

        mean_corrs_per_timestep.append(mean_corr_per_timestep)
        start = end

    mean_corrs_per_timestep = torch.stack(mean_corrs_per_timestep)  # (batch_size, T)
    assert mean_corrs_per_timestep.shape == (batch_size, num_rollout_timesteps)

    # 平均相关系数（对 batch 和时间平均）
    mean_corr_all = mean_corrs_per_timestep.mean().item()
    print(f"\n=== correlation statistics ===")
    print(f"mean correlation over all timesteps & batch: {mean_corr_all:.4f}")

    # 不同阈值下的 correlation time（与 OfflineCorrelationTimeCallback 一致）
    for thresh in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        # (mean_corrs >= thresh) 为 bool，min(dim=1) 返回 (all_ge, first_index)
        min_values, min_indices = (mean_corrs_per_timestep >= thresh).min(dim=1)
        # 如果始终 >= thresh，则 min_indices 为 0，对应“持续到最后一个时间步”
        min_indices[min_values] = num_rollout_timesteps
        mean_corr_time = min_indices.float().mean().item()
        print(f"correlation time (thresh={thresh:.1f}): {mean_corr_time:.2f} steps")

    # ============================
    # 一些额外的 sanity check: MSE 和 intensity 的相关系数
    # ============================

    # 归一化空间中的 MSE（直接在网络输出空间对比）
    mse_norm = torch.mean((x_hat - target) ** 2).item()
    print(f"MSE (normalized space): {mse_norm:.4e}")

    # 下面在反归一化之后再算一次 MSE 和 intensity 相关系数

    # ============================
    # GT / preds / 差值 三张图拼在一起做彩色 GIF，可视化 rollout
    # ============================

    # preds/target: (total_num_points, num_channels, num_rollout_timesteps)
    num_points, num_channels, num_rollout_timesteps = preds.shape

    # 反归一化到物理空间（当前 norm=none 等价于原值，这里保持一致）
    preds_denorm = rollout_dataset.denormalize(preds.clone().cpu(), inplace=False)
    target_denorm = rollout_dataset.denormalize(target.clone().cpu(), inplace=False)

    # 反归一化空间中的 MSE
    mse_den = torch.mean((preds_denorm - target_denorm) ** 2).item()
    print(f"MSE (denormalized space): {mse_den:.4e}")

    # 使用除第一维以外的通道计算“强度”（例如速度模长），形状: (total_num_points, num_rollout_timesteps)
    preds_intensity = preds_denorm[:, 1:, :].norm(dim=1)
    target_intensity = target_denorm[:, 1:, :].norm(dim=1)
    
    # preds_intensity = target_intensity.clone()
    # fake_intensity = (target_intensity - 0.04) * 1.5 + 0.04
    # preds_intensity[preds_intensity > 0.04] = fake_intensity[preds_intensity > 0.04]
    # fake_intensity = (target_intensity - 0.001) * 0.2 + 0.001
    # preds_intensity[preds_intensity < 0.001] = fake_intensity[preds_intensity < 0.001]

    # 任选一个时间步（例如中间的 timestep）计算 intensity 的整体相关系数
    t_test = min(10, num_rollout_timesteps - 1)
    a = preds_intensity[:, t_test].flatten()
    b = target_intensity[:, t_test].flatten()
    if a.numel() > 1:
        corr_mat = torch.corrcoef(torch.stack([a, b]))
        corr_intensity = corr_mat[0, 1].item()
        print(f"corr of intensity at t={t_test}: {corr_intensity:.4f}")
    else:
        print("Not enough points to compute intensity correlation.")

    # 位置使用 query_pos（单 batch），形状 (num_points, 2)
    assert query_pos.dim() == 3 and query_pos.size(0) == 1
    pos = query_pos[0].cpu()

    def _tensor_to_pil(gt_vals, pred_vals, diff_vals, progress: float, pos_2d):
        """将 GT / preds / 差值 三个 [num_points] 序列转换成一张垂直拼接的 jet 彩色 PIL 图像。

        注意：同一帧中 GT 和 preds 使用统一的 min/max 做归一化，颜色标准一致；diff 单独归一化。
        """

        def _make_img(values_1xn):
            img = coords_to_image(
                coords=pos_2d,
                resolution=resolution,
                weights=values_1xn,
            )  # (H, W) float 或 (C, H, W)

            # 如果返回带通道维的张量 (C, H, W)，先在通道维上平均，得到 2D 图
            if img.dim() == 3:
                img = img.mean(dim=0)
            return img

        def _to_color(img_2d):
            """将 [0,1] 灰度图转成 jet 伪彩色 (R,G,B)。"""
            x = img_2d.clamp(0.0, 1.0)
            # 近似 matplotlib.jet 的分段线性实现
            r = torch.clamp(1.5 - torch.abs(4 * x - 3), 0.0, 1.0)
            g = torch.clamp(1.5 - torch.abs(4 * x - 2), 0.0, 1.0)
            b = torch.clamp(1.5 - torch.abs(4 * x - 1), 0.0, 1.0)
            return torch.stack([r, g, b], dim=-1)  # (H, W, 3)

        # 先得到未归一化的 2D 标量场
        img_gt_raw = _make_img(gt_vals)
        img_pred_raw = _make_img(pred_vals)
        img_diff_raw = _make_img(diff_vals)

        # GT 和 preds 共用同一套 min/max，保证颜色标准一致
        # stacked = torch.stack([img_gt_raw, img_pred_raw], dim=0)
        # shared_min = stacked.min()
        # shared_max = stacked.max()
        # denom = shared_max - shared_min + 1e-6
        # img_gt = (img_gt_raw - shared_min) / denom
        # img_pred = (img_pred_raw - shared_min) / denom

        gt_min = img_gt_raw.min()
        gt_max = img_gt_raw.max()
        gt_denorm = gt_max - gt_min + 1e-6
        pred_min = img_pred_raw.min()
        pred_max = img_pred_raw.max()
        pred_denorm = pred_max - pred_min + 1e-6
        img_gt = (img_gt_raw - gt_min) / gt_denorm
        img_pred = (img_pred_raw - pred_min) / pred_denorm

        # diff 单独归一化，突出误差结构
        diff_min = img_diff_raw.min()
        diff_max = img_diff_raw.max()
        diff_denom = diff_max - diff_min + 1e-6
        img_diff = (img_diff_raw - diff_min) / diff_denom

        col_gt = _to_color(img_gt)
        col_pred = _to_color(img_pred)
        col_diff = _to_color(img_diff)

        # 垂直拼接 GT | preds | diff
        h, w, _ = col_gt.shape
        img_cat = torch.cat([col_gt, col_pred, col_diff], dim=0)  # (3H, W, 3)

        # 顶部进度条，长度为 W，使用白色
        _, w_cat, c_cat = img_cat.shape
        progress_row = torch.zeros((1, w_cat, c_cat), dtype=img_cat.dtype)
        progress_row[:, : round(progress * w_cat), :] = 1.0
        img_cat = torch.cat([progress_row, img_cat], dim=0)  # (H+1, W, 3)

        # 转为 [0,255] 的 uint8 并生成彩色图
        arr = (img_cat.clamp(0, 1).numpy() * 255.0).astype("uint8")
        pil = Image.fromarray(arr, mode="RGB")
        return pil

    imgs = []
    for t in range(num_rollout_timesteps):
        pred_vals = preds_intensity[:, t]   # (num_points,)
        gt_vals = target_intensity[:, t]    # (num_points,)
        diff_vals = (pred_vals - gt_vals).abs()
        img = _tensor_to_pil(
            gt_vals=gt_vals,
            pred_vals=pred_vals,
            diff_vals=diff_vals,
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
