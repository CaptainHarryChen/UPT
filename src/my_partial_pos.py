import torch
from torch_geometric.nn.pool import radius_graph
from torch.utils.data import DataLoader
from pathlib import Path

from PIL import Image
import einops

from kappautils.images.points_to_image import coords_to_image

from datasets import dataset_from_kwargs
from models import model_from_kwargs
from providers.dataset_config_provider import DatasetConfigProvider
from providers.path_provider import PathProvider
from utils.data_container import DataContainer
from utils.kappaconfig.util import get_stage_hp


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _tensor_to_pil(gt_vals, pred_vals, diff_vals, pos_2d, resolution, input_pos_2d=None):
    """将 GT / preds / 差值 三个 [num_points] 序列转换成一张垂直拼接的彩色 PIL 图像。

    - 对每个标量场先根据 pos_2d 投影到 2D 图像上
    - 分别做 [0,1] 归一化
    - 使用 jet 风格的伪彩色
    """

    def _make_img(values_1xn):
        img = coords_to_image(
            coords=pos_2d,
            resolution=resolution,
            weights=values_1xn,
        )  # (H, W) 或 (C, H, W)
        if img.dim() == 3:
            img = img.mean(dim=0)
        return img

    def _to_color(img_2d):
        x = img_2d.clamp(0.0, 1.0)
        r = torch.clamp(1.5 - torch.abs(4 * x - 3), 0.0, 1.0)
        g = torch.clamp(1.5 - torch.abs(4 * x - 2), 0.0, 1.0)
        b = torch.clamp(1.5 - torch.abs(4 * x - 1), 0.0, 1.0)
        return torch.stack([r, g, b], dim=-1)  # (H, W, 3)

    # 先得到 2D 标量场
    img_gt_raw = _make_img(gt_vals)
    img_pred_raw = _make_img(pred_vals)
    img_diff_raw = _make_img(diff_vals)

    # 各自归一化到 [0,1]
    def _norm(img):
        mn = img.min()
        mx = img.max()
        denom = (mx - mn + 1e-6)
        return (img - mn) / denom

    shared = torch.stack([img_gt_raw, img_pred_raw], dim=0)
    shared_norm = _norm(shared)
    img_gt = shared_norm[0]
    img_pred = shared_norm[1]
    # img_gt = _norm(img_gt_raw)
    # img_pred = _norm(img_pred_raw)
    img_diff = _norm(img_diff_raw)

    col_gt = _to_color(img_gt)
    col_pred = _to_color(img_pred)
    col_diff = _to_color(img_diff)

    # 如果提供了输入点坐标，则在 GT 图上做高亮标记
    if input_pos_2d is not None:
        # 构建一个稀疏的 0/1 掩码，然后进行轻微膨胀后叠加在 GT 上
        num_inputs = input_pos_2d.shape[0]
        input_weights = torch.ones(num_inputs, dtype=img_gt.dtype)
        mask = coords_to_image(
            coords=input_pos_2d,
            resolution=resolution,
            weights=input_weights,
        )  # (H, W) 或 (C, H, W)
        if mask.dim() == 3:
            mask = mask.mean(dim=0)
        mask = (mask > 0).float()

        # 简单“膨胀”：将掩码与自己平移后求最大值，形成稍大的点
        def _dilate(m):
            pads = [(0, 0), (0, 0)]
            m_pad = torch.nn.functional.pad(m[None, None], (1, 1, 1, 1), mode="constant", value=0)[0, 0]
            res = m
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    res = torch.maximum(res, m_pad[1 + dy : 1 + dy + m.shape[0], 1 + dx : 1 + dx + m.shape[1]])
            return res

        mask = _dilate(mask)
        mask = mask.clamp(0.0, 1.0)

        # 将掩码区域染成偏红色（不会完全覆盖底图，保持可见）
        highlight = torch.stack([
            0.9 * torch.ones_like(mask),  # R 高
            0.2 * torch.ones_like(mask),  # G 低
            0.2 * torch.ones_like(mask),  # B 低
        ], dim=-1)  # (H, W, 3)

        alpha = 0.7 * mask[..., None]  # (H, W, 1)
        col_gt = col_gt * (1.0 - alpha) + highlight * alpha

    # 垂直拼接 GT | preds | diff
    img_cat = torch.cat([col_gt, col_pred, col_diff], dim=0)  # (3H, W, 3)

    arr = (img_cat.clamp(0, 1).numpy() * 255.0).astype("uint8")
    pil = Image.fromarray(arr, mode="RGB")
    return pil


def main():
    # 1. 读取 stage 配置并构造 DataContainer / 模型
    stage_hp = get_stage_hp(
        hp_file="./src/yamls/cfd/e100_lr5e5_8M_lat512_fp32.yaml",
        template_path="./src/zztemplates",
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

    for dataset_key, dataset_kwargs in stage_hp["datasets"].items():
        datasets_dict[dataset_key] = dataset_from_kwargs(
            dataset_config_provider=dataset_config_provider,
            path_provider=path_provider,
            **dataset_kwargs,
        )

    data_container = DataContainer(
        **datasets_dict,
        seed=182376,
    )

    model = model_from_kwargs(
        **stage_hp["model"],
        input_shape=(None, 6),
        output_shape=(None, 3),
        path_provider=path_provider,
        data_container=data_container,
    )

    def load_ckpt(ckpt_path: Path, submodel):
        print(f"\nLoading checkpoint from: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device)
        state_dict = checkpoint["state_dict"]
        try:
            incompatible = submodel.load_state_dict(state_dict, strict=False)
            missing = getattr(incompatible, "missing_keys", [])
            unexpected = getattr(incompatible, "unexpected_keys", [])
            print(f"Loaded checkpoint. missing_keys={len(missing)}, unexpected_keys={len(unexpected)}")
        except Exception as e:
            print(f"Failed to load state_dict from checkpoint: {e}")

    # 根据你的训练结果修改这些 checkpoint 路径
    load_ckpt(
        ckpt_path=Path(
            "./outputs/stage1/train3/checkpoints/cfd_simformer_model.conditioner cp=latest model.th"
        ),
        submodel=model.conditioner,
    )
    load_ckpt(
        ckpt_path=Path(
            "./outputs/stage1/train3/checkpoints/cfd_simformer_model.decoder cp=latest model.th"
        ),
        submodel=model.decoder,
    )
    load_ckpt(
        ckpt_path=Path(
            "./outputs/stage1/train3/checkpoints/cfd_simformer_model.encoder cp=latest model.th"
        ),
        submodel=model.encoder,
    )
    load_ckpt(
        ckpt_path=Path(
            "./outputs/stage1/train3/checkpoints/cfd_simformer_model.latent cp=latest model.th"
        ),
        submodel=model.latent,
    )
    model.to(device)

    # 2. 从 train 数据集取一个 batch
    dataset_mode = "x mesh_pos query_pos mesh_edges geometry2d timestep velocity target"

    train_dataset, train_collator = data_container.get_dataset("train", mode=dataset_mode)

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=train_collator,
    )

    idx = 50
    it = iter(train_loader)
    for i in range(idx):
        (batch_data, ctx) = next(it)
    x, mesh_pos, query_pos, mesh_edges, geometry2d, timestep, velocity, target = batch_data

    print("\n=== original batch (train) ===")
    print("x.shape         ", x.shape)
    print("mesh_pos.shape  ", mesh_pos.shape)
    print("query_pos.shape ", query_pos.shape)
    print("target.shape    ", target.shape)

    # 几何分辨率（H, W），用 geometry2d 的最后两个维度
    resolution = (geometry2d.shape[-2], geometry2d.shape[-1])

    # 3. 在 mesh_pos 上随机采样一部分点作为输入
    num_points = mesh_pos.shape[0]
    sample_ratio = 0.05  # 随机保留 10% 的输入点，可以自行调节
    num_keep = max(1, int(num_points * sample_ratio))

    perm = torch.randperm(num_points)
    keep_idx = perm[:num_keep]
    keep_idx, _ = torch.sort(keep_idx)

    mesh_pos_sub = mesh_pos[keep_idx]
    x_sub = x[keep_idx]
    batch_idx = ctx["batch_idx"]
    batch_idx_sub = batch_idx[keep_idx]

    print("\n=== subsampled input ===")
    print("num_points       ", num_points)
    print("num_keep         ", num_keep)
    print("mesh_pos_sub.shape", mesh_pos_sub.shape)
    print("x_sub.shape      ", x_sub.shape)

    # 4. 基于子集重新构建 mesh_edges（始终按子集重建，忽略原始 mesh_edges）
    radius_graph_r = stage_hp["trainer"]["radius_graph_r"]
    radius_graph_max_num_neighbors = stage_hp["trainer"]["radius_graph_max_num_neighbors"]

    mesh_edges_sub = radius_graph(
        x=mesh_pos_sub,
        r=radius_graph_r,
        max_num_neighbors=radius_graph_max_num_neighbors,
        batch=batch_idx_sub,
        loop=True,
        flow="source_to_target",
    ).T

    print("mesh_edges_sub.shape", mesh_edges_sub.shape)

    # 5. 将张量搬到 device，并直接调用 model.forward（单步，不做 rollout）
    x_sub = x_sub.to(device)
    mesh_pos_sub = mesh_pos_sub.to(device)
    mesh_edges_sub = mesh_edges_sub.to(device)

    geometry2d = geometry2d.to(device)
    timestep = timestep.to(device)
    velocity = velocity.to(device)
    query_pos = query_pos.to(device)
    target = target.to(device)

    unbatch_idx = ctx["unbatch_idx"].to(device)
    unbatch_select = ctx["unbatch_select"].to(device)

    # 确认输入维度和模型预期一致
    _, input_dim = model.input_shape
    assert x_sub.shape[1] == input_dim, f"x_sub.shape[1]={x_sub.shape[1]} 与 model.input_shape[1]={input_dim} 不一致"

    model.eval()
    with torch.no_grad():
        outputs = model(
            x_sub,
            geometry2d=geometry2d,
            timestep=timestep,
            velocity=velocity,
            mesh_pos=mesh_pos_sub,
            query_pos=query_pos,
            mesh_edges=mesh_edges_sub,
            batch_idx=batch_idx_sub.to(device),
            unbatch_idx=unbatch_idx,
            unbatch_select=unbatch_select,
        )

    x_hat = outputs["x_hat"]  # (num_query_points_total, C)
    print("\n=== forward output ===")
    print("x_hat.shape      ", x_hat.shape)

    # 6. 在物理空间下做简单的误差统计并画一张图
    preds_denorm = train_dataset.denormalize(x_hat.detach().cpu(), inplace=False)
    target_denorm = train_dataset.denormalize(target.detach().cpu(), inplace=False)

    # 只用除第 0 通道以外的通道做“强度”可视化（例如速度模长）
    preds_intensity = preds_denorm[:, 1:].norm(dim=1)
    target_intensity = target_denorm[:, 1:].norm(dim=1)
    diff_intensity = (preds_intensity - target_intensity).abs()

    # 位置使用 query_pos（单 batch），形状 (num_query_points, 2)
    assert query_pos.dim() == 3 and query_pos.size(0) == 1
    pos = query_pos[0].detach().cpu()

    # 输入点位置使用 mesh_pos_sub（单 batch 的子集），形状 (num_keep, 2)
    input_pos_2d = mesh_pos_sub.detach().cpu()

    img = _tensor_to_pil(
        gt_vals=target_intensity,
        pred_vals=preds_intensity,
        diff_vals=diff_intensity,
        pos_2d=pos,
        resolution=resolution,
        input_pos_2d=input_pos_2d,
    )

    out_dir = path_provider.stage_output_path / "partial_pos"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "partial_pos_debug.png"
    img.save(str(out_path))
    print(f"saved visualization to: {out_path}")


if __name__ == "__main__":
    main()
