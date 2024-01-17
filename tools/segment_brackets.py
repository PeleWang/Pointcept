"""
This script offers a method that segments brackets from a cropped tooth mesh/point cloud.
Original author: Xiaoyang Wu
Modified by: Chuanbo Wang
"""
import os, sys
cwd = os.getcwd()
sys.path.append(cwd)
import open3d
import numpy as np
import argparse
import collections

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.nn.functional as F

from pointcept.models import build_model
from pointcept.utils.config import Config, DictAction
from pointcept.utils.logger import get_root_logger
from pointcept.utils.env import get_random_seed, set_seed
from pointcept.datasets.transform import Compose


def segment_bracket(mesh):
    """
    Segments the bracket in the input mesh, returns a point cloud with
    bracket points labeled in red, other points labeled in black.
    Args:
        mesh (open3d.geometry.TriangleMesh/PointCloud): The input mesh/point cloud.
    Raises:
        FileExistsError: if the weight file does not exist.
        TypeError: if the input data is not a open3d mesh/point cloud.
    Returns:
        pred_pc: the return point cloud.
    """
    parser = argparse.ArgumentParser(description='Pointcept Test Process')
    parser.add_argument('--config-file', default="exp/default/Sat Jun  3 08:25:53 2023/config.py", \
                        metavar="FILE", help="path to config file")
    parser.add_argument('--options', nargs='+', action=DictAction, help='custom options')
    args = parser.parse_args()

    # config_parser
    cfg = Config.fromfile(args.config_file)
    if args.options is not None:
        cfg.merge_from_dict(args.options)

    if cfg.seed is None:
        cfg.seed = get_random_seed()

    os.makedirs(cfg.save_path, exist_ok=True)

    # default_setup
    set_seed(cfg.seed)
    cfg.batch_size_val_per_gpu = cfg.batch_size_test  # TODO: add support to multi gpu test
    cfg.num_worker_per_gpu = cfg.num_worker  # TODO: add support to multi gpu test

    # tester init
    weight_name = cfg.save_path.split('/')[-1]
    logger = get_root_logger(log_file=os.path.join(cfg.save_path, "test-{}.log".format(weight_name)))
    logger.info("=> Loading config ...")
    logger.info(f"Save path: {cfg.save_path}")
    logger.info(f"Config:\n{cfg.pretty_text}")

    # build model
    logger.info("=> Building model ...")
    model = build_model(cfg.model).cuda()
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Num params: {n_parameters}")

    # load checkpoint
    weight_path = os.path.join(cfg.save_path, 'model/model_last.pth')
    if not os.path.isfile(weight_path):
        raise FileExistsError("weight file does not exist")
    checkpoint = torch.load(weight_path)
    state_dict = checkpoint['state_dict']
    new_state_dict = collections.OrderedDict()
    for name, value in state_dict.items():
        if name.startswith("module."):
            name = name[7:]  # module.xxx.xxx -> xxx.xxx
        new_state_dict[name] = value
    model.load_state_dict(new_state_dict, strict=True)
    model.eval()

    # put input mesh/point cloud
    input_dict = {}
    if isinstance(mesh, open3d.geometry.TriangleMesh):
        pc = mesh.sample_points_poisson_disk(number_of_points=20000, use_triangle_normal=True)
    elif isinstance(mesh, open3d.geometry.PointCloud):
        pc = mesh
    else:
        raise TypeError("The input data must be triangle mesh or point cloud")
    num_pts = np.asarray(pc.points).shape[0]
    input_dict['coord'] = np.asarray(pc.points).astype(np.float32)
    input_dict['normal'] = np.asarray(pc.normals).astype(np.float32)
    input_dict['segment'] = np.zeros((num_pts, 1)).astype(np.int64)
    input_dict['offset'] = np.array([num_pts]).astype(np.int64)

    transform = cfg.data.test.transform
    transform = Compose(transform)
    input_dict = transform(input_dict)
    
    input_dict['coord'] = torch.from_numpy(input_dict['coord'])
    input_dict['normal'] = torch.from_numpy(input_dict['normal'])
    input_dict['segment'] = torch.from_numpy(input_dict['segment'])
    input_dict['offset'] = torch.from_numpy(input_dict['offset'])
    input_dict['feat'] = torch.from_numpy(np.concatenate((input_dict['coord'], input_dict['normal']), 1))
    
    for key in input_dict.keys():
        if isinstance(input_dict[key], torch.Tensor):
            input_dict[key] = input_dict[key].cuda(non_blocking=True)

    pred = torch.zeros((num_pts, cfg.data.num_classes)).cuda()
    with torch.no_grad():
        pred_part = model(input_dict)  # (n, k)
        pred_part = pred_part["seg_logits"]
        pred_part = F.softmax(pred_part, -1)
        if cfg.empty_cache:
            torch.cuda.empty_cache()
        bs = 0
        for be in input_dict["offset"]:
            be = be.item()
            pred_part_label = torch.unsqueeze(torch.argmax(pred_part, axis=1), 1)
            pred[bs: be] += pred_part_label
            bs = be
                
        # pred = pred.max(1)[1].data.cpu().numpy()
        pred = pred_part_label.data.cpu().numpy()
        xyzs = input_dict['coord'].data.cpu().numpy()
        if 'shift' in input_dict.keys():
            xyzs += input_dict["shift"]
        # save the pred and label as ply point clouds
        pred_pc = open3d.geometry.PointCloud(open3d.utility.Vector3dVector(xyzs))
        pred_pc_colors = np.concatenate((pred, np.zeros((pred.shape[0], 2))), axis=1)
        pred_pc.paint_uniform_color([0,0,0])
        pred_pc.colors = open3d.utility.Vector3dVector(pred_pc_colors)
        return pred_pc


if __name__ == '__main__':
    input = open3d.io.read_point_cloud('859a-4168_LL_LowerJaw_1676306904649__9.ply')
    pred_pc = segment_bracket(input)
    open3d.io.write_point_cloud('./859a-4168_LL_LowerJaw_1676306904649__9_seg.ply', pred_pc)
