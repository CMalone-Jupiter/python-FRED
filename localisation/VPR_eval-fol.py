import os
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
import sys
sys.path.append(os.path.abspath("."))   # one level up
import numpy as np
# import cv2
# import open3d as o3d
# from scipy.spatial.transform import Rotation
# from utils.lidar import PointCloud
# from utils.camera import ImageData
# import utils.utils as utils
from utils.utils import get_all_corr_files
from FoL.reranking import run_rerank
from natsort import natsorted, index_natsorted
import torch
from tqdm import tqdm
from glob import glob
from math import floor

################## set device based on cuda availability #################
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print('CUDA availability: ' + str(torch.cuda.is_available()))

####################### Functions for matching using numpy on CPU or Pytorch on GPU ###################
def getMatchIndsCPU(ft_ref,ft_qry,topK=20,metric='cosine'):
    """
    metric: 'euclidean' or 'cosine'
    """
    # dMat = cdist(ft_ref,ft_qry,metric)

    ft_qry_norm = ft_qry / np.linalg.norm(ft_qry, axis=1, keepdims=True)  # Shape (M, N)
    ft_ref_norm = ft_ref / np.linalg.norm(ft_ref, axis=1, keepdims=True)  # Shape (C, N)

    # Step 2: Compute cosine similarity
    dMat = 1 - (ft_ref_norm @ ft_qry_norm.T)
    mInds = np.argsort(dMat,axis=0)[:topK].squeeze() # shape: K x ft_qry.shape[0]
    return mInds, dMat


def getMatchIndsGPU(ft_ref, ft_qry,topK=20, metric='cosine'):
    # metric: 'euclidean' or 'cosine'
    ft_qry_tensor = torch.Tensor(ft_qry).to(device)
    ft_ref_tensor = torch.Tensor(ft_ref).to(device)

    if metric == 'euclidean':
        # Use torch's cdist for Euclidean distance
        dMat = torch.cdist(ft_ref, ft_qry)
    
    elif metric == 'cosine':
        # # Normalize both the query and reference tensors
        ft_qry_norm = ft_qry_tensor / ft_qry_tensor.norm(dim=1, keepdim=True)
        ft_ref_norm = ft_ref_tensor / ft_ref_tensor.norm(dim=1, keepdim=True)
        # Compute cosine similarity (1 - cosine similarity for distance)
        dMat = 1 - ft_ref_norm @ ft_qry_norm.t()

    # Get the indices of the top 5 closest matches
    mInds = torch.argsort(dMat.cpu(), dim=0)[:topK].squeeze()
    
    return mInds, dMat

qry_sets = [
    '20210909_124816_v2',
]

ref_sets = [
    '20230509_115540_v2',
]

vpr_descs = [
    'FoL',
]


img_calib_file = f"./camera_calib.txt"

dist_tolerance = 10 # metres
# qry_idx = 4
slice_len = 1000

# User parameters
location = 'dalby-to-brigalow'

################ Reference filenames and directories #################################
ref_condition = ''
ref_camera_pos = 'front'

ref_timestamps = []
ref_utms = []
ref_img_filenames = []
ref_utm_filenames = []

for ref_set in ref_sets:
    print(f"Loading {ref_set}")
    
    ref_root_directory = f"../../Datasets/dalby/{location}"
    ref_vpr_root = f"../../Datasets/dalby/{location}/vpr_ftrs/"
    ref_image_dir = f"{ref_root_directory}/{ref_set}/{ref_camera_pos}-imgs/"
    ref_utm_dir = f"{ref_root_directory}/{ref_set}/utm/"


    this_ref_timestamp = [filename.split('.png')[0] for filename in natsorted(os.listdir(ref_image_dir)) if os.path.isfile(ref_image_dir+filename)]
    ref_utms = ref_utms+[np.loadtxt(ref_utm_dir+filename) for filename in natsorted(os.listdir(ref_utm_dir)) if os.path.isfile(ref_utm_dir+filename)][55::]
    ref_img_filenames = [filename for filename in natsorted(os.listdir(ref_image_dir)) if os.path.isfile(ref_image_dir+filename)]
    ref_utm_filenames = np.array([filename for filename in natsorted(os.listdir(ref_utm_dir)) if os.path.isfile(ref_utm_dir+filename)])[:len(os.listdir(ref_utm_dir))-55]
    ref_timestamps = ref_timestamps+this_ref_timestamp

ref_utms = np.array(ref_utms)

for vpr_desc in vpr_descs:

    all_results = []

    first = True

    print(f"Loading references")

    for ref_set in ref_sets:
        print(f"Loading {ref_set} {vpr_desc} descriptors")
        ref_root_directory = f"../../Datasets/dalby/{location}"
        ref_vpr_root = f"../../Datasets/dalby/{location}/vpr_ftrs/"

        ref_image_dir = f"{ref_root_directory}/{ref_set}/{ref_camera_pos}-imgs/"

        # ref_name_sort_idx = index_natsorted(os.listdir(ref_image_dir))

        if slice_len is None:

            # Get the two orderings
            glob_sorted_paths = sorted(glob(f"{ref_image_dir}/*.png"))
            glob_sorted_filenames = [os.path.basename(p) for p in glob_sorted_paths]

            # Get the indices that would sort glob_sorted_filenames into natsorted order
            ref_name_sort_idx = index_natsorted(glob_sorted_filenames)

            ref_ftr = np.load(f"{ref_vpr_root}/{ref_set}/{vpr_desc}/queries_descriptors.npy")
            ref_local_ftr = np.load(f"{ref_vpr_root}/{ref_set}/{vpr_desc}/qry_local_feats.npy")
            if first:
                ref_ftrs = ref_ftr[ref_name_sort_idx]
                ref_local_ftrs = ref_local_ftr[ref_name_sort_idx]
                first = False
            else:
                ref_ftrs = np.vstack((ref_ftrs, ref_ftr[ref_name_sort_idx]))
                ref_local_ftrs = np.vstack((ref_local_ftrs, ref_local_ftr[ref_name_sort_idx]))
        
        else:
            num_slices = floor(len(ref_img_filenames)/slice_len)
            if len(ref_img_filenames) % slice_len > 0:
                num_slices += 1

            for idx in tqdm(range(num_slices)):
                if idx == 0:
                    ref_ftrs = np.load(f"{ref_vpr_root}/{ref_set}/{vpr_desc}/sliced/queries_descriptors_slice_{idx:05d}.npy")
                    ref_local_ftrs = np.load(f"{ref_vpr_root}/{ref_set}/{vpr_desc}/sliced/qry_local_feats_slice_{idx:05d}.npy")
                else:
                    ref_ftrs = np.vstack((ref_ftrs, np.load(f"{ref_vpr_root}/{ref_set}/{vpr_desc}/sliced/queries_descriptors_slice_{idx:05d}.npy")))
                    ref_local_ftrs = np.vstack((ref_local_ftrs, np.load(f"{ref_vpr_root}/{ref_set}/{vpr_desc}/sliced/qry_local_feats_slice_{idx:05d}.npy")))

            
            print(f"Loaded ref ftr slices: {len(ref_ftrs)}")
            print(f"Loaded ref local ftr slices: {len(ref_local_ftrs)}")


    for qry_set in qry_sets:

        ################ Query filenames and directories #################################
        qry_condition = ''
        qry_camera_pos = 'front'

        qry_root_directory = f"../../Datasets/dalby/{location}"
        qry_vpr_root = f"../../Datasets/dalby/{location}/vpr_ftrs/"
        qry_image_dir = f"{qry_root_directory}/{qry_set}/{qry_camera_pos}-imgs/"
        qry_utm_dir = f"{qry_root_directory}/{qry_set}/utm/"


        qry_timestamps = [filename.split('.png')[0] for filename in natsorted(os.listdir(qry_image_dir)) if os.path.isfile(qry_image_dir+filename)]
        qry_utms = np.array([np.loadtxt(qry_utm_dir+filename) for filename in natsorted(os.listdir(qry_utm_dir)) if os.path.isfile(qry_utm_dir+filename)])
        # qry_name_sort_idx = index_natsorted(os.listdir(qry_image_dir))

        # if slice_len is None:
        #     # Get the two orderings
        #     glob_sorted_paths = sorted(glob(f"{qry_image_dir}/*.png"))
        #     glob_sorted_filenames = [os.path.basename(p) for p in glob_sorted_paths]

        #     # Get the indices that would sort glob_sorted_filenames into natsorted order
        #     qry_name_sort_idx = index_natsorted(glob_sorted_filenames)

        #     qry_ftrs = np.load(f"{qry_vpr_root}/{qry_set}/{vpr_desc}/queries_descriptors.npy")
        #     qry_local_ftrs = np.load(f"{qry_vpr_root}/{qry_set}/{vpr_desc}/qry_local_feats.npy")
        #     qry_ftrs = qry_ftrs[qry_name_sort_idx]
        #     qry_local_ftrs = qry_local_ftrs[qry_name_sort_idx]

        # mInds, dMat = getMatchIndsGPU(ref_ftrs,qry_ftrs,topK=1)
        # mInds = mInds.cpu().numpy()
        if slice_len is None:
            mInds = run_rerank(qry_ftrs, ref_ftrs, qry_local_ftrs, ref_local_ftrs, recall_values=[1, 5, 10, 20])[:,0] # 5, 10, 20
        else:
            print(f"Performing VPR on slices")
            num_slices = floor(len(qry_timestamps)/slice_len)
            if len(qry_timestamps) % slice_len > 0:
                num_slices += 1

            for idx in tqdm(range(num_slices)):
                qry_ftrs = np.load(f"{qry_vpr_root}/{qry_set}/{vpr_desc}/sliced/queries_descriptors_slice_{idx:05d}.npy")
                qry_local_ftrs = np.load(f"{qry_vpr_root}/{qry_set}/{vpr_desc}/sliced/qry_local_feats_slice_{idx:05d}.npy")
                if idx == 0:
                    mInds = run_rerank(qry_ftrs, ref_ftrs, qry_local_ftrs, ref_local_ftrs, recall_values=[1, 5, 10, 20])[:,0]
                else:
                    mInds_slice = run_rerank(qry_ftrs, ref_ftrs, qry_local_ftrs, ref_local_ftrs, recall_values=[1, 5, 10, 20])[:,0]
                    mInds = np.vstack((np.expand_dims(mInds, axis=1), np.expand_dims(mInds_slice, axis=1))).squeeze()

                del qry_ftrs
                del qry_local_ftrs

            print(f"VPR on query slices: {len(mInds)}")

        np.save(f"{qry_vpr_root}/{qry_set}/{vpr_desc}/mInds.npy", mInds)

        
        in_tol = []
        dists = []
        valid_qry = 0

        qry_utm_timestamps, qry_utm_idxs = get_all_corr_files(qry_timestamps, [qry_utm_dir,])
        ref_utm_timestamp, ref_utm_idxs = get_all_corr_files(ref_timestamps, [ref_utm_dir,])

        for qry_idx in tqdm(range(len(qry_timestamps))):

            qry_image_timestamp = qry_timestamps[qry_idx]
            qry_image_filename = f"{qry_image_dir}/{qry_image_timestamp}.png"
            qry_utm = qry_utms[qry_utm_idxs[qry_idx]]


            diffs = ref_utms - qry_utm           # shape (N, 2)
            qry_dists = np.linalg.norm(diffs, axis=1)   # shape (N,)
            if qry_dists.min() > dist_tolerance:
                continue
            else:
                valid_qry += 1

            ref_utm = ref_utms[ref_utm_idxs[int(mInds[qry_idx])]]

            diff = ref_utm - qry_utm           # shape (N, 2)
            dist = np.linalg.norm(diff)   # shape (N,)
            dists.append(dist)
            if dist < dist_tolerance:
                in_tol.append(1)
            else:
                in_tol.append(0)

            # qry_image = ImageData(qry_image_filename, img_calib_file)

            # fig, ax = plt.subplots(1, 2, figsize=(19.4, 6))
            # ax[0].clear()
            # ax[1].clear()

            # ax[0].imshow(qry_image.image[:, :, ::-1])
            # ax[0].set_title(f"{qry_image_timestamp}.png")
            # ax[0].axis("off")

            # # Show matching reference image
            # # ref_img_timestamp = utils.get_corr_files(ref_timestamps[int(mInds[qry_idx])], [ref_image_dir,])
            # ref_image = ImageData(f"{ref_image_dir}/{ref_timestamps[int(mInds[qry_idx])]}.png", img_calib_file)
            # ax[1].imshow(ref_image.image[:, :, ::-1])
            # ax[1].set_title(f"{ref_timestamps[int(mInds[qry_idx])]}\nDist={dist:.2f}m")

            # ax[1].axis("off")
            # fig.canvas.draw()

        print(f"Recall for {qry_set} using {vpr_desc}: {np.sum(np.array(in_tol))/valid_qry:.02%}")
        all_results.append(np.sum(np.array(in_tol))/valid_qry)
        # plt.figure()
        # plt.plot(np.clip(dists, 0, 30))
        # plt.ylim((0,35))
    
    print(f"All {vpr_desc} results:")
    print(all_results)

    # else:
    #     num_slices = floor(len(ref_img_filenames)/slice_len)
    #     if len(ref_img_filenames) % slice_len > 0:
    #         num_slices += 1

    #     for idx in range(num_slices):
    #         if idx == 0:
    #             ref_ftrs = np.load(f"{ref_vpr_root}/{ref_set}/{vpr_desc}/sliced/queries_descriptors_slice_{idx:05d}.npy")
    #         else:
    #             ref_ftrs = np.vstack((ref_ftrs, np.load(f"{ref_vpr_root}/{ref_set}/{vpr_desc}/sliced/queries_descriptors_slice_{idx:05d}.npy")))

        
    #     print(f"Loaded ref ftr slices: {len(ref_ftrs)}")




    #     if slice_len is None:
    #         mInds, dMat = getMatchIndsGPU(ref_ftrs,qry_ftrs,topK=1)
    #         mInds = mInds.cpu().numpy()
    #     else:
    #         print(f"Performing VPR on slices")
    #         num_slices = floor(len(qry_timestamps)/slice_len)
    #         if len(qry_timestamps) % slice_len > 0:
    #             num_slices += 1

    #         for idx in tqdm(range(num_slices)):
    #             qry_ftrs = np.load(f"{qry_vpr_root}/{qry_set}/{vpr_desc}/sliced/queries_descriptors_slice_{idx:05d}.npy")
    #             if idx == 0:
    #                 mInds, dMat = getMatchIndsGPU(ref_ftrs,qry_ftrs,topK=1)
    #                 mInds = mInds.cpu().numpy()
    #             else:
    #                 mInds_slice, dMat = getMatchIndsGPU(ref_ftrs,qry_ftrs,topK=1)
    #                 mInds_slice = mInds_slice.cpu().numpy()
    #                 mInds = np.vstack((np.expand_dims(mInds, axis=1), np.expand_dims(mInds_slice, axis=1))).squeeze()

    #         print(f"VPR on query slices: {len(mInds)}")
