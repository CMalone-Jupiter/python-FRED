import os
import sys
sys.path.append(os.path.abspath("."))   # one level up
import numpy as np
from natsort import natsorted, index_natsorted
import torch
from tqdm import tqdm
from glob import glob

################## set device based on cuda availability #################
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print('CUDA availability: ' + str(torch.cuda.is_available()))

####################### Functions for matching using numpy on CPU or Pytorch on GPU ###################

slice_size = 1000
# qry_set = '20210909_124816_v2'
qry_set = '20230509_115540_v2'
vpr_desc = 'FoL'
img_calib_file = f"./camera_calib.txt"
# User parameters
location = 'dalby-to-brigalow'

################ Query filenames and directories #################################
qry_condition = ''
qry_camera_pos = 'front'

qry_root_directory = f"../../Datasets/dalby/{location}"
qry_vpr_root = f"../../Datasets/dalby/{location}/vpr_ftrs/"
qry_image_dir = f"{qry_root_directory}/{qry_set}/{qry_camera_pos}-imgs/"
save_dir = f"../../Datasets/dalby/{location}/vpr_ftrs/{qry_set}/{vpr_desc}/sliced/"

os.makedirs(save_dir, exist_ok=True)


qry_timestamps = [filename.split('.png')[0] for filename in natsorted(os.listdir(qry_image_dir)) if os.path.isfile(qry_image_dir+filename)]

# Get the two orderings
glob_sorted_paths = sorted(glob(f"{qry_image_dir}/*.png"))
glob_sorted_filenames = [os.path.basename(p) for p in glob_sorted_paths]

# Get the indices that would sort glob_sorted_filenames into natsorted order
qry_name_sort_idx = index_natsorted(glob_sorted_filenames)

print(f"Loading query features")

qry_ftrs = np.load(f"{qry_vpr_root}/{qry_set}/{vpr_desc}/queries_descriptors.npy")

print(f"Loading query local features")
qry_local_ftrs = np.load(f"{qry_vpr_root}/{qry_set}/{vpr_desc}/qry_local_feats.npy")
qry_ftrs = qry_ftrs[qry_name_sort_idx]
qry_local_ftrs = qry_local_ftrs[qry_name_sort_idx]

assert qry_ftrs.shape[0] == qry_local_ftrs.shape[0], f"There should be equal number of global ({qry_ftrs.shape[0]}) and local ({qry_local_ftrs.shape[0]}) features"

check_len_ftrs = 0
check_len_local_ftrs = 0
slice_num = 0
# f"{42:05d}" 
print(f"Starting slice n dice")
for idx in tqdm(range(0, qry_ftrs.shape[0], slice_size)):
    end_idx = idx+min(slice_size, qry_ftrs.shape[0]-idx)
    qry_ftrs_slice = qry_ftrs[idx:end_idx]
    qry_local_ftrs_slice = qry_local_ftrs[idx:end_idx]

    np.save(f"{save_dir}/queries_descriptors_slice_{slice_num:05d}.npy", qry_ftrs_slice)
    np.save(f"{save_dir}/qry_local_feats_slice_{slice_num:05d}.npy", qry_local_ftrs_slice)


    check_len_ftrs += qry_ftrs_slice.shape[0]
    check_len_local_ftrs += qry_local_ftrs_slice.shape[0]
    slice_num += 1

print(f"Query descriptors: {qry_ftrs.shape[0]}, Slice query descriptors: {check_len_ftrs}")
print(f"Query local descriptors: {qry_local_ftrs.shape[0]}, Slice query local descriptors: {check_len_local_ftrs}")
print(f"Number of slices: {slice_num}")
