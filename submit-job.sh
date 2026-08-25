#!/bin/bash

#PBS -N vpr-extract
#PBS -l select=1:ncpus=10:mem=100gb:ngpus=1:gpu_id=A100
#PBS -l walltime=06:00:00
#PBS -m abe
#PBS -M cj.malone@qut.edu.au
#PBS -j oe

micromamba activate fred


# Move to repo
cd "$HOME/cloned_repos/python-FRED/" || exit 1

# Run Python script
python localisation/VPR_eval-fol.py
# python FoL/split_local_feats.py
