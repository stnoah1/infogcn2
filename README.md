# Skeleton-ODE: Learning Representation by Predicting the Future for Online Skeleton-based Action Recognition
<img width="1166" alt="framework" src="https://user-images.githubusercontent.com/37060326/202824122-e98197b6-ebe4-4487-8f08-aa08738cb86c.png">

## Abstract
Skeleton-based action recognition algorithms have achieved impressive results in recent years.
However, they are unsuitable for applications requiring real-time and online decision-making. This limitation is problematic for scenarios where immediate action is necessary, such as surveillance or robotics. 
To address this limitation, we propose a novel framework called SkeletonODE for online skeleton-based action recognition that provides real-time action category classification at any observation length.
The SkeletonODE framework is designed to predict future motion from the observation and learns to represent the entire sequence from the previous observation.
We accomplish this by reformulating future prediction as an extrapolation of the observation and adopting the concept of Neural Ordinary Differential Equation (ODE) to model the continuous flow of hidden states.
Our experiments demonstrate the superiority of SkeletonODE for online skeleton-based action recognition, showing comparable or better performance compared to existing methods on three popular skeleton-based action benchmarks.
These results indicate that SkeletonODE has significant potential for real-time and online action recognition applications.

## Dependencies

- Python >= 3.8
- PyTorch >= 1.9.0
- NVIDIA Apex
- tqdm, tensorboardX, wandb
- einops, torchdiffeq

## Data Preparation

### Download datasets.

#### There are 3 datasets to download:

- NTU RGB+D 60 Skeleton
- NTU RGB+D 120 Skeleton
- NW-UCLA

#### NTU RGB+D 60 and 120

1. Request dataset here: https://rose1.ntu.edu.sg/dataset/actionRecognition
2. Download the skeleton-only datasets:
   1. `nturgbd_skeletons_s001_to_s017.zip` (NTU RGB+D 60)
   2. `nturgbd_skeletons_s018_to_s032.zip` (NTU RGB+D 120)
   3. Extract above files to `./data/nturgbd_raw`

#### NW-UCLA

1. Download dataset from CTR-GCN repo: [https://github.com/Uason-Chen/CTR-GCN](https://github.com/Uason-Chen/CTR-GCN)
2. Move `all_sqe` to `./data/NW-UCLA`

### Data Processing

#### Directory Structure

Put downloaded data into the following directory structure:

```
- data/
  - NW-UCLA/
    - all_sqe
      ... # raw data of NW-UCLA
  - ntu/
  - ntu120/
  - nturgbd_raw/
    - nturgb+d_skeletons/     # from `nturgbd_skeletons_s001_to_s017.zip`
      ...
    - nturgb+d_skeletons120/  # from `nturgbd_skeletons_s018_to_s032.zip`
      ...
```

#### Generating Data

- Generate NTU RGB+D 60 or NTU RGB+D 120 dataset:

```
 cd ./data/ntu # or cd ./data/ntu120
 # Get skeleton of each performer
 python get_raw_skes_data.py
 # Remove the bad skeleton
 python get_raw_denoised_data.py
 # Transform the skeleton to the center of the first frame and vertically align to the ground
 python seq_transformation.py
```

## Training & Testing

### Training

- We set the seed number for Numpy and PyTorch as `1` for reproducibility.
- If you want to reproduce our works, please find the details in the supplementary matrials. The hyperparameter setting differs depending on the training dataset.
- This is an exmaple command for training SODE on NW-UCLA dataset. Please change the arguments if you want to customize the training.

```
python main.py --half=True --batch_size=32 --test_batch_size=64 \
    --step 50 60 --num_epoch=70 --num_worker=4 --dataset=NW-UCLA --num_class=10 \
    --datacase=ucla --weight_decay=0.0003 --num_person=1 --num_point=20 --graph=graph.ucla.Graph \
    --feeder=feeders.feeder_ucla.Feeder --base_lr 1e-1 --base_channel 64 \
    --window_size 52 --lambda_1=1e-0 --lambda_2=1e-1 --lambda_3=1e-3 --n_step 3
```

### Testing

- To test the trained models saved in <work_dir>, run the following command:

```
python main.py --half=True --test_batch_size=64 --num_worker=4 --dataset=NW-UCLA --num_class=10 \
    --datacase=ucla --num_person=1 --num_point=20 --graph=graph.ucla.Graph \
    --feeder=feeders.feeder_ucla.Feeder --base_channel 64 --window_size 52 --n_step 3 \
    --phase=test --weights=<path_to_weight>
```

## Acknowledgements

This repo is based on [2s-AGCN](https://github.com/lshiwjx/2s-AGCN), [CTR-GCN](https://github.com/Uason-Chen/CTR-GCN), and [InfoGCN](https://github.com/stnoah1/infogcn).
The data processing is borrowed from [SGN](https://github.com/microsoft/SGN), [HCN](https://github.com/huguyuehuhu/HCN-pytorch), and [Predict & Cluster](https://github.com/shlizee/Predict-Cluster).
We use the Differentiable ODE Solvers for pytorch from [torchdiffeq](https://github.com/rtqichen/torchdiffeq).

Thanks to the original authors for their work!
