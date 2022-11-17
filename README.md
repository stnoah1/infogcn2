# SODE

## Dependencies

- Python >= 3.8
- PyTorch >= 1.9.0
- NVIDIA Apex
- tqdm, tensorboardX, wandb, einops

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
- We set the seed number for Numpy and PyTorch as 1 for reproducibility.
- If you want to reproduce our works, please find the details in the supplementary matrials. The hyperparameter setting differs depending on the training dataset.
- This is an exmaple command for training InfoGCN on NTU RGB+D 60 Cross Subject split. Please change the arguments if you want to customize the training.

```
CUDA_VISIBLE_DEVICES=0 python main.py --half=True --batch_size=32 --test_batch_size=64 \
    --step 50 60 --num_epoch=70 --n_heads=3 --num_worker=4 \
    --dataset=NW-UCLA --num_class=10 --z_prior_gain=3 \
    --datacase=ucla --weight_decay=0.0005 --num_person=1 --num_point=20 --graph=graph.ucla.Graph --feeder=feeders.feeder_ucla.Feeder --base_lr 1e-1 --base_channel 64 --dct False --window_size 52 --lambda_1=1e-0 --lambda_2=1.0 --lambda_3=1e+1 --n_step 3 --save_epoch 0
```

### Testing

- To test the trained models saved in <work_dir>, run the following command:

```
python main.py --half=True --test_batch_size=128 --n_heads=3 --num_worker=4 \
    --k=1 --dataset=ntu --num_class=60 --use_vel=False --datacase=NTU60_CS \
    --num_person=2 --num_point=25 --graph=graph.ntu_rgb_d.Graph --feeder=feeders.feeder_ntu.Feeder \
    --phase=test --save_score=True --weights=<path_to_weight>
```

- To ensemble the results of different modalities, run the following command:
```
python ensemble.py \
   --dataset=ntu/xsub \
   --position_ckpts \
      <work_dir_1>/files/best_score.pkl \
      <work_dir_2>/files/best_score.pkl \
      ...
   --motion_ckpts \
      <work_dir_3>/files/best_score.pkl \
      <work_dir_4>/files/best_score.pkl \
      ...
```

## Acknowledgements

This repo is based on [2s-AGCN](https://github.com/lshiwjx/2s-AGCN) and [CTR-GCN](https://github.com/Uason-Chen/CTR-GCN). The data processing is borrowed from [SGN](https://github.com/microsoft/SGN), [HCN](https://github.com/huguyuehuhu/HCN-pytorch), and [Predict & Cluster](https://github.com/shlizee/Predict-Cluster).

Thanks to the original authors for their work!
