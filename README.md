# Official PyTorch implementation of </br> _InfoGCN++: Learning Representation by Predicting the Future for Online Skeleton-based Action Recognition_ [[Paper](https://arxiv.org/pdf/2310.10547.pdf)]
<img width="1166" alt="framework" src="https://user-images.githubusercontent.com/37060326/202824122-e98197b6-ebe4-4487-8f08-aa08738cb86c.png">

## Abstract
While skeleton-based action recognition has made significant achievements in recent years, its application in real-time scenarios has been hindered by the necessity for complete action observation prior to classification. This is a critical shortcoming for scenarios that demand immediate action recognition, such as in surveillance systems and robotic applications. 
In this work, we introduce InfoGCN++, an innovative framework designed specifically for online skeleton-based action recognition. It offers the advantage of providing real-time action category classification, irrespective of the length of observation. Unlike traditional methods, InfoGCN++ learns to predict future motion and construct a comprehensive representation of the entire sequence based on observed actions.
We achieve this by predicting future prediction as an extrapolation problem based on observed actions and adopting the concept of Neural Ordinary Differential Equations (ODEs) to model the continuous evolution of hidden states.
Our experiments on three widely used skeleton-based action benchmarks showcase the superiority of InfoGCN++ in online action recognition, with results either outperforming or on par with existing methods. This underscores InfoGCN++'s significant potential for transforming real-time and online action recognition applications. 

```
@article{chi2023infogcn++,
  title={InfoGCN++: Learning Representation by Predicting the Future for Online Human Skeleton-based Action Recognition},
  author={Chi, Seunggeun and Chi, Hyung-gun and Huang, Qixing and Ramani, Karthik},
  journal={arXiv preprint arXiv:2310.10547},
  year={2023}
}
```

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
