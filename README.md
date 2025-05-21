# DPCFN



## Dataset

The ASVSpoof2019 2021dataset can be downloaded from the following link:

[ASVSpoof2019 dataset](https://datashare.is.ed.ac.uk/handle/10283/3336)

### Training models

```
python model_main3.py --num_epochs=100 --track=logical --features=spect   --lr=0.0001
```

Please note that the CQCC features are computing using the Matlab code in [cqcc_extraction.m](./cqcc_extraction.m), so you need to run this file to generate cache files of CQCC featurs before attempting to traiin or evaluate models with CQCC features.

### To perform fusion of multiple results files

```
 python fuse_result.py --input FILE1 FILE2 FILE3 --output=RESULTS_FILE
```

### Evaluating Models

Run the model on the evaluation dataset to generate a prediction file.

```

python model_main3.py --is_eval --eval  --eval_output='eval_CM_scores 70.txt' --model_path=/g813_u1/mkj/twice_attention_networks-main/ta-network-main/models/model3_logical_spect_100_32_0.0001_Unet_innovation23/epoch_70.pth
```


Then compute the evaluation scores using on the development dataset

```
python evaluate_tDCF_asvspoof19.py RESULTS_FILE PATH_TO__asv_dev.txt 
```

## DDP training

By using the command `python narrow_setup.py`, the script automatically selects the suitable `GPU` for training.

Modifying the `cmd` variable in  `narrow_setup.py`  to perform training and testing.

The paper has been published in [Neurocomputing](https://www.sciencedirect.com/science/article/abs/pii/S0925231223009220).
