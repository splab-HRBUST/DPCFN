# Twice Attention Networks for Synthetic Speech Detection

## Detection

This repo has an implementation for our paper **Twice Attention Networks for Synthetic Speech Detection**:   

**Chen Chen, Yaozu Song, Bohan Dai, Deyun Chen. Twice attention networks for synthetic speech detection[J]. Neurocomputing, 2023, 559: 126799.**  

The experiment adopted https://github.com/nesl/asvspoof2019 as the baseline system for modification.

## Dataset

The ASVSpoof2019 dataset can be downloaded from the following link:

[ASVSpoof2019 dataset](https://datashare.is.ed.ac.uk/handle/10283/3336)

### Training models

```
python model_main2.py --num_epochs=100 --track=logical --features=spect   --lr=0.0001
```

Please note that the CQCC features are computing using the Matlab code in [cqcc_extraction.m](./cqcc_extraction.m), so you need to run this file to generate cache files of CQCC featurs before attempting to traiin or evaluate models with CQCC features.

### To perform fusion of multiple results files

```
 python fuse_result.py --input FILE1 FILE2 FILE3 --output=RESULTS_FILE
```

### Evaluating Models

Run the model on the evaluation dataset to generate a prediction file.

```

python model_main2.py --is_eval --eval  --eval_output='eval_CM_scores 70.txt' --model_path=/g813_u1/mkj/twice_attention_networks-main/ta-network-main/models/model3_logical_spect_100_32_0.0001_Unet_innovation23/epoch_70.pth
```


Then compute the evaluation scores using on the development dataset

```
python evaluate_tDCF_asvspoof19.py RESULTS_FILE PATH_TO__asv_dev.txt 
```

## DDP training

By using the command `python narrow_setup.py`, the script automatically selects the suitable `GPU` for training.

Modifying the `cmd` variable in  `narrow_setup.py`  to perform training and testing.

The paper has been published in [Neurocomputing](https://www.sciencedirect.com/science/article/abs/pii/S0925231223009220).


#用于合成语音检测的二次注意力网络
# #检测
这个仓库实现了我们的论文**Twice Attention Networks for Synthetic Speech Detection**:
**陈晨，宋耀祖，戴博涵，陈德云。基于二次注意力机制的合成语音检测[J]。** .神经网络，2023,559:126799

实验采用https://github.com/nesl/asvspoof2019作为基线系统进行修改。
# #数据集
ASVSpoof2019数据集可以从以下链接下载:
(ASVSpoof2019数据集)(https://datashare.is.ed.ac.uk/handle/10283/3336)

###训练模型
' ' '
Python model_main.py——num_epochs=100——track=[logical/physical]——features=[spect/mfcc/cqcc]——lr=0.00005

' ' '
请注意，CQCC特征是使用[cqcc_extraction.m](./cqcc_extraction.m)中的Matlab代码计算的，因此在尝试使用CQCC特征训练或评估模型之前，您需要运行此文件以生成CQCC特征的缓存文件。

###对多个结果文件进行融合
' ' '
python fuse_result.py——input FILE1 FILE2 FILE3——output=RESULTS_FILE .py

' ' '
###评估模型
在评估数据集上运行模型以生成预测文件。
' ' '

python model_main.py——eval——eval_output=RESULTS_FILE——model_path=CHECKPOINT_FILE

' ' '
然后在开发数据集上计算评估分数
' ' '
python evaluate_tDCF_asvspoof19.py RESULTS_FILE PATH_TO__asv_dev.txt

' ' '
## DDP培训
通过使用命令`python narrow_setup.py`，脚本会自动选择合适的`GPU`进行训练。
修改`narrow_setup.py`中的`cmd`变量来进行训练和测试。
论文发表在[Neurocomputing](https://www.sciencedirect.com/science/article/abs/pii/S0925231223009220)上。