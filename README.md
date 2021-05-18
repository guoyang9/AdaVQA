# AdaVQA: Overcoming Language Priors with Adapted Margin Cosine Loss
This repository is built upon the [code](https://github.com/hengyuan-hu/bottom-up-attention-vqa). Futher introduction will be given shortly.

Almost all flags can be set by yourself at `utils/config.py`!

## Prerequisites
    * python==3.7.7
    * nltk==3.4
    * bcolz==1.2.1
    * tqdm==4.31.1
    * numpy==1.18.4
    * pytorch==1.4.0
    * tensorboardX==2.1
    * torchvision==0.6.0
## Dataset
First of all, make all the data in the right position according to the `utils/config.py`!

* Please download the VQA-CP datasets in the original paper.
* The image features can be found at the UpDn repo.
* The pre-trained Glove features can be accessed via [GLOVE](https://nlp.stanford.edu/projects/glove/).


## Pre-processing

1. process questions and dump dictionary:
    ```
    python tools/create_dictionary.py
    ```

2. process answers and question types:

    ```
    python tools/compute_softscore.py
    ```
3. convert image features to h5:
    ```
    python tools/detection_features_converter.py 
    ```
## Model Training
```
python main.py --name test-VQA --gpu 0
```

## Model Evaluation 
```
python main.py --name test-VQA --eval-only
```
## Citation
If you want to use this code, please cite our paper as follows:
```
@Inproceedings{adaVQA,
  author    = {Yangyang Guo, Liqiang Nie, Zhiyong Cheng, Feng Ji, Ji Zhang, Alberto Del Bimbo},
  title     = {AdaVQA: Overcoming Language Priors with Adapted Margin Cosine Loss},
  booktitle = {IJCAI},
  year      = {2021},
}
```