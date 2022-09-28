# AdaVQA: Overcoming Language Priors with Adapted Margin Cosine Loss
Official implementation for the IJCAI'21 paper. 

This repository is built upon the [code](https://github.com/hengyuan-hu/bottom-up-attention-vqa). Thanks for the code sharing of the authors.

Almost all flags can be set by yourself at [utils/config.py](utils/config.py)! We have another extension paper with the LXMERT as baseline achieves SOTA results of `71.44` on the VQA-CP v2 dataset. You can easily combine this [loss](utils/losses.py) with our [LXMERT](https://github.com/guoyang9/LXMERT-VQACP) implementation.

|              | Y/N   | Num.  | Other | All   |
|--------------|-------|-------|-------|-------|
| AdaVQA(UpDn) | 72.47 | 53.81 | 45.58 | 54.67 |
| MMDB(LXMERT) | 91.37 | 65.55 | 62.61 | 71.44 |
|              |       |       |       |       |

### Prerequisites
* python==3.7.7
* pytorch==1.4.0
* tensorboardX==2.1
* torchvision==0.6.0
### Dataset
First of all, make all the data in the right position according to the `utils/config.py`!

* Please download the VQA-CP datasets in the original paper.
* The image features can be found at the UpDn repo.
* The pre-trained Glove features can be accessed via [GLOVE](https://nlp.stanford.edu/projects/glove/).


### Pre-processing
1. process questions and dump dictionary:
    ``` python
    python tools/create_dictionary.py
    ```
2. process answers and question types:

    ``` python
    python tools/compute_softscore.py
    ``` 
3. convert image features to h5:
    ``` python
    python tools/detection_features_converter.py 
    ```
### Model Training
``` python
python main.py --name test-VQA --gpu 0
```

### Model Evaluation 
``` python
python main.py --name test-VQA --eval-only
``` 
### Citation
If you want to use this code, please cite our papers as follows:
``` ruby
@Inproceedings{adaVQA,
  author    = {Yangyang Guo, Liqiang Nie, Zhiyong Cheng, Feng Ji, Ji Zhang, Alberto Del Bimbo},
  title     = {AdaVQA: Overcoming Language Priors with Adapted Margin Cosine Loss},
  booktitle = {IJCAI},
  year      = {2021},
}
@article{MMDB,
  author    = {Yangyang Guo and
               Liqiang Nie and
               Harry Cheng and
               Zhiyong Cheng and
               Mohan S. Kankanhalli and
               Alberto Del Bimbo},
  title     = {On Modality Bias Recognition and Reduction},
  journal   = {ACM ToMM},
  year      = {2022},
}
```
