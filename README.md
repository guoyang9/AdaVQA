# AdaVQA: Overcoming Language Priors with Adapted Margin Loss
This repository is built upon the [code](https://github.com/Cyanogenoid/vqa-counting.git) provided by @Yan Zhang. Futher introduction will be given shortly.
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
First of all, make all the data in the right position according to the `config.py`.

* Please download the VQA-CP datasets in the original paper.
* The pre-trained Glove features can be found on [glove website](https://nlp.stanford.edu/projects/glove/).
* The image features can be found at the UpDn repo.
## Preprocessing
1. There are two ways to extract image features: grid-based and rcnn-based.
	* grid-based: preprocess the image feature, including extracting pre-trained image faetures.
		```
		python preprocess/preprocess-image-grid.py
		```
	* rcnn-based: preprocess the image feature, including extracting pre-trained image faetures and the corresponding bounding boxes.
		```
		python preprocess/preprocess-image-rcnn.py
		```
1. Preprocess the vocabulary, including constructing glove embedding file and filtering top 3000 answers.
	```
	python preprocess/preprocess-vocab.py
	```

## Model Training
```
python main.py --name=test-vqa --gpu=0
```
## Model Test only
```
python main.py --test --name=somename --gpu=0
```

## Citation
If you plan to use this code as part of your published research, we'd appreciate it if you could cite our paper:
```
@Inproceedings{adaVQA,
  author    = {Yangyang Guo, Liqiang Nie, Zhiyong Cheng, Feng Ji, Ji Zhang, Alberto Del Bimbo},
  title     = {AdaVQA: Overcoming Language Priors with Adapted Margin Loss},
  booktitle = {IJCAI},
  year      = {2021},
}
```
