# declension-mi

This repository contains code accompanying the paper: Predicting Declension Class from Form and Meaning (Adina et al., ACL 2020). It is a study about the relationship between a noun's meaning / form and its declension class.


## Install Dependencies

Create a conda environment with
```bash
$ conda env create -f environment.yml
```
Then activate the environment and install your appropriate version of [PyTorch](https://pytorch.org/get-started/locally/).
```bash
$ conda install -y pytorch torchvision cudatoolkit=10.1 -c pytorch
$ # conda install pytorch torchvision cpuonly -c pytorch
```


## Parse data

To run the code on Czech data run:
```bash
$ python src/h01_data/parse.py --lang cze --rare-mode drop
```
Or, to run on everything, ask the authors for the german CELEX data (if you have access to it, since it is copyrighted) and then run:
```bash
$ source src/h01_data/parse_multi.sh
```

## Train models

To train the models and get results run:
```bash
$ python src/h02_learn/train.py --model <model> --languages <langs> --context <context>
```
Where `<context>` can be:
* none: No context used
* word2vec: Word2Vec context used

And the `<model>` is:
* `lstm`: which can be used with either context; or
* `mlp-word2vec`: only allowed with none context (word2vec is incorporated directly).

To get bayesian optimized hyper-parameters, run:
```bash
$ python src/h02_learn/train_bayes.py --model <model> --languages <langs>
```

After running bayesian optimization, get cross validation results by running:
```bash
$ python src/h02_learn/train_cv.py --model <model> --languages <langs> --opt
```
Where the `--opt` flag tells the script to load bayesian optimized hyper-parameters.

Or train all at once with:
```bash
$ python src/h02_learn/train_multi.sh
```

## Results

After training, results are saved in folder `results/cv/orig/`, where `test_loss` column in the csv files corresponds to the models' cross-entropies.


## Extra Information

#### Citation

If this code or the paper were usefull to you, consider citing it:

Citation Coming Soon!!!


#### Contact

To ask questions or report problems, please open an [issue](https://github.com/tpimentelms/declension-mi/issues).
