
``A Neural Model for User Geolocation and Lexical Dialectology`` 
====

Abstract
------------
We propose a simple yet effective text-
based user geolocation model based on
a neural network with one hidden layer,
which achieves state of the art performance
over three Twitter benchmark geolocation
datasets, in addition to producing word and
phrase embeddings in the hidden layer that
we show to be useful for detecting dialectal
terms. As part of our analysis of dialectal
terms, we release DAREDS, a dataset for
evaluating dialect term detection methods.

DAREDS dataset
--------------
DAREDS is a dataset consisting of dialect words and their
dialect region extracted from Dictionary of American Regional
English (DARE) available online at http://www.daredictionary.com.
It is available at the data directory.

Geolocation datasets
--------------------
We experiment with three Twitter geolocation datasets
available at https://github.com/utcompling/textgrounder.


Usage
-----



```
usage: geodare.py [-h] [-dataset str] [-model str] [-datadir DATADIR] [-tune]

optional arguments:
  -h, --help            show this help message and exit
  -dataset str, --dataset str
                        dataset name (cmu, na, world)
  -model str, --model str
                        dialectology model (mlp, lr, word2vec)
  -datadir DATADIR      directory where input datasets (cmu.pkl, na.pkl,
                        world.pkl) are located.
  -tune                 if true, tune the hyper-parameters.
```

The preprocessed pickle files (e.g. na.pkl) are the vectorized version of
the geolocation datasets and are available upon request.

Citation
--------
```
@InProceedings{rahimi2017a,
  author    = {Rahimi, Afshin  and  Cohn, Trevor  and  Baldwin, Timothy},
  title     = {A Neural Model for User Geolocation and Lexical Dialectology},
  booktitle = {Proceedings of ACL-2017 (short papers) preprint},
  year      = {2017},
  address   = {Vancouver, Canada},
  publisher = {Association for Computational Linguistics}
}
```

Contact
-------
Afshin Rahimi <afshinrahimi@gmail.com>




