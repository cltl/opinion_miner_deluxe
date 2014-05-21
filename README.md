#Opinion miner deluxe#

##Introduction##

Opinion miner based on machine learning that can be trained using a list of
KAF/NAF files. It is important to notice that the opinion miner module will not call
to any external module to obtain features. It will read all the features from the input KAF/NAF file,
so you have to make sure that your input file contains all the required information in advance (tokens,
terms, polarities, constituents, entitiess, dependencies...). There are two basic functionalities:

* Training: from a corpus of opinion annotated files, induce and learn the models for detecting opinions
* Classification: using the previous models, find and extract opinions in new text files.

We provide models already trained and evaluated on hotel, news, attractions and restaurants domains for all the languages covered
by the OpeNER project. Most of the users will just focus on this classification step, using the models that we provide. Some others
will need to retrain the system to adapt it to a new domain or language. In the next sections we will introduce these 2 differents
usages of the opinion miner deluxe

##Classification##

In this case you have the models already trained (either you trained them yourself or got the ones we provide) and you want just to detect
the opinions in a new file. The input format of your file needs to be valid KAF format. The script that perfoms the classification is the script
`classify_kaf_naf_file.py`. You can get information about the available parameters by running the script with the parameter -h.
```shell
classify_kaf_naf_file.py -h
usage: classify_kaf_naf_file.py [-h]
                                (-m MODEL_FOLDER | -d DOMAIN | -show-models)
                                [-keep-opinions] [-no-time]

Detect opinion triples in a KAF/NAF file

optional arguments:
  -h, --help       show this help message and exit
  -m MODEL_FOLDER  Folder storing the trained models
  -d DOMAIN        The domain where the models were trained
  -show-models     Show the models available and finish
  -keep-opinions   Keep the opinions from the input (by default will be deleted)
  -no-time         No include time in timestamp (for testing)
```

The script reads the input KAF file from the standard input and will write the output KAF into the standard output. The main parameter is the model that
will be used. There are two ways of specifyng this parameter:
* By using the -m FOLDER option, by means of which we can specify that we would like to use exactly the folder stored in the path FOLDER
* By using the -d DOMAIN option, where DOMAIN is the domain where the model that we want to use was trained.

We can get which are the models available by running:
```shell
classify_kaf_naf_file.py -show-models
#########################
Models available
#########################
  Model 0
    Lang: en
    Domain: hotel
    Folder: final_models/en/hotel_cfg1
    Desc: Trained with config1 in the last version of hotel annotations
  Model 1
    Lang: en
    Domain: news
    Folder: final_models/en/news_cfg1
    Desc: Trained with config1 using only the sentences annotated with news
....
....
```

You can train as use as many models as you want. You will need the file `models.cfg` which contains the metadata about which models
are available and how to refer to them (the domain). This is an example of the content of this file:
```shell
#LANG|domain|pathtomodel|description
en|hotel|final_models/en/hotel_cfg1|Trained with config1 in the last version of hotel annotations
en|news|final_models/en/news_cfg1|Trained with config1 using only the sentences annotated with news
nl|hotel|final_models/nl/hotel_cfg1|Trained with config1 in the last version of hotel annotations
nl|news|final_models/nl/news_cfg1|Trained with config1 using only the sentences annotated with news
```
So in each line a model is specified and represented using 4 fields, the language, the domain identifier (which will be used later to refer to this model),
the path to the folder and a text with a description. The language for the KAF file will be read directly from the KAF header, and considering this model
and the domain id provided to the script, the proper model will be loaded and used.

So if you want to tag a file with Dutch text called input.nl.kaf with the models trained on hotel reviews, and store the result on the file output.nl.kaf you just
should call to the program as:
```shell
cat input.nl.kaf | python classify_kaf_naf_file.py -d hotel > output.nl.kaf
```


##Training##

The task is divided into 2 steps
* Detection of opinion entities (holder, target and expression): using
Conditional Random Fields
* Opinion entity linking (expression<-target and expression-<holder): using
binary Support Vector Machines

In next subsections, a brief explanation of the 2 steps is given.

###Opinion Entity detection###

The first step when extracting opinions from text is to determine which portions of text represent the different opinion entities:

- Opinion expressions: very nice, really ugly ...
- Opinion targets: the hotel, the rooms, the staff ...
- Opinion holders: I, our family, the manager ...

In order to do this, three different Conditional Random Fields (CRF) classifiers have been trained using by default this set of features: tokens,
lemmas, part-of-speech tags, constituent labels and polarity of words and entities. These classifiers detect portions of text representeing differnet opinion
entities.


###Opinion Entity linking###

This step takes as input the opinion entities detected in the previous step, and links them to create the final opinions <expression/target/holder>.
In this case we have trained two binary Support Vector Machines (SVM), one that indicates the degree of association between a given target and a given expression,
and another one that gives the degree of linkage between a holder and an opinion expression. So given a list of expressions, a list of targets and holders detected
by the CRF classifiers, the SVM models try to select the best candidate from the target list for each expressions, and the best holder from the holder list, to create
the final opinion triple.

Considering a certain opinion expression and a target, these are the features by default used to represent this data for the SVM engine:

1) Textual features: tokens and lemmas of the expression and the target
2) Distance features: features representing the relative distance of both elements in the text (normalized to a discrete list of possible values: far/medium/close for instance),
  and if both elements are in the same sentence or not
3) Dependency features: to indicate the dependency relations between the two elements in the text (dependency path, and dependencies relations with the root of the sentence)

##Requirements##
This is the list of required libraries:
+ SVMLight: library for Support Vector Machines (http://svmlight.joachims.org/)
+ CRFsuite: library for Conditional Random Fields (http://www.chokkan.org/software/crfsuite/)
+ KafNafParserPy: library for parsing KAF or NAF files (https://github.com/cltl/KafNafParserPy)
+ VUA_pylib: library with functions used by the system (https://github.com/cltl/VUA_pylib)

To install SVMLight and CRFsuite please visit the corresponding webpages and follow the instructions given. For the last two python libraries,
you will only to clone the repositories and make sure that both are in the python path so Python is able to find them (the easiest way is
to modify the variable PYTHON_PATH to include the path to these libraries if you don't want to modify your system files).

##Setting the opinion miner##

You will need first to install all the requirements on your local machine and then create a configuration file like this one:

```shell
[general]
output_folder = feat

[crfsuite]
path_to_binary = crfsuite

[svmlight]
path_to_binary_learn = /home/izquierdo/tools/svm_light/svm_learn
path_to_binary_classify = /home/izquierdo/tools/svm_light/svm_classify
````

The `output_folder` variable is the folder where the trained models have been stored. The rest of parameters are the local paths to your installation
of CRFsuite and SVMLight. This file will be passed to the main script to detect opinions in a new KAF/NAF file:

````shell
cat my_file.kaf | classify_kaf_naf_file.py your_config_file.cfg
````

##Training your own models##

You will need first to install all the requirementes given and then follow these steps:

1) Prepare the KAF/NAF files that you will be used for training, with as many layers as possible (for the default configuration, preferably KAF
files with tokens, terms, polarities, entities, aspects, constituents and dependencies). A file with the complete path to each training KAF
file needs to be created (my_list_kafs.txt, for instance)

2) Create the feature template files or modify the existing ones on the folder `my_templates`

3) Prepare a configuration file (or modify the existing one my_training.cfg) like this one:

````shell
[general]
output_folder = feat
filename_training_list = /home/izquierdo/data/MPQA/13jan2014/list.25

[feature_templates]
expression = my_templates/templates_exp.txt
holder = my_templates/templates_holder.txt
target = my_templates/templates_target.txt

[valid_opinions]
negative = sentiment-neg
positive = sentiment-pos

[crfsuite]
path_to_binary = /home/izquierdo/bin/crfsuite
parameters = -a lbfgs

[svmlight]
path_to_binary_learn = /home/izquierdo/tools/svm_light/svm_learn
path_to_binary_classify = /home/izquierdo/tools/svm_light/svm_classify
parameters = -c 0.1
````

The `output_folder` variable is where you want to store your new models (will be used later for tagging new files), and the `filename_training_list` is the file
you created with the paths to all your training KAF/NAF files (my_list_kafs.txt). The section feature_templates contains pointers to the feature template files
you want to use. The section valid_opinions allows you to specify which opinions from the training KAF files you want to use, and a mapping from all the labels
used in the KAF files. So with this configuration:

````shell
[valid_opinions]
negative = sentiment-neg
positive = sentiment-pos
````

the opinion expressions classifier will be trained for two classes (negative and positive), and for instance all the opinion expressions with the label sentiment-neg in
your KAF files will be used as training instance for the negative classifier. This allows you to use different sets of labels for the opinion expressions, for instance
you could use KAF files with differente labels for the negative expressions, like sentiment-low-negative, sentiment-medium-negative and sentiment-high-negative. To train the
system considering all these instances as training material for the negative classifier you will need to specify:

````shell
[valid_opinions]
negative = sentiment-low-negative;sentiment-medium-negative;sentiment-high-negative
positive = sentiment-pos
````

The rest of sections on the config file (crfsuite and svm_light) indicate the paths to your local installation of these libraries and the parameters accepted
by these  (check the webpage of the libraries for information about these parameters)

 4) Once completed the previous step, the training can be performed calling to the script train.py:

````shell
train.py my_modified_train.cfg
````

This will used the config file (my_modified_train.cfg) to train the system and will store all the models and different intermediate files on the folder you set.


##How to add new features##
This section explains how to add new features to the system

###Adding new features to the opinion entity detection (CRF)###

1) Modify the function that generates the features `scripts/extract_features.py-> extract_features_from_kaf_naf_file(...)`

1.1) Modify the variable `features`, is a list of features for each token

1.2) Modify the variable labels, which gives a name to each feature (lenghts must match)

2) With the previous step you can extract the features for a single token only. You need specify which features you want to use from the context,
and if you want to use bigrams/trigrams. In order to do this 3 different features templates have to be filled. These files are plain text files, and
the default files used can be found on the subfolder `my_templates`. One different feature template can be specify for each CRF classifier. The format
of these files are a set of lines like `1 token -2 -1 0`, where:

- The first 1 is the length of the template, in this case unigram
- Then 'n' labels that will be used (must match with the labels generated by the feature extractor)
- Then the positions, in case of 2grams 3grams each position must be n/m/p

An example with bigrams: `2 token token -2/-1 -1/0 0/1 1/2` which would generate these templates:

````shell
(('token',-2),('token',-1))
(('token',-1),('token',0))
(('token',1),('token',1))
````

An one more example with trigrams: `3 token lemma pos -2/0/4 9/8/3`.

````shell
(('token',-2),('lemma',0),('pos',4))
(('token',9),('lemma',8),('pos',3))
````




###Adding new features to the opinion entity linking (SVM)###

You will need to modify the script `scripts/extract_feats_relations.py`. There is one function to extract the features from an opinion
expression and a target, for the SVM model expression - target, and another function with the same purpose for the SVM model expression-holder.
These functions are:

````shell
def extract_feats_exp_tar(exp_ids,tar_ids,knaf_obj):
    ...
    
def extract_feats_exp_hol(exp_ids,hol_ids,knaf_obj):
    ...
````

Both take as input a list of term identifiers for the expression and for the target/holder, and a kaf/naf tree object representing the input file,
so there is no need to parse it again. These functions return a list of features for the expression, a list of features for the holder/target and two
extra list of features (for the expression and for the target/holder), that will be used later to stablish features that represent a relation (like
the dependencies or whether both are in the same sentence or not.) In order to to this, there are two functions that take as input two set of features
and generate this relation features:

````shell
def get_extra_feats_exp_tar(extra_e, extra_t):
    ...

def get_extra_feats_exp_hol(extra_e, extra_h):
    ...
````

The main reason of this is that the features for each expression, target and holder is extracted only once, but later for instance each target will act
as a positive example in one case (with its correct expression), but as negative example for the rest of possible expressions in the file. So the relation
features can not be extracted in advance for a pair expression/target but has to be computed for each pair we consider, and in order to do this we need
the two get_extra_feats functions indicated above.

##Contact##
* Ruben Izquierdo
* Vrije University of Amsterdam
* ruben.izquierdobevia@vu.nl