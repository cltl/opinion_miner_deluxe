#Opinion miner deluxe#

##Introduction##

Opinion miner based on machine learning that can be trained using a list of
KAF/NAF files

The task is divided into 2 steps
* Detection of opinion entities (holder, target and expression): using
Conditional Random Fields
* Opinion entity linking (expression<-target and expression-<holder): using
binary Support Vector Machines

##Requirements##
This is the list of required libraries:
+ SVMLight: library for Support Vector Machines (http://svmlight.joachims.org/)
+ CRFsuite: library for Conditional Random Fields (http://www.chokkan.org/software/crfsuite/)
+ KafNafParserPy: library for parsing KAF or NAF files (https://github.com/cltl/KafNafParserPy)
+ VUA_pylib: library with functions used by the system (https://github.com/cltl/VUA_pylib)


##How to add new features##
This section explains how to add new features to the system

###Adding new features to the opinion entity detection (CRF)###

1) Modify the function that generates the features is scripts/extract_features.py-> extract_features_from_kaf_naf_file(...)
1.1) Modify the variable features, is a list of features for each token
1.2) Modify the variable labels, which gives a name to each feature (lenghts must match)

2) Modify the function that generates from the templates the features for CRF (considering context)
2.1) The functions are in the script train.py->train(expression/target/holder)_classifier.py
2.2) Modify only the variable "templates", using the same labels as in the the variable "labels" in 1.2


###Adding new features to the opinion entity linking (SVM)###

##Contact##
* Ruben Izquierdo
* Vrije University of Amsterdam
* ruben.izquierdobevia@vu.nl