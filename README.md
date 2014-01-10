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

###Adding new features to the opinion entity linking (SVM)###

##Contact##
* Ruben Izquierdo
* Vrije University of Amsterdam
* ruben.izquierdobevia@vu.nl