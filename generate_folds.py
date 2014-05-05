#!/usr/bin/env python

import sys
import getopt
import logging
import os
import random
from shutil import rmtree


def usage(cmd):
    print>>sys.stderr,'Usage: '+cmd+' options'
    print>>sys.stderr,'Options:'
    print>>sys.stderr,'\t-f --file: input file with a list of documents (required)'
    print>>sys.stderr,'\t-n --num: num of folds to create (required)'
    print>>sys.stderr,'\t-o --out: name of the main folder to store the subfolds (required)'
    print>>sys.stderr,'\t-s --subfolder: prefix for the subfolders (optional, default "fold")'
    print>>sys.stderr
    print>>sys.stderr,'Examples'
    print>>sys.stderr,'\tgenerate_folds.py -f vu.doclist.attitude.ula.xbank --num 10 -o out_folder'
    print>>sys.stderr,'\tgenerate_folds.py -f vu.doclist.attitude.ula.xbank --num 10 -o out_folder --subfolder my_custom_fold'


def generate_folds(input_file,num_folds,out_folder,name_subfolder='fold'):    
    # Load the input file
    logging.debug('Loading elements from '+input_file)
    elements = []
    fic = open(input_file,'rU')
    for line in fic:
        elements.append(line.strip())
    fic.close()
    logging.debug('Loaded '+str(len(elements))+' elements')
    
    ##Get just the %percent
    percent = 25
    original_len = len(elements)
    new_len = original_len*percent/100
    elements = elements[:new_len]
    
    ## Creating folders and subfolders:
    if os.path.exists(out_folder):
        print>>sys.stderr,'Output folder '+out_folder,'already exists'
        rmtree(out_folder)
        print>>sys.stderr,'It has been removed...'
        #sys.exit(-1)
        
    logging.debug('Creating '+out_folder+' and subfolders')
    folds = []
    os.mkdir(out_folder)
    for n in range(num_folds):
        my_name = os.path.join(out_folder,name_subfolder+'_'+str(n))
        os.mkdir(my_name)
        logging.debug('Created '+my_name)
        folds.append(my_name)
    ###################################################
    
    ## Creating folds
    size_of_fold = len(elements) / num_folds
    my_begin = 0
    my_end = size_of_fold
    
    random.shuffle(elements)
    for n in range(num_folds):
        this_fold = folds[n]
        my_test = elements[my_begin:my_end]
        my_train = elements[:my_begin]+elements[my_end:]
        if len( set(my_test) & set(my_train)) != 0:
            print>>sys.stderr,'Error overlapping'
            print>>sys.stderr,my_train
            print>>sys.stderr,my_test
        my_begin = my_end
        my_end = my_end + size_of_fold
        
        #Save the folds
        fic_train = open(os.path.join(this_fold,'train'),'w')
        logging.debug('Writing info to '+fic_train.name)
        for ele in my_train:
            fic_train.write(ele+'\n')
        fic_train.close()
        
        fic_test  =open(os.path.join(this_fold,'test'),'w')
        logging.debug('Writing info to '+fic_test.name)
        for ele in my_test:
            fic_test.write(ele+'\n')
        fic_test.close()
    ####
    logging.debug('Finished OK')

if __name__ == '__main__':
    logging.basicConfig(stream=sys.stderr,format='%(asctime)s - %(levelname)s - %(message)s',level=logging.DEBUG)

    input_file = None
    num_folds = None
    out_folder = None
    name_subfolder = 'fold'
    
    try:
        opts, args = getopt.getopt(sys.argv[1:],"f:n:o:s:",["file=","num=","out=","subfolder="])
        for opt, arg in opts:
            if opt in ['-f','--file']:
                input_file = arg
            elif opt in ['-n','--num']:
                num_folds = int(arg)
            elif opt in ['-o','--out']:
                out_folder = arg
            elif opt in ['-s','--subfolder']:
                name_subfolder = arg
    except getopt.GetoptError as e:
        print>>sys.stderr,'Warning: ',str(e)
        
    if input_file is None:
        print>>sys.stderr,'ERROR!!!! Input file missing'
        print
        usage(sys.argv[0])
        sys.exit(-1)
        
    if num_folds is None:
        print>>sys.stderr,'ERROR!!!! Num of folds missing'
        print
        usage(sys.argv[0])
        sys.exit(-1)
        
    if out_folder is None:
        print>>sys.stderr,'ERROR!!!! Out folder missing'
        print
        usage(sys.argv[0])
        sys.exit(-1)
        
    ###### END
    generate_folds(input_file,num_folds,out_folder)    

 