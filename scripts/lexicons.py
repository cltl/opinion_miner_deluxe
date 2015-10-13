#!/usr/bin/env python

import sys
import os
import csv
from subprocess import Popen,PIPE

def create_lexicons(path_to_script, training_file,exp_filename, tar_filename):
    cmd = ['python']
    cmd.append(path_to_script)
    cmd.append('-exp_csv')
    cmd.append(exp_filename)
    cmd.append('-tar_csv')
    cmd.append(tar_filename)
    cmd.append('-l')
    cmd.append(training_file)
    folder = os.path.dirname(exp_filename)
    log_out = open(os.path.join(folder,'log.out'),'wb')
    log_err = open(os.path.join(folder,'log.err'),'wb')
    
    lexicon_generator = Popen(' '.join(cmd),stdout=log_out, stderr=log_err, shell=True)
    ret_code = lexicon_generator.wait()
    log_out.close()
    log_err.close()
    
    print>>sys.stderr,' Lexicons created, on',folder,' ret code:',ret_code
    

def load_lexicon(lexicon_filename):
    ### LEXICON FROM THE DOMAIN
    fd = open(lexicon_filename,'rb')
    ##dialect = csv.Sniffer().sniff(fd.read(1024))
    ##fd.seek(0)
    #lex_reader = csv.reader(fd,dialect)
    lex_reader = csv.reader(fd,delimiter=';')
    my_lexicon = {}
    for n,row in enumerate(lex_reader):
        if n != 0:
            text_type,ratio,rel_freq,over_freq,lemmas,postags,freqwords = row
            this_pos = text_type.rfind('#')
            text = text_type[:this_pos]
            my_type =  text_type[this_pos+1:]
            my_lexicon[text.decode('utf-8')] = my_type.decode('utf-8')
    return my_lexicon
