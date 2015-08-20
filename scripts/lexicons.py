#!/usr/bin/env python

import sys
import os
import csv
from subprocess import Popen,PIPE

def load_lexicons(config_manager,lang,on_training=True):
    lexicons = []
    for this_label, this_type, path_to_lexicon in config_manager.get_lexicons():
        lexicon_obj = CLexicon()
        if path_to_lexicon == 'create_from_training':
            #Create it from training
            if on_training:
                create_lexicon_from_training(config_manager)
                
            #Expression lexicon
            lexicon_exp = CLexicon()
            lexicon_exp.load(config_manager.get_expression_lexicon_filename(),this_type,'exp_lex_from_train')
            lexicons.append(lexicon_exp)
            
            ##Target lexicon
            lexicon_tar = CLexicon()
            lexicon_tar.load(config_manager.get_target_lexicon_filename(),this_type,'tar_lex_from_train')
            lexicons.append(lexicon_tar)
        else:
            lexicon_obj = CLexicon()
            if '$LANG' in path_to_lexicon:
                path_to_lexicon = path_to_lexicon.replace('$LANG',lang)
            lexicon_obj.load(path_to_lexicon,this_type,this_label)
            lexicons.append(lexicon_obj)
    return lexicons

def create_lexicon_from_training(config_manager):
    path_to_script = '/home/izquierdo/opener_repos/opinion-domain-lexicon-acquisition/acquire_from_annotated_data.py'
    exp_filename = config_manager.get_expression_lexicon_filename()
    tar_filename = config_manager.get_target_lexicon_filename()
    training_file = config_manager.get_file_training_list()
    
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
    
    print>>sys.stderr,' Lexicons from training data created, on',folder,' ret code:',ret_code
    return 
    
class CLexicon:
    def __init__(self):
        self.type = None #lemma or multiword
        self.path_to_file = None
        self.label = None
        self.data = {}
        self.mapping_tokenids = {}
        
    def load(self,lexicon,type, label):
        self.type = type
        self.path_to_file = lexicon
        self.label = label
        if os.path.exists(self.path_to_file):
            if self.type == 'lemma':
                self.load_lemma_based()
            elif self.type == 'multiword':
                self.load_multiword_based()
            
    def is_lemma_based(self):
        return self.type == 'lemma'
    
    def is_multiword_based(self):
        return self.type == 'multiword'
    
    def get_label(self):
        return self.label
    
    def load_lemma_based(self):
        fic = open(self.path_to_file,'r')
        for line in fic:
            line = line.decode('utf-8').rstrip()
            tokens = line.split(';')
            lemma = tokens[4]
            polarity = tokens[2]
            self.data[lemma] = polarity

    def load_multiword_based(self):
        fd = open(self.path_to_file,'rb')
        lex_reader = csv.reader(fd,delimiter=';')
        for n,row in enumerate(lex_reader):
            if n != 0:
                text_type,ratio,rel_freq,over_freq,lemmas,postags,freqwords = row
                this_pos = text_type.rfind('#')
                text = text_type[:this_pos]
                my_type =  text_type[this_pos+1:]
                self.data[text.decode('utf-8')] = my_type.decode('utf-8')
                
                
    def create_mapping_for_tokenids(self,token_ids):
        self.mapping_tokenids = {}
        #Create index offset --> ids
        idx = 0
        my_map = {}
        text = ' '
        for token, tid in token_ids:
            for c in token:
                my_map[idx] = tid
                idx+=1
            text += token.lower()+' '
            idx+=1
        ####
        all_extracted = [] # List of [(ids,polarity), (ids, polarity)
    
        for substring, polarity in self.data.items():
            current_found = 0
            while True:
                start = text.find(' '+substring+' ',current_found)
                if start == -1: 
                    break
                end = start + len(substring)
                current_found = end
                ids = set(my_map[myidx] for myidx in range(start,end) if myidx in my_map)
                if len(ids) != 0:
                    all_extracted.append((ids,polarity))
                      
       
        #If w15 has been selected first, for instance (w14,w15,w16) will not be selected later in this file
        for ids,polarity in sorted(all_extracted, key=lambda t: len(t[0])):
            already_selected = False
            for this_id in ids:
                if this_id in self.mapping_tokenids:
                    already_selected = True
            
            if not already_selected:
                for this_id in ids:
                    self.mapping_tokenids[this_id] = polarity
    
    def get_value_for_lemma(self,lemma):
        return self.data.get(lemma,None)
                

    def get_value_for_tokenid(self,token_id):
        return self.mapping_tokenids.get(token_id,None)
    


    

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
