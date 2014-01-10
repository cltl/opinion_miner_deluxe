#!/usr/bin/env python

import sys
import os
from tempfile import NamedTemporaryFile
from subprocess import Popen, PIPE
import logging
import cPickle

from scripts.config_manager import Cconfig_manager
from scripts.extract_features import extract_features_from_kaf_naf_file
from scripts.crfutils import extract_features_to_crf 
from scripts.link_entities_distance import link_entities_distance
from scripts.relation_classifier import link_entities_svm
from KafNafParserPy import *



my_config_manager = Cconfig_manager()
__this_folder = os.path.dirname(os.path.realpath(__file__))
separator = '\t'
__desc = 'Deluxe opinion miner (CRF+SVM)'
__last_edited = '10jan2014'
__version = '2.0'

logging.basicConfig(stream=sys.stderr,format='%(asctime)s - %(levelname)s\n  + %(message)s', level=logging.CRITICAL)

terms_for_token = None


def load_obj_from_file(filename):
    fic = open(filename,'rb')
    obj = cPickle.load(fic)
    return obj

# Gets the output of crf and a list of token ids, and parses the B- or I- ...
# Output: [(['id0', 'id1', 'id2', 'id3'], 'holder'), (['id4', 'id5', 'id6'], 'target')]
def match_crfsuite_out(crfout,list_token_ids):
    matches = []
    inside = False
    current = []
    current_type = None
    num_token = 0
    for line in crfout.splitlines():
        if len(line) == 0:  #new sentence
            if inside:
                matches.append((current,current_type))
                current = []
                inside = False        
        else:
            if line=='O':
                if inside:
                    matches.append((current,current_type))
                    current = []
                    inside = False
            else:
                my_type = line[0]
                value = line[2:]
                if my_type == 'B':
                    if inside:
                        matches.append((current,current_type))
                    current = [list_token_ids[num_token]]
                    inside = True    
                    current_type = value    
                elif my_type == 'I':
                    if inside:
                        current.append(list_token_ids[num_token])
                    else:
                        current = [list_token_ids[num_token]]
                        current_type = value
                        inside = True
            num_token += 1
    if inside:
        matches.append((current,current_type))
    return matches


def extract_features(knaf_data):
    feat_file_desc = NamedTemporaryFile(delete=False)
    feat_file_desc.close()
    
    out_file = feat_file_desc.name
    err_file = out_file+'.log'
    
    labels, separator, knaf_obj = extract_features_from_kaf_naf_file(knaf_data,out_file,err_file,include_class=False)
    return out_file, err_file, knaf_obj
             
            
def convert_to_crf(input_file,templates):
    out_desc = NamedTemporaryFile(delete=False)
    out_desc.close()
    
    out_crf = out_desc.name
    
    ##Load description of features
    path_feat_desc = my_config_manager.get_feature_desc_filename()
    fic = open(path_feat_desc)
    fields = fic.read().strip()
    fic.close()
    ####
    
    extract_features_to_crf(input_file,out_crf,fields,separator,templates,possible_classes=None)
    return out_crf
             
             
             
def run_crfsuite_tag(input_file,model_file):
    crfsuite = my_config_manager.get_crfsuite_binary()
    cmd = [crfsuite]
    cmd.append('tag')
    cmd.append('-m '+model_file)
    cmd.append(input_file)

    crf_process = Popen(' '.join(cmd), stdin=PIPE, stdout=PIPE, stderr=PIPE, shell=True)
    crf_process.wait()
    output = crf_process.stdout.read()
    error = crf_process.stderr.read()
    return output,error


def detect_expressions(tab_feat_file,list_token_ids):
    #1) Convert to the correct CRF
    template_filename = my_config_manager.get_expression_template_filename()
    templates = load_obj_from_file(template_filename)
    
    crf_exp_file = convert_to_crf(tab_feat_file,templates)
    logging.debug('File with crf format for EXPRESSIONS '+crf_exp_file)
    model_file = my_config_manager.get_filename_model_expression()
    output_crf,error_crf = run_crfsuite_tag(crf_exp_file,model_file)
    logging.debug('Expressions crf error: '+error_crf)
    matches_exp = match_crfsuite_out(output_crf, list_token_ids)
    logging.debug('Detector expressions out: '+str(matches_exp))
    return matches_exp
    
    
    
    
    
def detect_targets(tab_feat_file, list_token_ids):
    template_filename = my_config_manager.get_target_template_filename()
    templates_target = load_obj_from_file(template_filename)    
    
    crf_target_file = convert_to_crf(tab_feat_file,templates_target)
    logging.debug('File with crf format for TARGETS '+crf_target_file)

    model_target_file = my_config_manager.get_filename_model_target()
    out_crf_target,error_crf = run_crfsuite_tag(crf_target_file, model_target_file)
    logging.debug('TARGETS crf error: '+error_crf)

    matches_tar = match_crfsuite_out(out_crf_target, list_token_ids)
    logging.debug('Detector targets out: '+str(matches_tar))
    return matches_tar
           
           
           
           
           
def detect_holders(tab_feat_file, list_token_ids):
    template_filename = my_config_manager.get_holder_template_filename()
    templates_holder = load_obj_from_file(template_filename)
    
    crf_holder_file = convert_to_crf(tab_feat_file,templates_holder)
    logging.debug('File with crf format for HOLDERS '+crf_holder_file)

    model_holder_file = my_config_manager.get_filename_model_holder()
    out_crf_holder,error_crf = run_crfsuite_tag(crf_holder_file, model_holder_file)
    logging.debug('HOLDERS crf error: '+error_crf)

    matches_holder = match_crfsuite_out(out_crf_holder, list_token_ids)
    logging.debug('Detector HOLDERS out: '+str(matches_holder))
    return matches_holder
              
              



def map_tokens_to_terms(list_tokens,knaf_obj):
    global terms_for_token
    if terms_for_token is None:
        terms_for_token = {}
        for term in knaf_obj.get_terms():
            termid = term.get_id()
            token_ids = term.get_span().get_span_ids()
            for tokid in token_ids:
                if tokid not in terms_for_token:
                    terms_for_token[tokid] = [termid]
                else:
                    terms_for_token[tokid].append(termid)
                    
    ret = set()
    for my_id in list_tokens:
        term_ids = terms_for_token[my_id]
        ret |= set(term_ids)
    return sorted(list(ret))
        
        
              
def add_opinions_to_knaf(triples,knaf_obj,map_to_terms=True):
    num_opinion =  0
    for type_exp, span_exp, span_tar, span_hol in triples:
        #Map tokens to terms       
        if map_to_terms:
            span_exp_terms = map_tokens_to_terms(span_exp,kaf_obj)
            span_tar_terms = map_tokens_to_terms(span_tar,kaf_obj)
            span_hol_terms = map_tokens_to_terms(span_hol, kaf_obj)
        else:
            span_hol_terms = span_hol
            span_tar_terms = span_tar
            span_exp_terms = span_exp
            
        ##Creating holder
        span_hol = Cspan()
        span_hol.create_from_ids(span_hol_terms)
        my_hol = Cholder()
        my_hol.set_span(span_hol)
        
        #Creating target
        span_tar = Cspan()
        span_tar.create_from_ids(span_tar_terms)
        my_tar = Ctarget()
        my_tar.set_span(span_tar)
        #########################

        ##Creating expression
        span_exp = Cspan()
        span_exp.create_from_ids(span_exp_terms)
        my_exp = Cexpression()
        my_exp.set_span(span_exp)
        my_exp.set_polarity(type_exp)
        my_exp.set_strength("1")
        #########################
        
        new_opinion = Copinion(type=knaf_obj.get_type())
        new_opinion.set_id('o'+str(num_opinion+1))
        new_opinion.set_holder(my_hol)
        new_opinion.set_target(my_tar)
        new_opinion.set_expression(my_exp)
        num_opinion += 1
        
        knaf_obj.add_opinion(new_opinion)
        
        
if __name__ == '__main__':
    config_file = sys.argv[1]
    my_config_manager.set_current_folder(__this_folder)
    my_config_manager.set_config(config_file)
    
   
    #Create a temporary file
    out_feat_file, err_feat_file, knaf_obj = extract_features(sys.stdin)
    
    #get all the tokens in order
    list_token_ids = []
    sentence_for_token = {}
    for token_obj in knaf_obj.get_tokens():
        token = token_obj.get_text()
        s_id = token_obj.get_sent()
        w_id = token_obj.get_id()
         
        list_token_ids.append(w_id)
        sentence_for_token[w_id] = s_id

       
    expressions = detect_expressions(out_feat_file,list_token_ids)
    targets = detect_targets(out_feat_file, list_token_ids)
    holders = detect_holders(out_feat_file, list_token_ids)
    
    # Entity linker based on distances
    ####triples = link_entities_distance(expressions,targets,holders,sentence_for_token)
    
    triples = link_entities_svm(expressions, targets, holders, knaf_obj, my_config_manager)
    knaf_obj.remove_opinion_layer()
    
    add_opinions_to_knaf(triples, knaf_obj,map_to_terms=False)   
    
    #Adding linguistic processor
    my_lp = Clp()
    my_lp.set_name(__desc)
    my_lp.set_version(__last_edited+'_'+__version)
    my_lp.set_timestamp()   ##Set to the current date and time
    knaf_obj.add_linguistic_processor('opinions',my_lp)
    knaf_obj.dump(sys.stdout)
    sys.exit(0)
    
    
    
    
    