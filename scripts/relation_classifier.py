#!/usr/bin/env python

from extract_feats_relations import *
from tempfile import NamedTemporaryFile
from subprocess import Popen, PIPE
from VUA_pylib.io import Cfeature_index
import os

config_manager = None
terms_for_token = None

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

    
def link_exp_tar(expressions,targets, knaf_obj):
    assigned_targets = []  #     (expression_type, exp_ids, 

    if len(targets) == 0:
        for exp_ids in expressions:
            assigned_targets.append([])
    elif len(targets) == 1:
        for exp_ids in expressions:
            assigned_targets.append(targets[0])
    else:
        feat_index_filename = config_manager.get_index_features_exp_tar_filename()
        feat_index = Cfeature_index()
        feat_index.load_from_file(feat_index_filename)
        examples_file = NamedTemporaryFile(delete=False)
        for exp_ids in expressions:
            for tar_ids in targets:
                feats_exp, feats_tar,extra_feats_exp, extra_feats_tar = extract_feats_exp_tar(exp_ids,tar_ids,knaf_obj)
                extra_feats = get_extra_feats_exp_tar(extra_feats_exp, extra_feats_tar)
                feat_index.encode_example_for_classification(feats_exp+feats_tar+extra_feats,examples_file,my_class='0')
        examples_file.close()
        ## In examples_file.name we can find the examples file
        
        ## The format in the example file will be:
        # exp1 --> tar1
        # exp1 --> tar2
        # exp1 --> tar3
        # exp2 --> tar1
        # exp2 --> tar2
        # exp2 --> tar3        
        
        model_file = config_manager.get_filename_model_exp_tar()
        results = run_svm_classify(examples_file.name, model_file)
        
        idx = 0         # This idx will iterate from 0 to num_exp X num_tar
        selected = []   # will stor for each exp --> (best_tar_idx, best_svm_val)
        for exp in expressions:
            #Selecting the best for this exp
            best_value = -1
            best_idx = -1
            for num_tar , tar in enumerate(targets):
                
                #This is the probably of exp to be related with the target num_tar
                value = results[idx]
                
                #print exp
                #print tar
                #print num_tar, value
                #print
                
                #We select the best among the targets for the exp processed
                if value > best_value:
                    best_value = value
                    best_idx = num_tar
                idx += 1
            selected.append((best_idx,best_value))
        #print selected
        
        for best_tar_idx, best_value in selected:
            assigned_targets.append(targets[best_tar_idx])
                
    return assigned_targets

def link_exp_hol(expressions,holders, knaf_obj):
    assigned_holders = []  #     (expression_type, exp_ids, 

    if len(holders) == 0:
        for exp_ids in expressions:
            assigned_holders.append([])
    elif len(holders) == 1:
        for exp_ids in expressions:
            assigned_holders.append(holders[0])
    else:
        feat_index_filename = config_manager.get_index_features_exp_hol_filename()
        feat_index = Cfeature_index()
        feat_index.load_from_file(feat_index_filename)
        examples_file = NamedTemporaryFile(delete=False)
        for exp_ids in expressions:
            for hol_ids in holders:
                feats_exp, feats_hol,extra_feats_exp, extra_feats_hol = extract_feats_exp_hol(exp_ids,hol_ids,knaf_obj)
                extra_feats = get_extra_feats_exp_hol(extra_feats_exp, extra_feats_hol)
                feat_index.encode_example_for_classification(feats_exp+feats_hol+extra_feats,examples_file,my_class='0')
        examples_file.close()
        ## In examples_file.name we can find the examples file
        
        ## The format in the example file will be:
        # exp1 --> hol1
        # exp1 --> hol2
        # exp1 --> hol3
        # exp2 --> hol1
        # exp2 --> hol2
        # exp2 --> hol3        
        
        model_file = config_manager.get_filename_model_exp_hol()
        results = run_svm_classify(examples_file.name, model_file)
        
        idx = 0         # This idx will iterate from 0 to num_exp X num_tar
        selected = []   # will stor for each exp --> (best_tar_idx, best_svm_val)
        for exp in expressions:
            #Selecting the best for this exp
            best_value = -1
            best_idx = -1
            for num_hol , hol in enumerate(holders):
                #This is the probably of exp to be related with the target num_tar
                value = results[idx]
                
                #We select the best among the targets for the exp processed
                if value > best_value:
                    best_value = value
                    best_idx = num_hol
                idx += 1
            selected.append((best_idx,best_value))
        #print selected
        
        for best_hol_idx, best_value in selected:
            assigned_holders.append(holders[best_hol_idx])
                
    return assigned_holders


    
def run_svm_classify(example_file,model_file):
    #usage: svm_classify [options] example_file model_file output_file
    svmlight = config_manager.get_svm_classify_binary()
    if not os.path.exists(svmlight):
        print>>sys.stderr,'SVMlight learn not found on',svmlight
        print>>sys.stderr,'Check the config filename and make sure the path is correctly set'
        print>>sys.stderr,'[svmlight]\npath_to_binary_learn = yourpathtolocalsvmlightlearn'
        sys.exit(-1)
                                                      
    cmd = [svmlight]
    cmd.append(example_file)
    cmd.append(model_file)
    tempout = NamedTemporaryFile(delete=False)
    tempout.close()
    
    cmd.append(tempout.name)
    svm_process = Popen(' '.join(cmd),stdin=PIPE, stdout=PIPE, stderr=PIPE, shell=True)
    svm_process.wait()
    str_err = svm_process.stderr.read()
    if len(str_err) != 0:
        print>>sys.stderr,'SVM light classify error '+str_err
        sys.exit(-1)
    #logging.debug('SVMlight classigfy log'+err_file)
    results = []
    fout = open(tempout.name,'r')
    for line in fout:
        results.append(float(line.strip()))
    fout.close()
    os.remove(tempout.name)
    return results
            


def link_entities_svm(expressions, targets, holders, knaf_obj,this_config_manager):
    all_types = []
    all_exp_ids = []
    all_tar_ids = []
    all_hol_ids = []
    global config_manager
    config_manager = this_config_manager
    
    for exp_ids,exp_type in expressions:
        all_types.append(exp_type)
        exp_term_ids = map_tokens_to_terms(exp_ids, knaf_obj)
        all_exp_ids.append(exp_term_ids)
    
    for tar_ids, tar_type in targets:
        tar_term_ids = map_tokens_to_terms(tar_ids, knaf_obj)
        all_tar_ids.append(tar_term_ids)

    for hol_ids, hol_type in holders:
        hol_term_ids = map_tokens_to_terms(hol_ids, knaf_obj)
        all_hol_ids.append(hol_term_ids)
    
    assigned_targets = link_exp_tar(all_exp_ids, all_tar_ids,knaf_obj)
    assigned_holders = link_exp_hol(all_exp_ids, all_hol_ids, knaf_obj)

    
    results = []
    for index, exp_type in enumerate(all_types):
        results.append((exp_type,all_exp_ids[index], assigned_targets[index],  assigned_holders[index]))
    return results

 
