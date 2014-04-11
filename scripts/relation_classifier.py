#!/usr/bin/env python

from extract_feats_relations import *
from tempfile import NamedTemporaryFile
from subprocess import Popen, PIPE
from VUA_pylib.io import Cfeature_index
import os

config_manager = None

    
def link_exp_tar(expressions,targets, knaf_obj,use_dependencies=True):
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
                feats = extract_feats_exp_tar(exp_ids,tar_ids,knaf_obj, use_dependencies=use_dependencies)
                feat_index.encode_example_for_classification(feats, examples_file,my_class='0')
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
            best_value = -100
            best_idx = -100
            #print>>sys.stderr,' Exp:', exp
            for num_tar , tar in enumerate(targets):
                
                #This is the probably of exp to be related with the target num_tar
                value = results[idx]
                #print>>sys.stderr,'  Target:',tar
                #print>>sys.stderr,'      Value:', value
                #print>>sys.stderr, exp
                #print>>sys.stderr, tar
                #print>>sys.stderr, num_tar, value
                #print
                
                #We select the best among the targets for the exp processed
                if value > best_value:
                    best_value = value
                    best_idx = num_tar
                idx += 1
            selected.append((best_idx,best_value))
            #print>>sys.stderr,'  Selected:', targets[best_idx]
        #print selected
        
        for best_tar_idx, best_value in selected:
            assigned_targets.append(targets[best_tar_idx])
            #print>>sys.stderr,  'SELECTED',best_tar_idx,targets[best_tar_idx]
        os.remove(examples_file.name)                
    return assigned_targets

def link_exp_tar_all(expressions,targets, knaf_obj,use_dependencies=True):
    pairs = []

    if len(targets) == 0:
        for exp_ids, exp_type in expressions:
            pairs.append((exp_ids,exp_type,[]))
    else:
        feat_index_filename = config_manager.get_index_features_exp_tar_filename()
        feat_index = Cfeature_index()
        feat_index.load_from_file(feat_index_filename)
        examples_file = NamedTemporaryFile(delete=False)
        for exp_ids, exp_type in expressions:
            for tar_ids in targets:
                feats = extract_feats_exp_tar(exp_ids,tar_ids,knaf_obj, use_dependencies=use_dependencies)
                feat_index.encode_example_for_classification(feats, examples_file,my_class='0')
        examples_file.close()
         
        model_file = config_manager.get_filename_model_exp_tar()
        results = run_svm_classify(examples_file.name, model_file)
        
        
        threshold = -0.75
        idx = 0
        for exp,exp_type in expressions:
            at_least_one = False
            for num_tar, tar in enumerate(targets):
                value = results[idx]
                idx += 1
                if value >= threshold:
                    pairs.append((exp,exp_type,tar))
                    at_least_one = True
            
            if not at_least_one:
                pairs.append((exp,exp_type,[]))
 
        os.remove(examples_file.name)                
    return pairs

def link_exp_hol(expressions,holders, knaf_obj,use_dependencies=True):
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
                feats = extract_feats_exp_hol(exp_ids,hol_ids,knaf_obj, use_dependencies=use_dependencies)
                feat_index.encode_example_for_classification(feats,examples_file,my_class='0')
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
                
        os.remove(examples_file.name)
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
        exp_term_ids = knaf_obj.map_tokens_to_terms(exp_ids)
        all_exp_ids.append((exp_term_ids, exp_type))
    
    for tar_ids, tar_type in targets:
        tar_term_ids = knaf_obj.map_tokens_to_terms(tar_ids)
        all_tar_ids.append(tar_term_ids)

    for hol_ids, hol_type in holders:
        hol_term_ids = knaf_obj.map_tokens_to_terms(hol_ids)
        all_hol_ids.append(hol_term_ids)
    
    #assigned_targets = link_exp_tar(all_exp_ids, all_tar_ids,knaf_obj)
    pairs_exp_tar = link_exp_tar_all(all_exp_ids, all_tar_ids, knaf_obj)
    
    results = []
    for exp_ids, exp_type, tar_ids in pairs_exp_tar:
        results.append((exp_type,exp_ids,tar_ids,[]))
    return results
    
    #assigned_holders = link_exp_hol(all_exp_ids, all_hol_ids, knaf_obj)
    

    
    results = []
    for index, exp_type in enumerate(all_types):
        results.append((exp_type,all_exp_ids[index], assigned_targets[index],  assigned_holders[index]))
    del config_manager
    config_manager = None
    return results

 
