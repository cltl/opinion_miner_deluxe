#!/usr/bin/env python

import sys

def write_to_output(my_class,feats, output):
    my_str = my_class
    for name, value in feats:
        my_str += '\t'+name+'='+value
    output.write(my_str.encode('utf-8')+'\n')
    
    
    
#########################################################################   
# EXTRACTION OF FEATURES FOR TRAINING THE RELATION CLASSIFIER EXP --> TARGET
#########################################################################  
# This function extracts features for the relation between expression adn target
# for the svm classifier
def extract_feats_exp_tar(exp_ids,tar_ids,knaf_obj, use_lemmas=True, use_tokens=True, use_dependencies=True):
    all_feats = []
    use_lemmas = use_tokens = False
    
    data_for_token = {}     # [token_id] -> (word, sentence_id)
    for num_token, token_obj in enumerate(knaf_obj.get_tokens()):
        word = token_obj.get_text()
        s_id = token_obj.get_sent()
        w_id = token_obj.get_id()
        
        data_for_token[w_id] = (word,s_id,num_token)
    
    # Loading data for terms
    data_for_term = {}      # [term_id] -> (lemma, span_token_ids)
    for term in knaf_obj.get_terms():
        termid = term.get_id()
        lemma = term.get_lemma()
        span = term.get_span()
        span_token_ids = []
        if span is not None:
            span_token_ids = span.get_span_ids()
        data_for_term[termid] = (lemma,span_token_ids)

    sentence_for_exp = None
    avg_position_exp = 0
    n_toks = 0
    for my_id in exp_ids:
        lemma, span_tok_ids = data_for_term[my_id]
        if use_lemmas:
            all_feats.append(('lemmaExp',lemma))
                
                
        for tok_id in span_tok_ids:
            token,sent_id,num_token = data_for_token[tok_id]
            avg_position_exp += num_token
            n_toks += 1
            if use_tokens:
                all_feats.append(('tokenExp',token))
            
            if sentence_for_exp is None:
                sentence_for_exp = sent_id
                
    avg_position_exp = avg_position_exp * 1.0 / n_toks  

    #Lemmas for target    
    sentence_for_tar = None
    avg_position_tar = 0
    n_toks = 0
    for my_id in tar_ids:
        lemma, span_tok_ids = data_for_term[my_id]
        if use_lemmas:
            all_feats.append(('lemmaTar',lemma))
                
        for tok_id in span_tok_ids:
            token,sent_id,num_token = data_for_token[tok_id]
            avg_position_tar += num_token
            n_toks += 1
            if use_tokens:
                all_feats.append(('tokenTar',token))

            if sentence_for_tar is None:
                sentence_for_tar = sent_id
    
    avg_position_tar = avg_position_tar * 1.0 / n_toks  
    
    if use_dependencies:
        dependency_extractor = knaf_obj.get_dependency_extractor()
        if dependency_extractor is not None:
            deps = dependency_extractor.get_shortest_path_spans(exp_ids,tar_ids)
            if deps is not None:
                all_feats.append(('deps-exp-tar','#'.join(deps)))
    
  
    if sentence_for_exp is not None and sentence_for_tar is not None and sentence_for_exp == sentence_for_tar:
        all_feats.append(('same_sentence','yes'))
    else:
        all_feats.append(('same_sentence','no'))
        
    ##Distance
    dist = abs(avg_position_exp - avg_position_tar)
    if dist <= 10:
        my_dist = 'veryclose'
    elif dist <=20:
        my_dist  = 'close'
    elif dist <=25:
        my_dist = 'far'
    else:
        my_dist = 'veryfar'
    all_feats.append(('distExpTar',my_dist))
    #all_feats.append(('absDist',str(dist))) 
  
    return all_feats

    
      
  
        
    
def create_rel_exp_tar_training(knaf_obj, output=sys.stdout, valid_opinions=None,use_dependencies=True):
    # Obtain pairs of features for Expression and Target
    pairs = [] # [(Exp,Tar), (E,T), (E,T)....]
    for opinion in knaf_obj.get_opinions():
        opi_id = opinion.get_id()
        opi_exp = opinion.get_expression()
        exp_type = ''
        exp_ids = []
        if opi_exp is not None:
            exp_type = opi_exp.get_polarity()
            span = opi_exp.get_span()
            if span is not None:
                exp_ids = span.get_span_ids()

        opi_tar = opinion.get_target()
        tar_ids = []
        if opi_tar is not None:
            span = opi_tar.get_span()
            if span is not None:
                tar_ids = span.get_span_ids()
        
       
        if valid_opinions is not None:
            if exp_type not in valid_opinions:
                continue    ## This opinions will not be used
        
        
        if len(tar_ids) != 0 and len(exp_ids) != 0:
            pairs.append((exp_ids,tar_ids))

            
    for idx1, (exp1, tar1) in enumerate(pairs):
        feats_positive = extract_feats_exp_tar(exp1,tar1,knaf_obj,use_dependencies)
        write_to_output('+1', feats_positive, output)
        for idx2, (exp2, tar2) in enumerate(pairs):
            if idx1 != idx2:
                feats_negative = extract_feats_exp_tar(exp1,tar2,knaf_obj,use_dependencies)
                write_to_output('-1', feats_negative, output)
                      
    




def extract_feats_exp_hol(exp_ids,hol_ids,knaf_obj, use_lemmas=True, use_tokens=True, use_dependencies=True):
    all_feats = []
    use_lemmas = use_tokens = False
    
    data_for_token = {}     # [token_id] -> (word, sentence_id)
    for num_token, token_obj in enumerate(knaf_obj.get_tokens()):
        word = token_obj.get_text()
        s_id = token_obj.get_sent()
        w_id = token_obj.get_id()
        
        data_for_token[w_id] = (word,s_id,num_token)
    
    # Loading data for terms
    data_for_term = {}      # [term_id] -> (lemma, span_token_ids)
    for term in knaf_obj.get_terms():
        termid = term.get_id()
        lemma = term.get_lemma()
        span = term.get_span()
        span_token_ids = []
        if span is not None:
            span_token_ids = span.get_span_ids()
        data_for_term[termid] = (lemma,span_token_ids)

    sentence_for_exp = None
    avg_position_exp = 0
    n_toks = 0
    for my_id in exp_ids:
        lemma, span_tok_ids = data_for_term[my_id]
        if use_lemmas:
            all_feats.append(('lemmaExp',lemma))
                
                
        for tok_id in span_tok_ids:
            token,sent_id,num_token = data_for_token[tok_id]
            avg_position_exp += num_token
            n_toks += 1
            if use_tokens:
                all_feats.append(('tokenExp',token))
            
            if sentence_for_exp is None:
                sentence_for_exp = sent_id
                
    avg_position_exp = avg_position_exp * 1.0 / n_toks  

    #Lemmas for HOLDER    
    sentence_for_hol = None
    avg_position_hol = 0
    n_toks = 0
    for my_id in hol_ids:
        lemma, span_tok_ids = data_for_term[my_id]
        if use_lemmas:
            all_feats.append(('lemmaHol',lemma))
                
        for tok_id in span_tok_ids:
            token,sent_id,num_token = data_for_token[tok_id]
            avg_position_hol += num_token
            n_toks += 1
            if use_tokens:
                all_feats.append(('tokenHol',token))

            if sentence_for_hol is None:
                sentence_for_hol = sent_id
                
    avg_position_hol = avg_position_hol * 1.0 / n_toks
    
    if use_dependencies:
        dependency_extractor = knaf_obj.get_dependency_extractor()
        if dependency_extractor is not None:
            deps = dependency_extractor.get_shortest_path_spans(exp_ids,hol_ids)
            if deps is not None:
                all_feats.append(('deps-exp-hol','#'.join(deps)))
    
  
    if sentence_for_exp is not None and sentence_for_hol is not None and sentence_for_exp == sentence_for_hol:
        all_feats.append(('same_sentence','yes'))
    else:
        all_feats.append(('same_sentence','no'))
        
    ##Distance
    dist = abs(avg_position_exp - avg_position_hol)
    if dist <= 10:
        my_dist = 'veryclose'
    elif dist <=20:
        my_dist  = 'close'
    elif dist <=25:
        my_dist = 'far'
    else:
        my_dist = 'veryfar'
    all_feats.append(('distExpHol',my_dist))
    #all_feats.append(('absDist',str(dist))) 
  
    return all_feats
  
                
    
def create_rel_exp_hol_training(knaf_obj, output=sys.stdout, valid_opinions=None,use_dependencies=True):
       
    # Obtain pairs of features for Expression and Holder
    pairs = [] # [(Exp,Hol), (E,H), (E,H)....]
    for opinion in knaf_obj.get_opinions():
        opi_exp = opinion.get_expression()
        exp_type = ''
        exp_ids = []
        if opi_exp is not None:
            exp_type = opi_exp.get_polarity()
            span = opi_exp.get_span()
            if span is not None:
                exp_ids = span.get_span_ids()

        opi_hol = opinion.get_holder()
        hol_ids = []
        if opi_hol is not None:
            span = opi_hol.get_span()
            if span is not None:
                hol_ids = span.get_span_ids()
                
        
        if valid_opinions is not None:
            if exp_type not in valid_opinions:
                continue    ## This opinions will not be used
        
        
        if len(exp_ids) != 0 and len(hol_ids) != 0:
            pairs.append((exp_ids,hol_ids))
            
            
    #for feat_exp, feat_tar
    for idx1, (expids1, tarids1) in enumerate(pairs):
        
        feats_positive = extract_feats_exp_hol(expids1,tarids1,knaf_obj, use_dependencies=use_dependencies)
        write_to_output('+1', feats_positive,output)
        
        for idx2, (expids2, tarids2) in enumerate(pairs):
            if idx1 != idx2:
                feats_negative = extract_feats_exp_hol(expids1,tarids2,knaf_obj, use_dependencies=use_dependencies)
                write_to_output('-1', feats_negative ,output)
    
    
