#!/usr/bin/env python

import sys

def write_to_output(my_class,feats1, feats2, extra, output):
    my_str = my_class
    for name, value in feats1+feats2+extra:
        my_str += '\t'+name+'='+value
    output.write(my_str.encode('utf-8')+'\n')
    
    
    
#########################################################################   
# EXTRACTION OF FEATURES FOR TRAINING THE RELATION CLASSIFIER EXP --> TARGET
#########################################################################  
# This function extracts features for the relation between expression adn target
# for the svm classifier
def extract_feats_exp_tar(exp_ids,tar_ids,knaf_obj):
    feats_for_exp = []
    feats_for_tar = []
    
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
        feats_for_exp.append(('lemmaExp',lemma))
                
                
        for tok_id in span_tok_ids:
            token,sent_id,num_token = data_for_token[tok_id]
            avg_position_exp += num_token
            n_toks += 1
            
            feats_for_exp.append(('tokenExp',token))
            
            if sentence_for_exp is None:
                sentence_for_exp = sent_id
                
    avg_position_exp = avg_position_exp * 1.0 / n_toks  

    #Lemmas for target    
    sentence_for_tar = None
    avg_position_tar = 0
    n_toks = 0
    for my_id in tar_ids:
        lemma, span_tok_ids = data_for_term[my_id]
        feats_for_tar.append(('lemmaTar',lemma))
                
        for tok_id in span_tok_ids:
            token,sent_id,num_token = data_for_token[tok_id]
            avg_position_tar += num_token
            n_toks += 1
            feats_for_tar.append(('tokenTar',token))

            if sentence_for_tar is None:
                sentence_for_tar = sent_id
    
    ## Dependency relations
    dependency_extractor = knaf_obj.get_dependency_extractor()
    if dependency_extractor is not None:
        #For expression
        deps_from_exp_to_root = dependency_extractor.get_shortest_path_to_root_span(exp_ids)
        if deps_from_exp_to_root is not None:
            if len(deps_from_exp_to_root) == 0:  #one term is the root of the sentence
                deps_from_exp_to_root = ['IS_ROOT']
        
            feats_for_exp.append(('first-dependency-exp',deps_from_exp_to_root[0]))
            feats_for_exp.append(('chain-dependency-exp','#'.join(deps_from_exp_to_root)))
    
        ##For target
        deps_from_tar_to_root = dependency_extractor.get_shortest_path_to_root_span(tar_ids)
        if deps_from_tar_to_root is not None:
            if len(deps_from_tar_to_root) == 0:
                deps_from_tar_to_root = ['IS_ROOT']
            
            feats_for_tar.append(('first-dependency-tar',deps_from_tar_to_root[0]))
            feats_for_tar.append(('chain-dependency-tar','#'.join(deps_from_tar_to_root))) 
    
    ## EXTRA FEATURES
    ## This will be used to establish a relation between expression and target, it is not a feature for one single entity

    avg_position_tar = avg_position_tar * 1.0 / n_toks  
    
    extra_feats_exp = {}
    extra_feats_exp['sent'] = sentence_for_exp
    extra_feats_exp['avg_position'] = avg_position_exp
    extra_feats_exp['total_tokens'] = len(data_for_token)
        
    extra_feats_tar = {}
    extra_feats_tar['sent'] = sentence_for_tar
    extra_feats_tar['avg_position'] = avg_position_tar
            
            
    return feats_for_exp,feats_for_tar, extra_feats_exp, extra_feats_tar
    
def get_extra_feats_exp_tar(extra_e, extra_t):
    ## Obtaining if both are in the same sentence or not
    feats = []
    sent_e = extra_e.get('sent')
    sent_t = extra_t.get('sent')
    if sent_e is not None and sent_t is not None and sent_e == sent_t :
        feats.append(('same_sentence','yes'))
    else:
        feats.append(('same_sentence','no'))
    ############################################
    
    
    ## Obtaining the "normalized" distance from expression to target    
    avg_e = extra_e.get('avg_position')
    avg_t = extra_t.get('avg_position')
    total_tokens = extra_e.get('total_tokens')
    dist = abs(avg_e - avg_t)
    my_dist = None
    if dist <= total_tokens *10.0 / 100:
        my_dist = 'veryclose'
    elif dist <= total_tokens * 20.0 / 100:
        my_dist = 'close'
    elif dist <= total_tokens * 50.0 / 100:
        my_dist = 'medium'
    elif dist <= total_tokens * 75.0 / 100:
        my_dist = 'far'
    else:
        my_dist = 'superfar'
    feats.append(('distExpTar',my_dist))
    return feats
      
  
        
    
def create_rel_exp_tar_training(knaf_obj, output=sys.stdout, valid_opinions=None):
    
   
    # Obtain pairs of features for Expression and Target
    pairs = [] # [(Exp,Tar), (E,T), (E,T)....]
    for opinion in knaf_obj.get_opinions():
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
            feats_for_exp, feats_for_tar, extra_feats_exp, extra_feats_tar = extract_feats_exp_tar(exp_ids,tar_ids,knaf_obj)              
            pairs.append((feats_for_exp,feats_for_tar, extra_feats_exp, extra_feats_tar))
            
    #for feat_exp, feat_tar
    for idx1, (e1, t1,extra_e1, extra_t1) in enumerate(pairs):
        
        ## Same sentence
        extra_feats1 = get_extra_feats_exp_tar(extra_e1, extra_t1)
        
        write_to_output('+1', e1, t1,extra_feats1, output)
        for idx2, (e2, t2, extra_e2, extra_t2) in enumerate(pairs):
            if idx1 != idx2:
                extra_feats2 = get_extra_feats_exp_tar(extra_e1, extra_t2)
                write_to_output('-1', e1, t2,extra_feats2, output)              
    


#########################################################################   
# EXTRACTION OF FEATURES FOR TRAINING THE RELATION CLASSIFIER EXP --> HOL
#########################################################################   
def get_extra_feats_exp_hol(extra_e, extra_h):
    ## Obtaining if both are in the same sentence or not
    feats = []
    sent_e = extra_e.get('sent')
    sent_h = extra_h.get('sent')
    if sent_e is not None and sent_h is not None and sent_e == sent_h :
        feats.append(('same_sentence','yes'))
    else:
        feats.append(('same_sentence','no'))
    ############################################
    
    
    ## Obtaining the "normalized" distance from expression to target    
    avg_e = extra_e.get('avg_position')
    avg_h = extra_h.get('avg_position')
    total_tokens = extra_e.get('total_tokens')
    dist = abs(avg_e - avg_h)
    my_dist = None
    if dist <= total_tokens *10.0 / 100:
        my_dist = 'veryclose'
    elif dist <= total_tokens * 20.0 / 100:
        my_dist = 'close'
    elif dist <= total_tokens * 50.0 / 100:
        my_dist = 'medium'
    elif dist <= total_tokens * 75.0 / 100:
        my_dist = 'far'
    else:
        my_dist = 'superfar'
    feats.append(('distExpHol',my_dist))
    return feats  

def extract_feats_exp_hol(exp_ids,hol_ids,knaf_obj):
    feats_for_exp = []
    feats_for_hol = []

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
        feats_for_exp.append(('lemmaExp',lemma))
                
        for tok_id in span_tok_ids:
            
            token,sent_id,num_token = data_for_token[tok_id]
            avg_position_exp += num_token
            n_toks += 1
            feats_for_exp.append(('tokenExp',token))
            if sentence_for_exp is None:
                sentence_for_exp= sent_id
    avg_position_exp = avg_position_exp * 1.0 / n_toks 

    #Lemmas for holder    
    sentence_for_hol = None
    avg_position_hol = 0
    n_toks = 0
    for my_id in hol_ids:
        lemma, span_tok_ids = data_for_term[my_id]
        feats_for_hol.append(('lemmaHol',lemma))
                
        for tok_id in span_tok_ids:
            token,sent_id,num_token = data_for_token[tok_id]
            avg_position_hol += num_token
            n_toks += 1
            feats_for_hol.append(('tokenHol',token))
            if sentence_for_hol is None:
                sentence_for_hol = sent_id
                
                
    ## Dependency relations
    dependency_extractor = knaf_obj.get_dependency_extractor()
    #For expression
    if dependency_extractor is not None:
        deps_from_exp_to_root = dependency_extractor.get_shortest_path_to_root_span(exp_ids)
        if deps_from_exp_to_root is not None:
            if len(deps_from_exp_to_root) == 0:  #one term is the root of the sentence
                deps_from_exp_to_root = ['IS_ROOT']
        
            feats_for_exp.append(('first-dependency-exp',deps_from_exp_to_root[0]))
            feats_for_exp.append(('chain-dependency-exp','#'.join(deps_from_exp_to_root)))
    
        ##For HOLDER
        #print>>sys.stderr,'HOL_IDS', hol_ids
        deps_from_hol_to_root = dependency_extractor.get_shortest_path_to_root_span(hol_ids)
        if deps_from_hol_to_root is not None:
            if len(deps_from_hol_to_root) == 0:
                deps_from_hol_to_root = ['IS_ROOT']
            
            feats_for_hol.append(('first-dependency-hol',deps_from_hol_to_root[0]))
            feats_for_hol.append(('chain-dependency-hol','#'.join(deps_from_hol_to_root))) 
    #################        
            
             
    avg_position_hol = avg_position_hol * 1.0 / n_toks 
    
    extra_feats_exp = {}
    extra_feats_exp['sent'] = sentence_for_exp
    extra_feats_exp['avg_position'] = avg_position_exp
    extra_feats_exp['total_tokens'] = len(data_for_token)
        
    extra_feats_hol = {}
    extra_feats_hol['sent'] = sentence_for_hol
    extra_feats_hol['avg_position'] = avg_position_hol
    return feats_for_exp,feats_for_hol, extra_feats_exp, extra_feats_hol    
                
    
def create_rel_exp_hol_training(knaf_obj, output=sys.stdout, valid_opinions=None):
    
   
    
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
            feats_for_exp, feats_for_hol, extra_feats_exp, extra_feats_hol  = extract_feats_exp_hol(exp_ids,hol_ids,knaf_obj)              
            pairs.append((feats_for_exp,feats_for_hol, extra_feats_exp, extra_feats_hol))
            
    #for feat_exp, feat_tar
    for idx1, (e1, h1, extra_e1, extra_h1) in enumerate(pairs):
        extra_feats1 = get_extra_feats_exp_hol(extra_e1, extra_h1)
        write_to_output('+1', e1, h1,extra_feats1,output)
        for idx2, (e2, h2, extra_e2, extra_h2) in enumerate(pairs):
            if idx1 != idx2:
                extra_feats2 = get_extra_feats_exp_hol(extra_e1, extra_h2)
                write_to_output('-1', e1, h2,extra_feats2,output)
    #return dependency_extractor
    
    
