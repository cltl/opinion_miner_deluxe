#!/usr/bin/env python

import sys
import codecs
import csv
import os
from operator import itemgetter

from VUA_pylib.lexicon import MPQA_subjectivity_lexicon


def get_first_term_id(token_data,term_data,this_ids):
    vector_tid_pos = []
    for tid in this_ids:
        span_token = term_data[tid][2]
        min_token_pos = min(token_data[tok_id][2] for tok_id in span_token)
        vector_tid_pos.append((tid,min_token_pos))
    vector_tid_pos.sort(key=itemgetter(1))
    return vector_tid_pos[0][0]


def get_mapping_from_lexicon(token_ids,lexicon):
    #Create index offset --> ids
    idx = 0
    my_map = {}
    text = ' '
    for token, tid in token_ids:
        for c in token:
            my_map[idx] = tid
            idx+=1
        text += token+' '
        idx+=1
    ####
    all_extracted = [] # List of [(ids,polarity), (ids, polarity)
    
    
    for substring, polarity in lexicon.items():
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
            
    final_selected = {}
    
    #If w15 has been selected first, for instance (w14,w15,w16) will not be selected later in this file
    for ids,polarity in sorted(all_extracted, key=lambda t: len(t[0])):
        already_selected = False
        for this_id in ids:
            if this_id in final_selected:
                already_selected = True
        
        if not already_selected:
            for this_id in ids:
                final_selected[this_id] = polarity
    return final_selected 

    
def load_propagation_lexicon(propagation_lex_filename):
    ##Creates a lexicon (map) [lemma] --> polarity
    propagated_lexicon = {}
    if not os.path.exists(propagation_lex_filename):
        print>>sys.stderr,'The propagated lexicon on', propagation_lex_filename,'does not exist'
    else:
        fic = open(propagation_lex_filename,'r')
        for line in fic:
            line = line.decode('utf-8').rstrip()
            tokens = line.split(';')
            lemma = tokens[4]
            polarity = tokens[2]
            propagated_lexicon[lemma] = polarity
    return propagated_lexicon
            
    
#def extract_features_from_kaf_naf_file(knaf_obj,out_file=None,log_file=None,include_class=True,accepted_opinions=None, exp_lex= None, tar_lex=None, propagation_lex_filename=None):
def extract_features_from_kaf_naf_file(knaf_obj,out_file=None,log_file=None,include_class=True,accepted_opinions=None, lexicons=[]):
    
    labels = []
    
    polarities_found_and_skipped = []
    separator = '\t'
    restore_out = None
    log_on = False
    
    if log_file is not None:
        log_desc = codecs.open(log_file, 'w', encoding='UTF-8')
        log_on = True
    
    if out_file is not None:
        restore_out = sys.stdout
        sys.stdout = open(out_file,'a')
            
        
    
    print>>log_desc,'Extracting features from ',knaf_obj.get_filename()
    
    
    
    ###########################
    ## EXTRACTING TOKENS #######
    token_data = {} ## token_data['w_1'] = ('house','s_1')
    tokens_in_order = []
    num_token = 0 
    tokens_ids = []
    for token_obj in knaf_obj.get_tokens(): 
        token = token_obj.get_text()
        s_id = token_obj.get_sent()
        w_id = token_obj.get_id()
        tokens_ids.append((token,w_id))
        token_data[w_id] = (token,s_id,num_token)
        tokens_in_order.append(w_id)
        num_token += 1
    if log_on:
        print>>log_desc,'  Number of tokens: ',len(tokens_in_order)
    ###########################
    
    #We need to create the mappings
    for lexicon in lexicons:
        lexicon.create_mapping_for_tokenids(tokens_ids)
        
        
    ###########################
    ## EXTRACTING TERMS #######
    term_data = {}  #(term_lemma,term_pos,term_span,polarity)
    term_for_token = {}
    sentence_for_term = {}
    for term_obj in knaf_obj.get_terms():
        term_id = term_obj.get_id()
        term_lemma = term_obj.get_lemma()
        term_pos = term_obj.get_morphofeat()
        # if there is no morphofeat feature, we try to get the pos from the 'pos' attrib
        if term_pos == None:
            term_pos = term_obj.get_pos()
        if term_pos is not None:
            term_pos = term_pos.split(' ')[0] #[:2]  ## Only the 2 first chars of the pos string
        else:
            term_pos = 'unknown'
        
          
        term_span = term_obj.get_span().get_span_ids()

        sentiment = term_obj.get_sentiment()
        polarity = None
        if sentiment is not None:
          polarity = sentiment.get_polarity()
          if polarity is None:
            modifier = sentiment.get_modifier()
            polarity = modifier
        if polarity is None:  polarity='-'
          
        term_data[term_id] = (term_lemma,term_pos,term_span,polarity)
        for tok_id in term_span:
            term_for_token[tok_id] = term_id
        
        if tok_id in token_data:
            sentence_id = token_data[tok_id][1]
            sentence_for_term[term_id] = sentence_id
        else:
            sentence_for_term[term_id] = '0'
            
    if log_on:
        print>>log_desc,'  Number of terms loaded: '+str(len(term_data))
    ###########################
    
    ###########################
    # EXTRACTING ENTITIES FOR EACH TERM
    ###########################   
    entity_for_term = {}
    for ent_obj in knaf_obj.get_entities():
        ent_type = ent_obj.get_type()
        for reference_obj in ent_obj.get_references():
            for span_obj in reference_obj:
                for t_id in span_obj.get_span_ids():
                    entity_for_term[t_id] = ent_type
    if log_on:
        print>>log_desc,'Entities:'+str(entity_for_term)

    ###########################
    # EXTRACTING PROPERTIES FOR EACH TERM
    ###########################  
    property_for_term = {}
    for prop_obj in knaf_obj.get_properties():
        prop_type = prop_obj.get_type()
        for reference_obj in prop_obj.get_references():
            for span_obj in reference_obj:
                for t_id in span_obj.get_span_ids():
                    property_for_term[t_id] = prop_type
    if log_on:
        print>>log_desc,'Properties:'+str(property_for_term)

    ###########################
    # EXTRACTING CLASS FOR EACH TERM
    ###########################
    class_for_term_id = {}
    if include_class:
        for opinion in knaf_obj.get_opinions():
            ## opinion expression
            opinion_id = opinion.get_id()
            opinion_exp = opinion.get_expression()
            exp_type = ''
            exp_strength = ''
            exp_ids = []
            if opinion_exp is not None:
                exp_type = opinion_exp.get_polarity()
                exp_strength = opinion_exp.get_strength()
                span = opinion_exp.get_span()
                if span is not None:
                    exp_ids = span.get_span_ids()
                    
            opinion_hol = opinion.get_holder()
            hol_ids = []
            if opinion_hol is not None:
                span = opinion_hol.get_span()
                if span is not None:
                    hol_ids = span.get_span_ids()
                    
            opinion_tar = opinion.get_target()
            tar_ids = []
            if opinion_tar is not None:
                span = opinion_tar.get_span()
                if span is not None:
                    tar_ids = span.get_span_ids()
            
            ############################
            
            if accepted_opinions is not None:
                if '*' in accepted_opinions:    ##All polarities are valid, in config there is something like dse = *
                    mapped_type = accepted_opinions['*']
                elif exp_type in accepted_opinions:
                    #Get the mapping label
                    mapped_type = accepted_opinions[exp_type]
                else:
                    # This opinion wont be considered
                    polarities_found_and_skipped.append(exp_type)
                    continue    
            else:
                mapped_type = exp_type
            
            
            if log_on:
                print>>log_desc,'  Opinion',opinion_id
                print>>log_desc,'    Expression:'
                print>>log_desc,'      ids:',exp_ids
                print>>log_desc,'      terms:',[term_data[i][0] for i in exp_ids]
                
            if len(exp_ids) != 0:
                first_term_id = get_first_term_id(token_data,term_data,exp_ids)
                for t_id in exp_ids:
                    if t_id == first_term_id:  type='B-'
                    else:  type='I-'
                    class_for_term_id[t_id]=type+mapped_type
                    
            
            
            if log_on:
                print>>log_desc,'    Target:'
                print>>log_desc,'      ids:',tar_ids
                print>>log_desc,'      terms:',[term_data[i][0] for i in tar_ids]
                
            if len(tar_ids) != 0:
                first_term_id = get_first_term_id(token_data,term_data,tar_ids)
                for t_id in tar_ids:
                    if t_id == first_term_id:  type='B-'
                    else:  type='I-'
                    class_for_term_id[t_id]=type+'target'        
            
            if log_on:
                print>>log_desc,'    Holder:'
                print>>log_desc,'      ids:',hol_ids
                print>>log_desc,'      terms:',[term_data[i][0] for i in hol_ids]
                
            if len(hol_ids) != 0:
                first_term_id = get_first_term_id(token_data,term_data,hol_ids)
                for t_id in hol_ids:
                    if t_id == first_term_id:  type='B-'
                    else:  type='I-'
                    class_for_term_id[t_id]=type+'holder'    
        ##############
            
            
    my_mpqa_subj_lex = MPQA_subjectivity_lexicon()
    ## WRITE TO THE OUTPUT
    
    
   
    
   
    prev_sent = None
    for token_id in tokens_in_order:
        token,sentence_id,num_token = token_data[token_id]
        
        term_id = term_for_token.get(token_id,None)
        
        #This is required for wrong KAF files that contain missing terms (tokens not linked with terms)
        if term_id is not None:
            print>>log_desc,'Processing data',term_id
            data = term_data.get(term_id,None)
            if data is not None:
                term_lemma,term_pos,term_span,polarity = data
                entity = entity_for_term.get(term_id,'-')
                property = property_for_term.get(term_id,'-')
                this_class = class_for_term_id.get(term_id,'O')
                
                
                #Mpqa subjectivy from the mpqa corpus
                mpqa_subj = mpqa_pol = '-'
                if my_mpqa_subj_lex is not None:
                    mpqa_data = my_mpqa_subj_lex.get_type_and_polarity(token,term_pos)
                    if mpqa_data is not None:
                        mpqa_subj, mpqa_pol = mpqa_data
                               
                                
                                
                ## Constituency features
                constituency_extractor = knaf_obj.get_constituency_extractor()
                feature_phrase = 'XXX'
                if constituency_extractor is not None:
                    this_phrase, subsumed_together = constituency_extractor.get_deepest_phrase_for_termid(term_id)
                    if this_phrase is not None:
                        feature_phrase = this_phrase
                ######################
                                  
                                  
                lexicon_features = []
                for lexicon in lexicons:
                    value = '-'
                    if lexicon.is_lemma_based():
                        value = lexicon.get_value_for_lemma(term_lemma)
                    elif lexicon.is_multiword_based():
                        value = lexicon.get_value_for_tokenid(token_id)
                    if value == None:
                        value = '-'
                    lexicon_features.append((lexicon.get_label(),value))
                        
                ### Expression from the domain lexicon
                #polarity_from_domain = mapping_wid_polarity.get(token_id,'-')
                
                ## Polarity from the propagated lexicon
                #polarity_from_propagation = propagated_lex.get(term_lemma,'-')
                
                ## Target from the training lexicon
                #aspect_from_domain = mapping_wid_aspect.get(token_id,'-')
                
                ##############################################################################################
                ## FEATURE GENERATION!!!!
                ##############################################################################################
                labels =   ['sentence_id','token_id','token','lemma',    'pos',    'term_id', 'pol/mod']
                features = [ sentence_id,  token_id,  token,  term_lemma, term_pos, term_id,   polarity ]
                
                for label, value in lexicon_features:
                    labels.append(label)
                    features.append(value)
                
                
                features.append(mpqa_subj)
                labels.append('mpqa_subj')
                
                features.append(mpqa_pol)
                labels.append('mpqa_pol')
                
                labels.extend(['entity','property','phrase_type','y'])
                features.extend([entity,property,feature_phrase,this_class])
                
                ##############################################################################################
                ##############################################################################################
                
                
        if prev_sent is not None and sentence_id != prev_sent: print>>sys.stdout    #breakline 
        print>>sys.stdout,(separator.join(features)).encode('utf-8')
        
        prev_sent=sentence_id
    print>>sys.stdout   #Last breakline required for crfsuite
    

    print>>log_desc
    ## Restoring
    if log_on:
        log_desc.close()

    if restore_out is not None:
        sys.stdout.close()
        sys.stdout = restore_out
        
    return labels, separator, polarities_found_and_skipped

