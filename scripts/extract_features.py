#!/usr/bin/env python

import sys
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

def extract_features_from_kaf_naf_file(knaf_obj,out_file=None,err_file=None,include_class=True,accepted_opinions=None):
    labels = ['sentence_id','token_id','token','lemma','pos','term_id','pol/mod','mpqa_subjectivity','mpqa_polarity','entity','property','phrase_type','y']

    separator = '\t'
    restore_err = None
    restore_out = None
    
    if err_file is not None:
        restore_err = sys.stderr
        sys.stderr = open(err_file,'w')
    
    if out_file is not None:
        restore_out = sys.stdout
        sys.stdout = open(out_file,'a')
            
        
    
    print>>sys.stderr,'Extracting features from ',knaf_obj.get_filename()
    
    
    
    ###########################
    ## EXTRACTING TOKENS #######
    token_data = {} ## token_data['w_1'] = ('house','s_1')
    tokens_in_order = []
    num_token = 0 
    for token_obj in knaf_obj.get_tokens(): 
        token = token_obj.get_text()
        s_id = token_obj.get_sent()
        w_id = token_obj.get_id()
        token_data[w_id] = (token,s_id,num_token)
        tokens_in_order.append(w_id)
        num_token += 1
    print>>sys.stderr,'  Number of tokens: ',len(tokens_in_order)
    ###########################

    
    ###########################
    ## EXTRACTING TERMS #######
    term_data = {}  #(term_lemma,term_pos,term_span,polarity)
    term_for_token = {}
    sentence_for_term = {}
    for term_obj in knaf_obj.get_terms():
        term_id = term_obj.get_id()
        term_lemma = term_obj.get_lemma()
        term_pos = term_obj.get_morphofeat()
        if term_pos is not None:
            term_pos = term_pos.split(' ')[0] #[:2]  ## Only the 2 first chars of the pos string
        else:
            term_pos = ''
        
          
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
        sentence_id = token_data[tok_id][1]
        sentence_for_term[term_id] = sentence_id
    print>>sys.stderr,'  Number of terms loaded: '+str(len(term_data))
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
    print>>sys.stderr,'Entities:'+str(entity_for_term)

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
    print>>sys.stderr,'Properties:'+str(property_for_term)

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
                if exp_type in accepted_opinions:
                    #Get the mapping label
                    mapped_type = accepted_opinions[exp_type]
                else:
                    # This opinion wont be considered
                    continue
            else:
                mapped_type = exp_type
            
            
            print>>sys.stderr,'  Opinion',opinion_id
            print>>sys.stderr,'    Expression:'
            print>>sys.stderr,'      ids:',exp_ids
            print>>sys.stderr,'      terms:',[term_data[i][0] for i in exp_ids]
            if len(exp_ids) != 0:
                first_term_id = get_first_term_id(token_data,term_data,exp_ids)
                for t_id in exp_ids:
                    if t_id == first_term_id:  type='B-'
                    else:  type='I-'
                    class_for_term_id[t_id]=type+mapped_type
                    
            
            
            
            print>>sys.stderr,'    Target:'
            print>>sys.stderr,'      ids:',tar_ids
            print>>sys.stderr,'      terms:',[term_data[i][0] for i in tar_ids]
            if len(tar_ids) != 0:
                first_term_id = get_first_term_id(token_data,term_data,tar_ids)
                for t_id in tar_ids:
                    if t_id == first_term_id:  type='B-'
                    else:  type='I-'
                    class_for_term_id[t_id]=type+'target'        
            
            
            print>>sys.stderr,'    Holder:'
            print>>sys.stderr,'      ids:',hol_ids
            print>>sys.stderr,'      terms:',[term_data[i][0] for i in hol_ids]
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
        if term_id is not None:
            data = term_data.get(term_id,None)
            if data is not None:
                term_lemma,term_pos,term_span,polarity = data
                entity = entity_for_term.get(term_id,'-')
                property = property_for_term.get(term_id,'-')
                this_class = class_for_term_id.get(term_id,'O')
                
                #Mpqa subjectivy from the mpqa corpus
                mpqa_type = mpqa_pol = '-'
                if my_mpqa_subj_lex is not None:
                    mpqa_data = my_mpqa_subj_lex.get_type_and_polarity(token,term_pos)
                    if mpqa_data is not None:
                        mpqa_type, mpqa_pol = mpqa_data
                                  
                                
                                
                ## Constituency features
                constituency_extractor = knaf_obj.get_constituency_extractor()
                feature_phrase = 'XXX'
                if constituency_extractor is not None:
                    this_phrase, subsumed_together = constituency_extractor.get_deepest_phrase_for_termid(term_id)
                    if this_phrase is not None:
                        feature_phrase = this_phrase
                ######################
                                  
                ##############################################################################################
                ## FEATURE GENERATION!!!!
                ##############################################################################################
                
                features = [sentence_id,token_id,token,term_lemma,term_pos,term_id, polarity]
                features.extend([mpqa_type, mpqa_pol, entity,property,feature_phrase,this_class])
                
                ##############################################################################################
                ##############################################################################################
                
                
        if prev_sent is not None and sentence_id != prev_sent: print>>sys.stdout    #breakline 
        print>>sys.stdout,separator.join(features)
        
        prev_sent=sentence_id
    print>>sys.stdout   #Last breakline required for crfsuite
    

    print>>sys.stderr
    ## Restoring
    if restore_err is not None:
        sys.stderr.close()
        sys.stderr = restore_err
    if restore_out is not None:
        sys.stdout.close()
        sys.stdout = restore_out
    return labels, separator

