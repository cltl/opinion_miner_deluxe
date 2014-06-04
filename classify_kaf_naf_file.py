#!/usr/bin/env python

import sys
import os
import csv
from tempfile import NamedTemporaryFile
from subprocess import Popen, PIPE
import logging
import cPickle
import argparse

from scripts import lexicons as lexicons_manager
from scripts.config_manager import Cconfig_manager, internal_config_filename
from scripts.extract_features import extract_features_from_kaf_naf_file
from scripts.crfutils import extract_features_to_crf 
from scripts.link_entities_distance import link_entities_distance
from scripts.relation_classifier import link_entities_svm
from KafNafParserPy import *
from subjectivity_detector import classify_sentences


DEBUG=0

my_config_manager = Cconfig_manager()
__this_folder = os.path.dirname(os.path.realpath(__file__))
separator = '\t'
__desc = 'Deluxe opinion miner (CRF+SVM)'
__last_edited = '10jan2014'
__version = '2.0'

logging.basicConfig(stream=sys.stderr,format='%(asctime)s - %(levelname)s\n  + %(message)s', level=logging.CRITICAL)

terms_for_token = None



def remove_sentences_no_opinionated(kaf_obj,threshold=0.25):
       
    # Get the sentences
    ids_sentences = {}
    for token in kaf_obj.get_tokens():
        value = token.get_text()
        sent = token.get_sent()
        if sent not in ids_sentences:
            ids_sentences[sent] = [value]
        else:
            ids_sentences[sent].append(value)

    lang = kaf_obj.get_language()
    model = '/home/izquierdo/cltl_repos/opinion_miner_deluxe/subjectivity_detector/hotel_new_models/'+lang
    if not os.path.exists(model):
        print 'Model file ',model,'does not exist. Skipping'
        return None
    
    ids = ids_sentences.keys()
    sentences = ids_sentences.values()
    values = classify_sentences(sentences, model)    
    sent_ids_to_remove = set()
    
    for n, sent in enumerate(sentences):
        #print ids[n]
        #print '\t',' '.join(sent).encode('utf-8')
        #print '\t',values[n]
        if values[n] < threshold:
            sent_ids_to_remove.add(ids[n])
            
    ##Remove from the input all the sentences with sent id in sent_ids
    for sent_id in sent_ids_to_remove:
        kaf_obj.remove_tokens_of_sentence(sent_id)
        


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



def extract_features(kaf_naf_obj):
    feat_file_desc = NamedTemporaryFile(delete=False)
    feat_file_desc.close()
    
    out_file = feat_file_desc.name
    err_file = out_file+'.log'
    
    expressions_lexicon = None
    targets_lexicon = None
    if my_config_manager.get_use_training_lexicons():
        expression_lexicon_filename = my_config_manager.get_expression_lexicon_filename()
        target_lexicon_filename = my_config_manager.get_target_lexicon_filename()
        
        expressions_lexicon = lexicons_manager.load_lexicon(expression_lexicon_filename)
        targets_lexicon =lexicons_manager.load_lexicon(target_lexicon_filename)

    #def extract_features_from_kaf_naf_file(knaf_obj,out_file=None,log_file=None,include_class=True,accepted_opinions=None, exp_lex= None):
    labels, separator,polarities_skipped = extract_features_from_kaf_naf_file(kaf_naf_obj,out_file,err_file,include_class=False, exp_lex=expressions_lexicon,tar_lex=targets_lexicon)
    return out_file, err_file
             
            
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
    if not os.path.exists(crfsuite):
        print>>sys.stderr,'CRFsuite not found on',crfsuite
        print>>sys.stderr,'Check the config filename and make sure the path is correctly set'
        print>>sys.stderr,'[crfsuite]\npath_to_binary = yourpathtolocalcrfsuite'
        sys.exit(-1)

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
    templates = my_config_manager.get_templates_expr() 
    
    crf_exp_file = convert_to_crf(tab_feat_file,templates)
    logging.debug('File with crf format for EXPRESSIONS '+crf_exp_file)
    if DEBUG:
        print>>sys.stderr,'#'*50
        print>>sys.stderr,'CRF FEATURES EXPRESSION'
        f = open(crf_exp_file)
        print>>sys.stderr,f.read()
        f.close()
        print>>sys.stderr,'#'*50
    
    model_file = my_config_manager.get_filename_model_expression()
    output_crf,error_crf = run_crfsuite_tag(crf_exp_file,model_file)
    
    logging.debug('Expressions crf error: '+error_crf)
    matches_exp = match_crfsuite_out(output_crf, list_token_ids)
    if DEBUG:
        print>>sys.stderr,'#'*50
        print>>sys.stderr,'CRF output for EXPRESSION'
        print>>sys.stderr,'Raw output CRF:', output_crf
        print>>sys.stderr,'List token ids:',str(list_token_ids)
        print>>sys.stderr,'MATCHES:',str(matches_exp)
        print>>sys.stderr,'TEMP FILE:',crf_exp_file
        print>>sys.stderr,'#'*50
  
    
    logging.debug('Detector expressions out: '+str(matches_exp))
    os.remove(crf_exp_file)
    return matches_exp
    
    
    
    
    
def detect_targets(tab_feat_file, list_token_ids):
    templates_target =  my_config_manager.get_templates_target()
    
    crf_target_file = convert_to_crf(tab_feat_file,templates_target)
    logging.debug('File with crf format for TARGETS '+crf_target_file)
    if DEBUG:
        print>>sys.stderr,'#'*50
        print>>sys.stderr,'CRF FEATURES TARGETS'
        f = open(crf_target_file)
        print>>sys.stderr,f.read()
        f.close()
        print>>sys.stderr,'#'*50
        
    model_target_file = my_config_manager.get_filename_model_target()
    out_crf_target,error_crf = run_crfsuite_tag(crf_target_file, model_target_file)
    logging.debug('TARGETS crf error: '+error_crf)

    matches_tar = match_crfsuite_out(out_crf_target, list_token_ids)
    
    if DEBUG:
        print>>sys.stderr,'#'*50
        print>>sys.stderr,'CRF output for TARGETS'
        print>>sys.stderr,'Raw output CRF:', out_crf_target
        print>>sys.stderr,'List token ids:',str(list_token_ids)
        print>>sys.stderr,'MATCHES:',str(matches_tar)
        print>>sys.stderr,'#'*50
        
    logging.debug('Detector targets out: '+str(matches_tar))
    os.remove(crf_target_file)
    return matches_tar
           
           
           
           
           
def detect_holders(tab_feat_file, list_token_ids):
    templates_holder = my_config_manager.get_templates_holder()
    
    crf_holder_file = convert_to_crf(tab_feat_file,templates_holder)
    logging.debug('File with crf format for HOLDERS '+crf_holder_file)
    if DEBUG:
        print>>sys.stderr,'#'*50
        print>>sys.stderr,'CRF FEATURES HOLDERS'
        f = open(crf_holder_file)
        print>>sys.stderr,f.read()
        f.close()
        print>>sys.stderr,'#'*50
        
    model_holder_file = my_config_manager.get_filename_model_holder()
    out_crf_holder,error_crf = run_crfsuite_tag(crf_holder_file, model_holder_file)
    logging.debug('HOLDERS crf error: '+error_crf)

    matches_holder = match_crfsuite_out(out_crf_holder, list_token_ids)

    if DEBUG:
        print>>sys.stderr,'#'*50
        print>>sys.stderr,'CRF output for HOLDERS'
        print>>sys.stderr,'Raw output CRF:', out_crf_holder
        print>>sys.stderr,'List token ids:',str(list_token_ids)
        print>>sys.stderr,'MATCHES:',str(matches_holder)
        print>>sys.stderr,'#'*50
        
    logging.debug('Detector HOLDERS out: '+str(matches_holder))
    os.remove(crf_holder_file)
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
        
        
              
def add_opinions_to_knaf(triples,knaf_obj,text_for_tid,ids_used, map_to_terms=True,include_polarity_strength=True):
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
        
        hol_text = ' '.join(text_for_tid[tid] for tid in span_hol_terms)
        my_hol.set_comment(hol_text)
        
        #Creating target
        span_tar = Cspan()
        span_tar.create_from_ids(span_tar_terms)
        my_tar = opinion_data.Ctarget()
        my_tar.set_span(span_tar)
        tar_text = ' '.join(text_for_tid[tid] for tid in span_tar_terms)
        my_tar.set_comment(tar_text)
        #########################

        ##Creating expression
        span_exp = Cspan()
        span_exp.create_from_ids(span_exp_terms)
        my_exp = Cexpression()
        my_exp.set_span(span_exp)
        my_exp.set_polarity(type_exp)
        if include_polarity_strength:
            my_exp.set_strength("1")
        exp_text = ' '.join(text_for_tid[tid] for tid in span_exp_terms)
        my_exp.set_comment(exp_text)
        #########################
        
        #To get the first possible ID not already used
        new_id = None
        while True:
            new_id = 'o'+str(num_opinion+1)
            if new_id not in ids_used:
                ids_used.add(new_id)
                break
            else:
                num_opinion += 1
        new_opinion = Copinion(type=knaf_obj.get_type())
        new_opinion.set_id(new_id)
        if len(span_hol_terms) != 0:    #To avoid empty holders
            new_opinion.set_holder(my_hol)
            
        if len(span_tar_terms) != 0:    #To avoid empty targets
            new_opinion.set_target(my_tar)
            
        new_opinion.set_expression(my_exp)
        
        knaf_obj.add_opinion(new_opinion)
        
##
# Input_file_stream can be a filename of a stream
# Opoutfile_trasm can be a filename of a stream
#Config file must be a string filename
def tag_file_with_opinions(input_file_stream, output_file_stream,model_folder,kaf_obj=None, remove_existing_opinions=True,include_polarity_strength=True,timestamp=True):
    
    config_filename = os.path.join(model_folder,internal_config_filename)
    if not os.path.exists(config_filename):
        print>>sys.stderr,'Config file not found on:',config_filename
        sys.exit(-1)
    
    my_config_manager.set_current_folder(__this_folder)
    my_config_manager.set_config(config_filename)
    
    if kaf_obj is not None:
        knaf_obj = kaf_obj
    else:
        knaf_obj = KafNafParser(input_file_stream)
        
    ids_used = set()
    if remove_existing_opinions:
        knaf_obj.remove_opinion_layer()
    else:
        for opi in knaf_obj.get_opinions():
            ids_used.add(opi.get_id())
        
    ##LEAVE ONLY THE SENTENCES THAT MIGHT BE OPINIONATED    
    ##remove_sentences_no_opinionated(knaf_obj)  

        
        
    
    #get all the tokens in order
    list_token_ids = []
    text_for_wid = {}
    text_for_tid = {}
    sentence_for_token = {}
    for token_obj in knaf_obj.get_tokens():
        token = token_obj.get_text()
        s_id = token_obj.get_sent()
        w_id = token_obj.get_id()
        text_for_wid[w_id] = token
         
        list_token_ids.append(w_id)
        sentence_for_token[w_id] = s_id
        
    if len(list_token_ids) == 0: #There are no tokens because the subjectivy detector removed all of them...
        my_lp = Clp()
        my_lp.set_name(__desc)
        my_lp.set_version(__last_edited+'_'+__version)
        if timestamp:
            my_lp.set_timestamp()   ##Set to the current date and time
        else:
            my_lp.set_timestamp('*')
        knaf_obj.add_linguistic_processor('opinions',my_lp)
        knaf_obj.dump(output_file_stream)
        sys.exit(0)
        #DONE
        
    for term in knaf_obj.get_terms():
        tid = term.get_id()
        toks = [text_for_wid.get(wid,'') for wid in term.get_span().get_span_ids()]
        text_for_tid[tid] = ' '.join(toks)

    #Create a temporary file
    out_feat_file, err_feat_file = extract_features(knaf_obj)
    if DEBUG:
        print>>sys.stderr,'#'*50
        print>>sys.stderr,'FEATURE FILE'
        f = open(out_feat_file)
        print>>sys.stderr,f.read()
        f.close()
        print>>sys.stderr,'#'*50
    
       
    expressions = detect_expressions(out_feat_file,list_token_ids)
    targets = detect_targets(out_feat_file, list_token_ids)
    holders = detect_holders(out_feat_file, list_token_ids)
    
    os.remove(out_feat_file)
    os.remove(err_feat_file)

    if DEBUG:
        print>>sys.stderr,"Expressions detected:"
        for e in expressions:
            print>>sys.stderr,'\t',e, ' '.join([text_for_wid[wid] for wid in e[0] ]) 
        print>>sys.stderr
    
        print>>sys.stderr,'Targets detected'
        for t in targets:
            print>>sys.stderr,'\t',t, ' '.join([text_for_wid[wid] for wid in t[0] ]) 
        print>>sys.stderr
        
        print>>sys.stderr,'Holders',holders
        for h in holders:
            print>>sys.stderr,'\t',h, ' '.join([text_for_wid[wid] for wid in h[0] ]) 
        print>>sys.stderr
    
    
    # Entity linker based on distances
    ####triples = link_entities_distance(expressions,targets,holders,sentence_for_token)
    
    triples = link_entities_svm(expressions, targets, holders, knaf_obj, my_config_manager)
    
    ids_used = set()
    if remove_existing_opinions:
        knaf_obj.remove_opinion_layer()
    else:
        for opi in knaf_obj.get_opinions():
            ids_used.add(opi.get_id())
        
    
    add_opinions_to_knaf(triples, knaf_obj,text_for_tid,ids_used, map_to_terms=False,include_polarity_strength=include_polarity_strength)   
    
    #Adding linguistic processor
    my_lp = Clp()
    my_lp.set_name(__desc)
    my_lp.set_version(__last_edited+'_'+__version)
    if timestamp:
        my_lp.set_timestamp()   ##Set to the current date and time
    else:
        my_lp.set_timestamp('*')
    knaf_obj.add_linguistic_processor('opinions',my_lp)
    knaf_obj.dump(output_file_stream)
    
  

def obtain_predefined_model(lang,domain,just_show=False):
    #This function will read the models from the file models.cfg and will return
    #The model folder for the lang and domain
    # format of the file: 1 model per line: lang|domain|path_to_folder
    model_file = os.path.join(__this_folder,'models.cfg')
    fic = open(model_file)
    use_this_model = None
    if just_show:
        print '#'*25
        print 'Models available'
        print '#'*25
        
    nm = 0
    for line in fic:
        if line[0]!='#':
            this_lang, this_domain, this_model,this_desc = line.strip().split('|')
            if just_show:
                print '  Model',nm
                print '    Lang:',this_lang
                print '    Domain:', this_domain
                print '    Folder:',this_model
                print '    Desc:',this_desc
                nm+= 1
            else:
                if this_lang == lang and this_domain == domain:
                    use_this_model = this_model
                    break
    fic.close()
    if just_show:
         print '#'*25
    return use_this_model
        
if __name__ == '__main__':
    
    argument_parser = argparse.ArgumentParser(description='Detect opinion triples in a KAF/NAF file')
    group = argument_parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-m',dest='model_folder',help='Folder storing the trained models')
    group.add_argument('-d', dest='domain',help='The domain where the models were trained')
    group.add_argument('-show-models', dest='show_models', action='store_true',help='Show the models available and finish')
    
    argument_parser.add_argument('-keep-opinions',dest='keep_opinions',action='store_true',help='Keep the opinions from the input (by default will be deleted)')
    argument_parser.add_argument('-no-time',dest='timestamp',action='store_false',help='No include time in timestamp (for testing)')
    arguments = argument_parser.parse_args()

    if arguments.show_models:
        obtain_predefined_model(None,None,just_show=True)
        sys.exit(0)
        
    knaf_obj = KafNafParser(sys.stdin)
    model_folder = None
    if arguments.model_folder is not None:
        model_folder = arguments.model_folder
    else:
        #Obtain the language
        lang = knaf_obj.get_language()
        model_folder = obtain_predefined_model(lang,arguments.domain)
            
        
    tag_file_with_opinions(None, sys.stdout,model_folder,kaf_obj=knaf_obj,remove_existing_opinions=(not arguments.keep_opinions),timestamp=arguments.timestamp)
    sys.exit(0)
    
    
    
    
    