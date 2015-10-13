#!/usr/bin/env python

import os
import sys
from subprocess import Popen, PIPE
from tempfile import NamedTemporaryFile
from KafNafParserPy import KafNafParser

from VUA_pylib.io import Cfeature_index, Cfeature_file

def extract_features_polarity_classifier_from_tokens(tokens):
    features = []
    #Unigrams
    for token in tokens:
        features.append(('unigram',token))
        
    ##bigram
    for n in range(len(tokens)-1):
        features.append(('bigram',tokens[n]+'#'+tokens[n+1]))
    
    ##trigram
    for n in range(len(tokens)-2):
        features.append(('trigram',tokens[n]+'#'+tokens[n+1]+'#'+tokens[n+2]))  
    
    return features

def get_tokens_for_terms(kaf_obj,term_ids):
    list_tokens_offset = []
    there_are_offset = False
    for tid in term_ids:
        token_ids = kaf_obj.get_dict_tokens_for_termid(tid)
        for wid in token_ids:
            token_obj = kaf_obj.get_token(wid)
            text = token_obj.get_text()
            offset = token_obj.get_offset()
            if offset is not None:
                there_are_offset = True
                offset =  int(offset)
            else:
                offset = 0 
            list_tokens_offset.append((text,offset))
    if there_are_offset:
        list_tokens_offset.sort(key=lambda t: t[1])
        
    tokens = []
    for token,offset in list_tokens_offset:
        tokens.append(token)
    return tokens
            
def extract_features_polarity_classifier_from_kaf(kaf_obj,fd, pos_neg_labels):
    for opinion in kaf_obj.get_opinions():
        expression = opinion.get_expression()
        if expression is not None:
            polarity = expression.get_polarity()
            term_ids = expression.get_span().get_span_ids()
            list_tokens_offset = get_tokens_for_terms(kaf_obj,term_ids)

            if polarity in pos_neg_labels:
                tokens = get_tokens_for_terms(kaf_obj,term_ids)
                features = extract_features_polarity_classifier_from_tokens(tokens)
                
                ##Guess the class
                this_class = None
                if pos_neg_labels[polarity] == 'positive':
                    this_class = '+1'
                elif pos_neg_labels[polarity] == 'negative':
                    this_class = '-1'
                    
                    
                fd.write(this_class)
                for name, feat in features:
                   fd.write('\t'+str(name.encode('utf-8'))+'='+str(feat.encode('utf-8')))
                fd.write('\n')
  


def run_svm_classify(svmlight, example_file,model_file):
    #usage: svm_classify [options] example_file model_file output_file
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

def classify(kaf_obj,term_ids,index_filename,model_filename, svm_path):
    index_features = Cfeature_index()
    index_features.load_from_file(index_filename)
    test_file = NamedTemporaryFile(delete=False)
    tokens = get_tokens_for_terms(kaf_obj,term_ids)
    features = extract_features_polarity_classifier_from_tokens(tokens)
    index_features.encode_example_for_classification(features,test_file)
    test_file.close()
    results = run_svm_classify(svm_path, test_file.name,model_filename)
    os.remove(test_file.name)
    if results[0] >= 0:
        return 'positive'
    else:
        return 'negative'



    
    
        

if __name__ == '__main__':
    import glob
    #feature_file = 'my_feat_file'
    #fd = open(feature_file,'w')
    #for kaf_file in glob.glob('/home/izquierdo/data/opinion_annotations_en/kaf/hotel/*.kaf'):
    #    print kaf_file
    #    knaf_obj = KafNafParser(kaf_file)
    #    extract_features_polarity_classifier_from_kaf(knaf_obj, fd)
    #fd.close()
    #print ' Feature file in ',feature_file
    #train_polarity_classifier(feature_file)
    kaf_obj = KafNafParser('dutch00011_f1b91e00bddbf62fbb35e4755e786406.kaf')
    list_terms = []
    list_ids = []
    for opinion in kaf_obj.get_opinions():
        exp = opinion.get_expression()
        pol = exp.get_polarity()
        if pol in ['Positive','Negative','StrongPositive','StrongNegative']:
            this_id = (opinion.get_id(),pol)
            ids = exp.get_span().get_span_ids()
            list_ids.append(this_id)
            list_terms.append(ids)
    index_filename = '/home/izquierdo/cltl_repos/opinion_miner_deluxe/check_me/polarity_classifier/index.features'
    model_filename = '/home/izquierdo/cltl_repos/opinion_miner_deluxe/check_me/polarity_classifier/model.svm'
    svm_path = '/home/izquierdo/bin/svm_classify'
    results = classify(kaf_obj,list_terms,index_filename,model_filename, svm_path)
    for n in range(len(results)):
        print list_ids[n], results[n]
