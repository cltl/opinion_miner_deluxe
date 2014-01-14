#!/usr/bin/env python

import sys
import os
import logging
import shutil
import glob
from subprocess import Popen, PIPE
import cPickle
import time



from scripts.config_manager import Cconfig_manager
from scripts.extract_features import extract_features_from_kaf_naf_file
from scripts.crfutils import extract_features_to_crf    
from scripts.extract_feats_relations import create_rel_exp_tar_training, create_rel_exp_hol_training
from VUA_pylib.io import Cfeature_file, Cfeature_index
from KafNafParserPy import KafNafParser



#Globa configuration
my_config_manager = Cconfig_manager()

logging.basicConfig(stream=sys.stderr,format='%(asctime)s - %(levelname)s\n  + %(message)s', level=logging.DEBUG)
__this_folder = os.path.dirname(os.path.realpath(__file__))


def save_obj_to_file(obj,filename):
    fic = open(filename,'wb')
    cPickle.dump(obj,fic)
    fic.close()

def create_folders():
    out_folder = my_config_manager.get_output_folder()

   
    logging.debug('Complete path to output folder: '+out_folder)
    
    # Remove the folder if it exists
    if os.path.exists(out_folder):
        shutil.rmtree(out_folder)
        logging.debug('Output folder exists and was removed')
    
    os.mkdir(out_folder)
    logging.debug('Created '+out_folder)

    feat_folder = my_config_manager.get_feature_folder_name()
    logging.debug('Created '+feat_folder)
    os.mkdir(feat_folder)

    crf_exp = my_config_manager.get_crf_expression_folder()
    os.mkdir(crf_exp)
    logging.debug('Created '+crf_exp) 
    
    crf_target = my_config_manager.get_crf_target_folder()
    os.mkdir(crf_target)
    logging.debug('Created '+crf_target) 
    
    crf_holder = my_config_manager.get_crf_holder_folder()
    os.mkdir(crf_holder)
    logging.debug('Created '+crf_holder)
    
    datasets_folder = my_config_manager.get_training_datasets_folder()
    os.mkdir(datasets_folder)
    logging.debug('Created '+datasets_folder)
    
    models_folder = my_config_manager.get_model_foldername()
    os.mkdir(models_folder)
    logging.debug('Created '+models_folder)
    
    relation_folder = my_config_manager.get_folder_relation_classifier()
    os.mkdir(relation_folder)
    logging.debug('Created '+relation_folder)


def load_training_files():
    file_training_files_cfg = my_config_manager.get_file_training_list()
    train_files = []
    path_to_file = ''
    if os.path.isabs(file_training_files_cfg):
        path_to_file = file_training_files_cfg
    else:
        path_to_file = os.path.join(__this_folder,file_training_files_cfg)
    logging.debug('Reading training files from '+path_to_file)
    try:
        fic = open(path_to_file,'r')
        for line in fic:
            train_files.append(line.strip())
        fic.close()
    except Exception as e:
        print>>sys.stderr,'Exception reading '+path_to_file,' -->'+str(e)
        sys.exit(-1)
    return train_files
    
    
def extract_all_features():
    train_files = load_training_files()
    logging.debug('Loaded '+str(len(train_files))+' files')

    feat_folder = my_config_manager.get_feature_folder_name()
    label_feats = separator = None
    my_stdout, my_stderr = sys.stdout,sys.stderr
    
    rel_exp_tar_filename = my_config_manager.get_relation_exp_tar_training_filename()
    exp_tar_rel_fic = open(rel_exp_tar_filename,'w')
   
    rel_exp_hol_filename = my_config_manager.get_relation_exp_hol_training_filename()
    exp_hol_rel_fic = open(rel_exp_hol_filename,'w') 
    
      
    accepted_opinions = my_config_manager.get_mapping_valid_opinions()
    
    for num_file, train_file in enumerate(train_files):
        logging.debug('Processing '+train_file)
        base_name = os.path.basename(train_file)
        out_file = os.path.join(feat_folder,'file#'+str(num_file)+'#'+base_name+".feat")
        err_file = out_file+'.log'
        
        #Creates the output file
        # Returns the labels for the features and the separator used
        if True:
            kaf_naf_obj = KafNafParser(train_file)
            
            label_feats, separator = extract_features_from_kaf_naf_file(kaf_naf_obj,out_file,err_file, accepted_opinions=accepted_opinions)
            print>>exp_tar_rel_fic,'#'+train_file
            print>>exp_hol_rel_fic,'#'+train_file
            # SET valid_opinions to None to use all the possible opinions in the KAF file for extracitng relations 
            create_rel_exp_tar_training(kaf_naf_obj, output=exp_tar_rel_fic, valid_opinions=accepted_opinions)
            create_rel_exp_hol_training(kaf_naf_obj ,output=exp_hol_rel_fic, valid_opinions=accepted_opinions) 
        if False:
        #except Exception as e:
            sys.stdout, sys.stderr = my_stdout, my_stderr
            print>>sys.stderr,str(e),dir(e)
            pass
        
    #Re-set the stdout and stderr
    exp_tar_rel_fic.close()
    exp_hol_rel_fic.close()
    
    sys.stdout,sys.stderr = my_stdout, my_stderr
    #Sabe labelfeats and separator in a file
    filename = my_config_manager.get_feature_desc_filename()
    fic = open(filename,'w')
    fic.write(' '.join(label_feats)+'\n')
    fic.close()
    logging.debug('Description of features --> '+filename)
    
    
    
def train_expression_classifier():
    # 1) Create the training file from all the features
    # Load the feature description
    path_feat_desc = my_config_manager.get_feature_desc_filename()
    fic = open(path_feat_desc)
    fields = fic.read().strip()
    fic.close()
    separator = '\t'
    feat_folder = my_config_manager.get_feature_folder_name()
    crf_folder = my_config_manager.get_crf_expression_folder()
    # Create all the CRF files calling to the crfutils.extract_features_to_crf 
       
    crf_out_files = []
    templates = (
        (('token',-2),), (('lemma',-2),), (('pos',-2),),(('pol/mod',-2),),(('mpqa_subjectivity',-2),),(('mpqa_polarity',-2),),#(('phrase_type',-2),),
        (('token',-1),), (('lemma',-1),), (('pos',-1),),(('pol/mod',-1),),(('mpqa_subjectivity',-1),),(('mpqa_polarity',-1),),(('phrase_type',-1),),
        (('token',+0),), (('lemma',+0),), (('pos',+0),),(('pol/mod',+0),),(('mpqa_subjectivity',+0),),(('mpqa_polarity',+0),),(('phrase_type',+0),),
        (('token',+1),), (('lemma',+1),), (('pos',+1),),(('pol/mod',+1),),(('mpqa_subjectivity',+1),),(('mpqa_polarity',+1),),(('phrase_type',+1),),
        (('token',+2),), (('lemma',+2),), (('pos',+2),),(('pol/mod',+2),),(('mpqa_subjectivity',+2),),(('mpqa_polarity',+2),),#(('phrase_type',+2),),
    ) 
    possible_classes = my_config_manager.get_possible_expression_values()
    template_filename = my_config_manager.get_expression_template_filename()
    save_obj_to_file(templates,template_filename)
    
    # Only set the target class for the tokens of possible_classes
    # For others, it's set to O (out sequence)
    for feat_file in glob.glob(feat_folder+'/*.feat'):
        base_name = os.path.basename(feat_file)
        base_name = base_name[:-5]
        out_crf = os.path.join(crf_folder,base_name)
        logging.debug('Creating crf file in --> '+out_crf)
        
        try:
            extract_features_to_crf(feat_file,out_crf,fields,separator,templates,possible_classes)
            crf_out_files.append(out_crf)
        except:
            print>>sys.stderr,'Failed conversion to tab-expression -> CRF: ',feat_file
    ###########################################################################################
   
    # Concatenate all the crf files just created 
    out_f = open(my_config_manager.get_training_dataset_exp(),'w')
    for crf_file in crf_out_files:
        f = open(crf_file)
        out_f.write(f.read())
        f.close()
    out_f.close()
    logging.debug('Created training data for crf, op.exp '+my_config_manager.get_training_dataset_exp())
    #############################################
    
    #Train the model
    crf_params = my_config_manager.get_crfsuite_params()
    input_file = my_config_manager.get_training_dataset_exp()
    model_file = my_config_manager.get_filename_model_expression()
    logging.debug('Training the classifier for opinion expressions (could take a while)')
    run_crfsuite(crf_params,input_file,model_file)
    
    
    
def train_target_classifier():
    
    # 1) Create the training file from all the features
    # Load the feature description
    path_feat_desc = my_config_manager.get_feature_desc_filename()
    fic = open(path_feat_desc)
    fields = fic.read().strip()
    fic.close()
    separator = '\t'
    feat_folder = my_config_manager.get_feature_folder_name()
    crf_folder = my_config_manager.get_crf_target_folder()
    # Create all the CRF files calling to the crfutils.extract_features_to_crf    
    crf_out_files = []
    templates = (
                     (('token',-2),), (('lemma',-2),), (('pos',-2),), (('entity',-2),), (('property',-2),),#(('phrase_type',-2),),
                     (('token',-1),), (('lemma',-1),), (('pos',-1),), (('entity',-1),), (('property',-1),),(('phrase_type',-1),),
                     (('token',+0),), (('lemma',+0),), (('pos',+0),), (('entity',+0),), (('property',+0),),(('phrase_type',+0),),
                     (('token',+1),), (('lemma',+1),), (('pos',+1),), (('entity',+1),), (('property',+1),),(('phrase_type',+1),),
                     (('token',+2),), (('lemma',+2),), (('pos',+2),), (('entity',+2),), (('property',+2),),#(('phrase_type',+2),),
        )
    possible_classes = ['target']
    template_filename = my_config_manager.get_target_template_filename()
    save_obj_to_file(templates,template_filename)
    for feat_file in glob.glob(feat_folder+'/*.feat'):
        base_name = os.path.basename(feat_file)
        base_name = base_name[:-5]
        out_crf = os.path.join(crf_folder,base_name)
        logging.debug('Creating crf file in --> '+out_crf)
        
        try:
            extract_features_to_crf(feat_file,out_crf,fields,separator,templates,possible_classes)
            crf_out_files.append(out_crf)
        except:
            print>>sys.stderr,'Failed conversion to tab-target-> CRF: ',feat_file
    ###########################################################################################
   
    # Concatenate all the crf files just created 
    out_f = open(my_config_manager.get_training_dataset_target(),'w')
    for crf_file in crf_out_files:
        f = open(crf_file)
        out_f.write(f.read())
        f.close()
    out_f.close()
    logging.debug('Created training data for crf, op.exp '+my_config_manager.get_training_dataset_target())
    #############################################
    
    #Train the model
    crf_params = my_config_manager.get_crfsuite_params()
    input_file = my_config_manager.get_training_dataset_target()
    model_file = my_config_manager.get_filename_model_target()
    logging.debug('Training the classifier for opinion target (could take a while)')
    run_crfsuite(crf_params,input_file,model_file)
    
    
    

def train_holder_classifier():
    
    # 1) Create the training file from all the features
    # Load the feature description
    path_feat_desc = my_config_manager.get_feature_desc_filename()
    fic = open(path_feat_desc)
    fields = fic.read().strip()
    fic.close()
    separator = '\t'
    feat_folder = my_config_manager.get_feature_folder_name()
    crf_folder = my_config_manager.get_crf_holder_folder()
    # Create all the CRF files calling to the crfutils.extract_features_to_crf    
    crf_out_files = []
    templates = (
                     (('token',-2),), (('lemma',-2),), (('pos',-2),), (('entity',-2),), (('property',-2),),#(('phrase_type',-2),),
                     (('token',-1),), (('lemma',-1),), (('pos',-1),), (('entity',-1),), (('property',-1),),(('phrase_type',-1),),
                     (('token',+0),), (('lemma',+0),), (('pos',+0),), (('entity',+0),), (('property',+0),),(('phrase_type',+0),),
                     (('token',+1),), (('lemma',+1),), (('pos',+1),), (('entity',+1),), (('property',+1),),(('phrase_type',+1),),
                     (('token',+2),), (('lemma',+2),), (('pos',+2),), (('entity',+2),), (('property',+2),),#(('phrase_type',+2),),
        )
    possible_classes = ['holder']
    template_filename = my_config_manager.get_holder_template_filename()
    save_obj_to_file(templates,template_filename)
    for feat_file in glob.glob(feat_folder+'/*.feat'):
        base_name = os.path.basename(feat_file)
        base_name = base_name[:-5]
        out_crf = os.path.join(crf_folder,base_name)
        logging.debug('Creating crf file in --> '+out_crf)
    
        try:
            extract_features_to_crf(feat_file,out_crf,fields,separator,templates,possible_classes)
            crf_out_files.append(out_crf)
        except:
            print>>sys.stderr,'Failed conversion to tab-holder -> CRF: ',feat_file
    ###########################################################################################
   
    # Concatenate all the crf files just created 
    out_f = open(my_config_manager.get_training_dataset_holder(),'w')
    for crf_file in crf_out_files:
        f = open(crf_file)
        out_f.write(f.read())
        f.close()
    out_f.close()
    logging.debug('Created training data for crf, op.exp '+my_config_manager.get_training_dataset_holder())
    #############################################
    
    #Train the model
    crf_params = my_config_manager.get_crfsuite_params()
    input_file = my_config_manager.get_training_dataset_holder()
    model_file = my_config_manager.get_filename_model_holder()
    logging.debug('Training the classifier for opinion holder (could take a while)')
    run_crfsuite(crf_params,input_file,model_file)
    

def run_crfsuite(crf_params,input_file,model_file):
    
    crfsuite = my_config_manager.get_crfsuite_binary()
    cmd = [crfsuite]
    cmd.append('learn')
    cmd.append(crf_params)
    cmd.append('-m '+model_file)
    cmd.append(input_file)
    err_file = model_file+'.log'
    err_fic = open(err_file,'w')
    crf_process = Popen(' '.join(cmd), stdin=PIPE, stdout=err_fic, stderr=PIPE, shell=True)
    crf_process.wait()
    str_err = crf_process.stderr.read()
    if len(str_err) != 0:
        print>>sys.stderr,'CRF error!!: '+str_err
        sys.exit(-1)
    err_fic.close()
    logging.debug('Crfsuite log '+err_file) 
    
    
    
    
############################################
################ RELATION TRAINING #########
###########################################

def train_classifier_relation_exp_tar():
    #Load the human readable training file    
    train_filename = my_config_manager.get_relation_exp_tar_training_filename()
    feature_file_obj = Cfeature_file(train_filename)
    ###########################################

    
    # Convert it into index based feature file, for svm-light
    feature_index = Cfeature_index()
    feat_bin_filename = my_config_manager.get_rel_exp_tar_training_idx_filename()
    fic_out = open(feat_bin_filename,'w')
    feature_index.encode_feature_file_to_svm(feature_file_obj,fic_out)
    fic_out.close()
    ###########################################

    
    ## Save the feature index
    feat_index_filename = my_config_manager.get_index_features_exp_tar_filename()
    feature_index.save_to_file(feat_index_filename)
    #########################
    
    # Train the model
    example_file = my_config_manager.get_rel_exp_tar_training_idx_filename()
    model = my_config_manager.get_filename_model_exp_tar()
    svm_opts = my_config_manager.get_svm_params()
    logging.debug('Training SVMlight classifier for RELATION(expression,target) in '+model+ '(could take a while)')
    run_svmlight_learn(example_file,model,svm_opts)
    ###########################################




def train_classifier_relation_exp_hol():
    #Load the human readable training file    
    train_filename = my_config_manager.get_relation_exp_hol_training_filename()
    feature_file_obj = Cfeature_file(train_filename)
    ###########################################

    
    # Convert it into index based feature file, for svm-light
    feature_index = Cfeature_index()
    feat_bin_filename = my_config_manager.get_rel_exp_hol_training_idx_filename()
    fic_out = open(feat_bin_filename,'w')
    feature_index.encode_feature_file_to_svm(feature_file_obj,fic_out)
    fic_out.close()
    ###########################################

    
    ## Save the feature index
    feat_index_filename = my_config_manager.get_index_features_exp_hol_filename()
    feature_index.save_to_file(feat_index_filename)
    #########################
    
    # Train the model
    example_file = my_config_manager.get_rel_exp_hol_training_idx_filename()
    model = my_config_manager.get_filename_model_exp_hol()
    svm_opts = my_config_manager.get_svm_params()
    logging.debug('Training SVMlight classifier for RELATION(expression,holder) in '+model+ '(could take a while)')
    run_svmlight_learn(example_file,model,svm_opts)
    ###########################################
    
    
def run_svmlight_learn(example_file,model_file,params):
    svmlight = my_config_manager.get_svm_learn_binary()
    cmd = [svmlight]
    cmd.append(params)
    cmd.append(example_file)
    cmd.append(model_file)
    err_file = model_file+'.log'
    err_fic = open(err_file,'w')
    svm_process = Popen(' '.join(cmd),stdin=PIPE, stdout=err_fic, stderr=PIPE, shell=True)
    svm_process.wait()
    str_err = svm_process.stderr.read()
    if len(str_err) != 0:
        print>>sys.stderr,'SVM light error '+str_err
        sys.exit(-1)
    err_fic.close()
    logging.debug('SVMlight learn log'+err_file)
    
def write_to_flag(msg,openas='a'):
    flag = open(my_config_manager.get_flag_filename(),openas)
    my_time = time.strftime('%Y-%m-%dT%H:%M:%S%Z')
    flag.write(msg+' --> '+my_time+'\n')
    flag.close() 

if __name__ == '__main__':
    file_config = sys.argv[1]
    
    # Read configuration from the config file
    my_config_manager.set_current_folder(__this_folder)
    my_config_manager.set_config(file_config)
      
    
    # Check if the output folder exists or create it
    create_folders()
    write_to_flag('Beginning\n','w')
       
    #Will create the subfolder out_folder/subfolder_feats with files *feat
    write_to_flag('START extract features')
    extract_all_features()
    write_to_flag('DONE extract features\n')
    
    # training the expression classifier
    write_to_flag('START training expression classifier')
    train_expression_classifier()
    write_to_flag('DONE training expression classifier\n')

    
    # Training the target classifier
    write_to_flag('START training target classifier')
    train_target_classifier()
    write_to_flag('DONE training target classifier\n')
    
    # training the holder classifier
    write_to_flag('START training expression classifier')
    train_holder_classifier()
    write_to_flag('DONE training holder classifier\n')

    
    write_to_flag('START training relation expression - target classifier')
    train_classifier_relation_exp_tar()
    write_to_flag('DONE training relation expression - target classifier\n')
    
    write_to_flag('START training relation expression - holder classifier')
    train_classifier_relation_exp_hol()
    write_to_flag('DONE training relation expression - holder classifier\n')
    
    
    logging.debug('ALL TRAINING DONE')
    write_to_flag('FINISHED ')
    
    
    

    
    
    sys.exit(0)