#!/usr/bin/env python

import sys
import os
import logging
import shutil
import glob
from subprocess import Popen, PIPE
import cPickle
import time
import csv
from collections import defaultdict


from scripts import lexicons as lexicons_manager
from scripts.config_manager import Cconfig_manager, internal_config_filename
from scripts.extract_features import extract_features_from_kaf_naf_file
from scripts.crfutils import extract_features_to_crf    
from scripts.extract_feats_relations import create_rel_exp_tar_training, create_rel_exp_hol_training
from VUA_pylib.io import Cfeature_file, Cfeature_index
from KafNafParserPy import KafNafParser



#Globa configuration
my_config_manager = Cconfig_manager()

__this_folder = os.path.dirname(os.path.realpath(__file__))


def save_obj_to_file(obj,filename):
    fic = open(filename,'wb')
    cPickle.dump(obj,fic)
    fic.close()

def create_folders(config_filename):
    global my_config_manager
        
    # Read configuration from the config file
    my_config_manager.set_current_folder(__this_folder)
    my_config_manager.set_config(config_filename)
    
    out_folder = my_config_manager.get_output_folder()

   
    logging.debug('Complete path to output folder: '+out_folder)
    
    # Remove the folder if it exists
    if os.path.exists(out_folder):
        shutil.rmtree(out_folder)
        logging.debug('Output folder exists and was removed')
    
    os.mkdir(out_folder)
    logging.debug('Created '+out_folder)

    #Copy the config filename to out_folder/config.cfg
    my_cfg = os.path.join(out_folder,internal_config_filename)
    shutil.copyfile(config_filename,my_cfg)

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
    
    ##Templates folder
    template_folder = my_config_manager.get_feature_template_folder_name()
    os.mkdir(template_folder)
    logging.debug('Created '+template_folder)
        
    ##Copy template files
    my_config_manager.copy_feature_templates()

    ##Folder for lexicons
    lexicons_folder = my_config_manager.get_lexicons_folder()
    os.mkdir(lexicons_folder)
    logging.debug('Created '+lexicons_folder)

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
    
    ### LEXICON FROM THE DOMAIN
    expressions_lexicon = None
    targets_lexicon = None
    if my_config_manager.get_use_training_lexicons():
        # Create the lexicons
        
        ##GUESS THE LANG:
        first_train_file = train_files[0]
        obj = KafNafParser(first_train_file)
        lang = obj.get_language()
        
        expression_lexicon_filename = my_config_manager.get_expression_lexicon_filename()
        target_lexicon_filename = my_config_manager.get_target_lexicon_filename()
        
        
        this_exp_lex = my_config_manager.get_use_this_expression_lexicon()            
        this_tar_lex = my_config_manager.get_use_this_target_lexicon()

        
        if this_exp_lex is None or this_tar_lex is None:
            path_to_lex_creator = '/home/izquierdo/opener_repos/opinion-domain-lexicon-acquisition/acquire_from_annotated_data.py'
            training_filename = my_config_manager.get_file_training_list()
            lexicons_manager.create_lexicons(path_to_lex_creator,training_filename,expression_lexicon_filename,target_lexicon_filename)
        
        ##Once created we have to copy the previous one in case:
        if this_exp_lex is not None:
            if "$LANG" in this_exp_lex:
                this_exp_lex = this_exp_lex.replace('$LANG',lang)
            shutil.copy(this_exp_lex, expression_lexicon_filename)
            
        if this_tar_lex is not None:
            if "$LANG" in this_tar_lex:
                this_tar_lex = this_tar_lex.replace('$LANG',lang)
            shutil.copy(this_tar_lex,target_lexicon_filename)
        
        expressions_lexicon = lexicons_manager.load_lexicon(expression_lexicon_filename)
        targets_lexicon =  lexicons_manager.load_lexicon(target_lexicon_filename)
        
        this_propagation_lexicon = my_config_manager.get_propagation_lexicon_name()
        if this_propagation_lexicon is not None:
            if "$LANG" in this_propagation_lexicon:
                this_propagation_lexicon = this_propagation_lexicon.replace('$LANG',lang)
                
        print>>sys.stderr,'Propagated lexicon',this_propagation_lexicon
        
        
        

    ## Configuration for the relational alcasifier
    use_deps_now = my_config_manager.get_use_dependencies()
    use_toks_lems_now = my_config_manager.get_use_tokens_lemmas()
      
    accepted_opinions = my_config_manager.get_mapping_valid_opinions()
    use_dependencies_now = my_config_manager.get_use_dependencies()
    polarities_found_and_skipped = []
    for num_file, train_file in enumerate(train_files):
        logging.debug('Extracting features '+os.path.basename(train_file))
        base_name = os.path.basename(train_file)
        out_file = os.path.join(feat_folder,'file#'+str(num_file)+'#'+base_name+".feat")
        err_file = out_file+'.log'
        
        #Creates the output file
        # Returns the labels for the features and the separator used
        if True:
            kaf_naf_obj = KafNafParser(train_file)
            
            label_feats, separator, pols_skipped_this = extract_features_from_kaf_naf_file(kaf_naf_obj,out_file,err_file, 
                                                                                           accepted_opinions=accepted_opinions, 
                                                                                           exp_lex=expressions_lexicon, 
                                                                                           tar_lex=targets_lexicon,
                                                                                           propagation_lex_filename=this_propagation_lexicon)
            polarities_found_and_skipped.extend(pols_skipped_this)
            print>>exp_tar_rel_fic,'#'+train_file
            print>>exp_hol_rel_fic,'#'+train_file
            # SET valid_opinions to None to use all the possible opinions in the KAF file for extracitng relations 
            create_rel_exp_tar_training(kaf_naf_obj, output=exp_tar_rel_fic, valid_opinions=accepted_opinions,use_dependencies=use_dependencies_now,use_tokens=use_toks_lems_now,use_lemmas=use_toks_lems_now)
            create_rel_exp_hol_training(kaf_naf_obj ,output=exp_hol_rel_fic, valid_opinions=accepted_opinions,use_dependencies=use_dependencies_now,use_tokens=use_toks_lems_now,use_lemmas=use_toks_lems_now)
        if False:
        #except Exception as e:
            sys.stdout, sys.stderr = my_stdout, my_stderr
            print>>sys.stderr,str(e),dir(e)
            pass
        
    ##Show just for information how many instances have been skipped becase the polarity of opinion expression was not allowed
    count = defaultdict(int)
    for exp_label in polarities_found_and_skipped:
        count[exp_label] += 1
    info = '\nOpinions skipped because the polarity label is not included in the configuration\n'
    info += 'Accepted opinions: '+' '.join(accepted_opinions.keys())+'\n'
    info += 'Number of complete opinions skipped\n'
    for label, c in count.items():
        info+=' '+label+' :'+str(c)+'\n'
    info+='\n'
    logging.debug(info)
    ###################################################
    
    
        
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
    
    templates_exp = my_config_manager.get_templates_expr() 
    possible_classes = my_config_manager.get_possible_expression_values()
      
    # Only set the target class for the tokens of possible_classes
    # For others, it's set to O (out sequence)
    for feat_file in glob.glob(feat_folder+'/*.feat'):
        base_name = os.path.basename(feat_file)
        base_name = base_name[:-5]
        out_crf = os.path.join(crf_folder,base_name)
        logging.debug('Creating crf file in --> '+out_crf)
        
        try:
            extract_features_to_crf(feat_file,out_crf,fields,separator,templates_exp,possible_classes)
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
    templates_target = my_config_manager.get_templates_target()
    possible_classes = ['target']
    for feat_file in glob.glob(feat_folder+'/*.feat'):
        base_name = os.path.basename(feat_file)
        base_name = base_name[:-5]
        out_crf = os.path.join(crf_folder,base_name)
        logging.debug('Creating crf file in --> '+out_crf)
        
        try:
            extract_features_to_crf(feat_file,out_crf,fields,separator,templates_target,possible_classes)
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
    templates_holder = my_config_manager.get_templates_holder()
    possible_classes = ['holder']
    for feat_file in glob.glob(feat_folder+'/*.feat'):
        base_name = os.path.basename(feat_file)
        base_name = base_name[:-5]
        out_crf = os.path.join(crf_folder,base_name)
        logging.debug('Creating crf file in --> '+out_crf)
    
        try:
            extract_features_to_crf(feat_file,out_crf,fields,separator,templates_holder,possible_classes)
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
    if not os.path.exists(crfsuite):
        print>>sys.stderr,'CRFsuite not found on',crfsuite
        print>>sys.stderr,'Check the config filename and make sure the path is correctly set'
        print>>sys.stderr,'[crfsuite]\npath_to_binary = yourpathtolocalcrfsuite'
        sys.exit(-1)
        
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
    
    if not os.path.exists(svmlight):
        print>>sys.stderr,'SVMlight learn not found on',svmlight
        print>>sys.stderr,'Check the config filename and make sure the path is correctly set'
        print>>sys.stderr,'[svmlight]\npath_to_binary_learn = yourpathtolocalsvmlightlearn'
        sys.exit(-1)
        
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
    
def train_all(file_config):
  

      
    
    # Check if the output folder exists or create it
    create_folders(file_config)
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
    
    
if __name__ == '__main__':
    logging.basicConfig(stream=sys.stderr,format='%(asctime)s - %(levelname)s\n %(message)s', level=logging.DEBUG)
    file_config = sys.argv[1]
    train_all(file_config)
           
    sys.exit(0)