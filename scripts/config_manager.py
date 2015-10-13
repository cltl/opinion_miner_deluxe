import os
import ConfigParser
import shutil

internal_config_filename= 'config.cfg'


def load_templates_from_file(filename):
    templates = []
    fic = open(filename,'r')
    for line in fic:
        line = line.strip()
        if line != '' and line[0]!='#': #Not empty lines or starting with #
            tokens = line.split(' ')
            my_len = int(tokens[0])
            labels = tokens[1:my_len+1]
            values = tokens[my_len+1:]
            for value in values:
                new_template = []
                single_values = value.split('/')
                for n in range(len(labels)):
                    new_template.append((labels[n],int(single_values[n])))
                templates.append(new_template)
    fic.close()
    return templates

class Cconfig_manager:
    def __init__(self):
        self.config = ConfigParser.ConfigParser()
        self.out_folder = None
        self.this_folder = None
        self.templates_expr = None
        self.templates_holder = None
        self.templates_target = None 
        
    def set_current_folder(self,t):
        self.this_folder = t
        
    def get_flag_filename(self):
        my_name = 'flag'
        return os.path.join(self.get_output_folder(),my_name)
    
    def set_config(self,file_cfg):
        self.config.read(file_cfg)
        output_folder_cfg = self.config.get('general','output_folder')
        out_folder = ''
        if os.path.isabs(output_folder_cfg):
            self.out_folder = output_folder_cfg
        else:
            self.out_folder = os.path.join(self.this_folder,output_folder_cfg)

    def get_use_dependencies(self):
        use_dependencies = True ##Default
        if self.config.has_section('relation_features'):
            if self.config.has_option('relation_features', 'use_dependencies'):
                use_dependencies = self.config.getboolean('relation_features', 'use_dependencies')
        return use_dependencies
        
        
    def get_use_training_lexicons(self):
        use_lexicons = True ##Default
        if self.config.has_section('relation_features'):
            if self.config.has_option('relation_features', 'use_training_lexicons'):
                use_lexicons = self.config.getboolean('relation_features', 'use_training_lexicons')
        return use_lexicons   
     
    def get_use_tokens_lemmas(self):
        use_them = True
        if self.config.has_section('relation_features'):
            if self.config.has_option('relation_features', 'use_tokens_lemmas'):
                use_them = self.config.getboolean('relation_features', 'use_tokens_lemmas')
        return use_them
    
    def get_propagation_lexicon_name(self):
        lexicon_name = None
        if self.config.has_section('lexicons'):
            if self.config.has_option('lexicons','propagation_lexicon'):
                lexicon_name = self.config.get('lexicons','propagation_lexicon')
        return lexicon_name
                
                
    def set_out_folder(self,o):
        self.out_folder = o
        
    def get_training_datasets_folder(self):
        my_name='training_datasets'
        outfolder=self.get_output_folder()
        return os.path.join(outfolder,my_name)
                                    
    def get_training_dataset_exp(self):
        my_name = 'training_set_exp.crf'
        return os.path.join(self.get_training_datasets_folder(),my_name)
  
    def get_training_dataset_target(self):
        my_name = 'training_set_target.crf'
        return os.path.join(self.get_training_datasets_folder(),my_name)   
    
    
    def get_training_dataset_holder(self):
        my_name = 'training_set_holder.crf'
        return os.path.join(self.get_training_datasets_folder(),my_name)  
    
    
    ## FEATURE TEMPLATES
    def get_feature_template_folder_name(self):
        my_name = 'feature_templates'
        return os.path.join(self.get_output_folder(),my_name)  
    
    def get_feature_template_exp_name(self):
        my_name = 'feat_template_expr.txt'
        return os.path.join(self.get_feature_template_folder_name(),my_name) 

    def get_feature_template_tar_name(self):
        my_name = 'feat_template_target.txt'
        return os.path.join(self.get_feature_template_folder_name(),my_name) 

    def get_feature_template_hol_name(self):
        my_name = 'feat_template_holder.txt'
        return os.path.join(self.get_feature_template_folder_name(),my_name)     
    
    def copy_feature_templates(self):
        #Exp
        temp_exp_orig = self.config.get('feature_templates','expression')
        temp_exp_target = self.get_feature_template_exp_name()
        if not os.path.isabs(temp_exp_orig):
            temp_exp_orig = os.path.join(self.this_folder,temp_exp_orig)
        shutil.copyfile(temp_exp_orig, temp_exp_target)

        temp_tar_orig = self.config.get('feature_templates','target')
        temp_tar_target = self.get_feature_template_tar_name()
        if not os.path.isabs(temp_tar_orig):
            temp_tar_orig = os.path.join(self.this_folder,temp_tar_orig)
        shutil.copyfile(temp_tar_orig, temp_tar_target)

        temp_hol_orig = self.config.get('feature_templates','holder')
        temp_hol_target = self.get_feature_template_hol_name()
        if not os.path.isabs(temp_hol_orig):
            temp_hol_orig = os.path.join(self.this_folder,temp_hol_orig)
        shutil.copyfile(temp_hol_orig, temp_hol_target)

    def get_templates_expr(self):
        if self.templates_expr is None:
            filename_template = self.get_feature_template_exp_name()
            self.templates_expr = load_templates_from_file(filename_template)
        return self.templates_expr

    def get_templates_holder(self):
        if self.templates_holder is None:
            filename_template = self.get_feature_template_hol_name()
            self.templates_holder = load_templates_from_file(filename_template)
        return self.templates_holder
    
    def get_templates_target(self):
        if self.templates_target is None:
            filename_template = self.get_feature_template_tar_name()
            self.templates_target = load_templates_from_file(filename_template)
        return self.templates_target    
    
    def get_lexicons_folder(self):
        my_name = 'lexicons'
        return os.path.join(self.get_output_folder(),my_name)
    
    ###############
    def get_expression_lexicon_filename(self):
        my_name = 'polarity_lexicon.csv'
        return os.path.join(self.get_lexicons_folder(),my_name)

    def get_use_this_expression_lexicon(self):
        use_it = None
        if self.config.has_section('relation_features'):
            if self.config.has_option('relation_features', 'use_this_expression_lexicon'):
                use_it = self.config.get('relation_features', 'use_this_expression_lexicon')
        return use_it
  
    def get_use_this_target_lexicon(self):
        use_it = None
        if self.config.has_section('relation_features'):
            if self.config.has_option('relation_features', 'use_this_target_lexicon'):
                use_it = self.config.get('relation_features', 'use_this_target_lexicon')
        return use_it  
    
    def get_target_lexicon_filename(self):
        my_name = 'target_lexicon.csv'
        return os.path.join(self.get_lexicons_folder(),my_name)    
    
    def get_feature_folder_name(self):
        subfolder_feats = 'tab_feature_files'  
        out_folder = self.get_output_folder()
        return os.path.join(out_folder,subfolder_feats)
        
    def get_crf_expression_folder(self):
        my_name='crf_feat_files_exp'
        out_folder = self.get_output_folder()
        return os.path.join(out_folder,my_name)
    
    def get_crf_target_folder(self):
        my_name='crf_feat_files_target'
        out_folder = self.get_output_folder()
        return os.path.join(out_folder,my_name)
    
    def get_crf_holder_folder(self):
        my_name='crf_feat_files_holder'
        out_folder = self.get_output_folder()
        return os.path.join(out_folder,my_name)        
    
    def get_output_folder(self):
        return self.out_folder
    
    def get_feature_desc_filename(self):
        file_feat_desc = 'feature_desc.txt'     #description of features
        out_folder = self.get_output_folder()
        return os.path.join(out_folder,file_feat_desc)
    
    def get_file_training_list(self):
        return self.config.get('general','filename_training_list')
    
    def get_crfsuite_binary(self):
        return self.config.get('crfsuite','path_to_binary')
    
    def get_crfsuite_params(self):
        return self.config.get('crfsuite','parameters')
    
    def get_svm_learn_binary(self):
        return self.config.get('svmlight','path_to_binary_learn')
    
    def get_svm_classify_binary(self):
        return self.config.get('svmlight','path_to_binary_classify')
    
    def get_svm_params(self):
        return self.config.get('svmlight','parameters')
    
    
    def get_svm_threshold_exp_tar(self):
        thr = -1
        if self.config.has_option('relation_features', 'exp_tar_threshold'):
            thr = self.config.getfloat('relation_features', 'exp_tar_threshold')
        return thr
  
    def get_svm_threshold_exp_hol(self):
        thr = -1
        if self.config.has_option('relation_features', 'exp_hol_threshold'):
            thr = self.config.getfloat('relation_features', 'exp_hol_threshold')
        return thr  
    
    
    # [valid_opinions]
    # positive = sentiment-neg
    # negative = sentiment-pos
    def get_mapping_valid_opinions(self):
        mapping = {}
        for mapped_opinion, values_in_corpus in self.config.items('valid_opinions'):
            values = [ v for v in values_in_corpus.split(';') if v != '']
            for v in values:
                mapping[v] = mapped_opinion
        return mapping
                
    def get_possible_expression_values(self):
        labels = [key for key,_ in self.config.items('valid_opinions')]
        return labels
    
    def get_model_foldername(self):
        my_name = 'models'
        out_folder = self.get_output_folder()
        return os.path.join(out_folder,my_name)
    
    def get_filename_model_expression(self):
        my_name = 'model_opi_exp.crf'
        return os.path.join(self.get_model_foldername(),my_name)
    
    def get_filename_model_target(self):
        my_name = 'model_opi_target.crf'
        return os.path.join(self.get_model_foldername(),my_name)  
    
    def get_filename_model_holder(self):
        my_name = 'model_opi_holder.crf'
        return os.path.join(self.get_model_foldername(),my_name)     
     
    def get_folder_relation_classifier(self):
        my_name = 'relation_classifier'
        return os.path.join(self.out_folder,my_name)
    
    def get_relation_exp_tar_training_filename(self):
        my_name = 'training_exp_tar.feat'
        return os.path.join(self.get_folder_relation_classifier(),my_name)

    def get_relation_exp_hol_training_filename(self):
        my_name = 'training_exp_hol.feat'
        return os.path.join(self.get_folder_relation_classifier(),my_name)
    
    def get_rel_exp_tar_training_idx_filename(self):
        my_name = 'training_exp_tar.idx'
        return os.path.join(self.get_folder_relation_classifier(),my_name)

    def get_rel_exp_hol_training_idx_filename(self):
        my_name = 'training_exp_hol.idx'
        return os.path.join(self.get_folder_relation_classifier(),my_name)
    
    def get_index_features_exp_tar_filename(self):
        my_name = 'feat_index.exp_tar.bin'
        return os.path.join(self.get_folder_relation_classifier(),my_name)
    
    def get_index_features_exp_hol_filename(self):
        my_name = 'feat_index.exp_hol.bin'
        return os.path.join(self.get_folder_relation_classifier(),my_name)

    def get_filename_model_exp_tar(self):
        my_name = 'model_relation_exp_tar.svmlight'
        return os.path.join(self.get_folder_relation_classifier(),my_name)
        
    def get_filename_model_exp_hol(self):
        my_name = 'model_relation_exp_hol.svmlight'
        return os.path.join(self.get_folder_relation_classifier(),my_name)

      
