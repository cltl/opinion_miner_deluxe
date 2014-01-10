import os
import ConfigParser


class Cconfig_manager:
    def __init__(self):
        self.config = ConfigParser.ConfigParser()
        self.out_folder = None
        self.this_folder = None
        
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
    
    def get_expression_template_filename(self):           
        my_name = 'template_crf_expression.bin'
        return os.path.join(self.out_folder,my_name)
    
    def get_target_template_filename(self):           
        my_name = 'template_crf_target.bin'
        return os.path.join(self.out_folder,my_name)

    def get_holder_template_filename(self):           
        my_name = 'template_crf_holder.bin'
        return os.path.join(self.out_folder,my_name)    
    
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
      
      
