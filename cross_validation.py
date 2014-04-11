#!/usr/bin/env python

import argparse
import sys
import ConfigParser
import os
import shutil
import time

from subprocess import Popen, PIPE
from generate_folds import generate_folds
from train import train_all
from classify_kaf_naf_file import tag_file_with_opinions
from KafNafParserPy import KafNafParser


## CHANGES
# 1stApril2014 --> Included evaluation for the opinion miner basic

default_jar_file='/home/izquierdo/code/triple_evaluation/lib/TripleEvaluation-1.0-jar-with-dependencies.jar'
path_to_basic = '/home/izquierdo/opener_repos/opinion-detector-basic/core/opinion_detector_basic_multi.py'

'''
java -Xmx812m -cp $eval_jar_file \
          vu.tripleevaluation.conversion.ConvertKafToTriples \
          --kaf-file $1 --opinion
'''
     
def convert_to_triple(jar_file,input_file):
    cmd = ['java -Xmx812m']
    cmd.append('-cp')
    cmd.append(jar_file)
    cmd.append('vu.tripleevaluation.conversion.ConvertKafToTriples')
    cmd.append('--kaf-file')
    cmd.append(input_file)
    cmd.append('--opinion')
    converter = Popen(' '.join(cmd),shell=True, stderr=sys.stderr)
    converter.wait()
    
'''
java -Xmx812m -cp $eval_jar_file \
              vu.tripleevaluation.evaluation.EvaluateTriples \
              --gold-standard-triples $1 \
              --system-triples $2 --ignore-relation
'''
   
## The output is generated in the same folder of the system_triples
## with the extension .xls and .log 
def evaluate_triples(jar_file,gold_triples,system_triples,evaluation_folder):
    # We need to run it 3 times to
    #  1) Get the prec/rec for opinion expression
    #  2) Get the prec/rec for target
    #  3) Get the prec/red for holder 
    runs = []
    runs.append(([],'.expression'))
    runs.append((['--element-second-filter','target'],'.target'))
    runs.append((['--element-second-filter','holder:'],'.holder'))   # The : is not an error!
    for extra_opts, extension in runs:
        cmd = ['java -Xmx812m']
        cmd.append('-cp')
        cmd.append(jar_file)
        cmd.append('vu.tripleevaluation.evaluation.EvaluateTriples')
        cmd.append('--gold-standard-triples')
        cmd.append(gold_triples)
        cmd.append('--system-triples')
        cmd.append(system_triples)
        cmd.extend(extra_opts)
    
               
        converter = Popen(' '.join(cmd),shell=True, stderr=PIPE, stdout=PIPE)
        converter.wait()
        
        # The output must be on system_triples.trp and .log
        # Move it to the evaluation folder with the proper extension
        base_file = os.path.basename(system_triples).replace('.trp','')
        shutil.move(system_triples+'.xls', evaluation_folder+'/'+base_file+extension)
        shutil.move(system_triples+'.log', evaluation_folder+'/'+base_file+extension+'.log')


# This function reads all the opinions from the inputfile, maps the opinion labels
# with the mapped givenn in config_file:
# [valid_opinions]
#negative = Negative;StrongNegative
#positive = Positive;StrongPositive
#
# So all Negative,StrongNEgative are mapped to negative
# and writes the output to output_file
def map_opinion_labels(input_file,output_file,config_file):
    # Load the mapping from the config_file
    mapping = {}
    parser = ConfigParser.ConfigParser()
    parser.read(config_file)
    for mapped_opinion, values_in_corpus in parser.items('valid_opinions'):
        values = [ v for v in values_in_corpus.split(';') if v != '']
        for v in values:
            mapping[v] = mapped_opinion
    del parser
    ##################        
    
    input_kaf = KafNafParser(input_file)
    for opinion in input_kaf.get_opinions():
        exp = opinion.get_expression()
        polarity = exp.get_polarity()
        mapped_polarity = mapping[polarity]
        exp.set_polarity(mapped_polarity)
    input_kaf.dump(output_file)
    
    
def extract_figures(evaluation_file):
    fic = open(evaluation_file)
    prec_first = rec_first = prec_second = rec_second = prec_rel = rec_rel = 0
    in_rel_section = False
    for line in fic:
        if line.find('Precision of first elements') != -1:
            fields = line.strip().split('\t')
            prec_first = fields[1]
            if prec_first=='NaN': prec_first='0'
        elif line.find('Recall of first elements') != -1:
            fields = line.strip().split('\t')
            rec_first = fields[1]
            if rec_first=='NaN': rec_first='0'
        if line.find('Precision of second elements') != -1:
            fields = line.strip().split('\t')
            prec_second = fields[1]
            if prec_second=='NaN': prec_second='0'
        elif line.find('Recall of second elements') != -1:
            fields = line.strip().split('\t')
            rec_second = fields[1]
            if rec_second=='NaN': rec_second='0'   
        elif line.find('Results per relation') != -1:
            in_rel_section = True       
        elif in_rel_section and line.startswith('Total'):
            fields = line.strip().split('\t')
            rec_rel = fields[-2]
            prec_rel = fields[-1]
    fic.close()
    return float(prec_first)*100,float(rec_first)*100,float(prec_second)*100,float(rec_second)*100,float(prec_rel),float(rec_rel)


#This functions calls to the 
def run_basic_version(input_file,output_file):
  cmd = path_to_basic
  fin = open(input_file,'r')
  fout = open(output_file,'w')
  basic_opinion_miner = Popen(cmd,stdin=fin, stdout=fout,stderr=PIPE,shell=True)
  fin.close()
  basic_opinion_miner.wait()
  fout.close()

    
        
if __name__ == '__main__':
    #map_opinion_labels('input.kaf','output.kaf','models/hotel_set1_set2_nl_validation/fold_0/config.cfg')
    #sys.exit(0)
        
    import logging
    logging.basicConfig(stream=sys.stderr,format='%(asctime)s - %(levelname)s\n %(message)s', level=logging.DEBUG)

    
    argument_parser = argparse.ArgumentParser(description="Perform fold-cross validation on the opinion mining system")
    argument_parser.add_argument('-n', type=int, default=10, dest='num_folds',help="Num folds (default 10)")
    argument_parser.add_argument('--config_file','-f',dest='config_file', required=True, help='Configuration file')
    argument_parser.add_argument('--eval_jar_file', action='store',dest='eval_jar_file', default=default_jar_file,help='Path to triple evaluation jar file (Default '+default_jar_file+')')
    
    arguments = argument_parser.parse_args()
    
    ##
    corpus_filename = None
    output_folder = None
    ##
    
    ## Get the list of KAF files from the config filename
    my_config = ConfigParser.ConfigParser()
    my_config.read(arguments.config_file)
    
    if not my_config.has_option('general', 'filename_training_list'):
        print>>sys.stderr,'[general] => filename_training_list option missing in ',config_file
        sys.exit(-1)
      
    if not my_config.has_option('general','output_folder'):
        print>>sys.stderr,'[general] => output_folder option missing in ',config_file
        sys.exit(-1)
        
    #This is the file with the list of KAF/NAf files for training
    corpus_filename = my_config.get('general', 'filename_training_list')    
    output_folder = my_config.get('general','output_folder')

    
    # Creating the folds in this way:
    # output_folder/fold_[0-9]+    
    
    #With this we generate new folds and so on
    generate_folds(corpus_filename,arguments.num_folds,output_folder)
    
    
    ## With this we copy it from a referencre created split of folds
    '''
    reference = '/home/izquierdo/cltl_repos/opinion_miner_deluxe/models/hotel_nl_trainandtest_all'
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    shutil.copytree(reference, output_folder)
    '''
    
    ## Process each fold
    
    #For the deluxe
    all_files = 0
    all_p_e = all_r_e = all_p_t = all_r_t = all_p_h = all_r_h = all_p_e_t = all_r_e_t = all_p_e_h = all_r_e_h = 0
    
    #For the basic
    all_files_basic = 0
    all_p_e_basic = all_r_e_basic = all_p_t_basic = all_r_t_basic = all_p_h_basic = all_r_h_basic = 0
    all_p_e_t_basic = all_r_e_t_basic = all_p_e_h_basic = all_r_e_h_basic = 0    
    for num in range(arguments.num_folds):
        this_folder = output_folder+'/fold_'+str(num)
        this_model_folder = this_folder+'/models'
        this_train_file = this_folder+'/train'
        this_config = this_folder+'/config.cfg'
        ##this_log = this_folder+'/train.log'        
        
        #Modifying the configuration and saving it to the proper file
        my_config.set('general','output_folder',this_model_folder)
        my_config.set('general', 'filename_training_list',this_train_file)
        fp = open (this_config,'w')
        my_config.write(fp)
        fp.close()
        ##################
        
        # We need to run now the training for this folder
        #def train_all(file_config):
        print 'Training'
        print '  Folder',this_folder,

        this_pid = os.fork()
        if this_pid == 0:
            train_all(this_config)
            sys.exit(0)
        else:
            while True:
                sys.stdout.write('.')
                sys.stdout.flush()
                status = os.waitpid(this_pid,os.WNOHANG)
                if status[0] != 0:   break
                time.sleep(1)
            
        #####
        ## Do the evaluation
        ####
        

        folder_out_kafs = this_folder+'/output_files'
        if os.path.exists(folder_out_kafs):
            shutil.rmtree(folder_out_kafs)
        os.mkdir(folder_out_kafs)
        
        folder_out_kafs_basic = this_folder+'/output_files_basic'
        if os.path.exists(folder_out_kafs_basic):
            shutil.rmtree(folder_out_kafs_basic)
        os.mkdir(folder_out_kafs_basic)
        
        folder_gold_triples = this_folder+'/gold_triples'
        if os.path.exists(folder_gold_triples): shutil.rmtree(folder_gold_triples)
        os.mkdir(folder_gold_triples)
        
        fold_test_corpora = this_folder+'/test'
        fold_test_corpora_desc = open(fold_test_corpora,'r')
        
        evaluation_folder = this_folder+'/evaluation'
        if os.path.exists(evaluation_folder): 
            shutil.rmtree(evaluation_folder)
        os.mkdir(evaluation_folder)

        evaluation_folder_basic = this_folder+'/evaluation_basic'
        if os.path.exists(evaluation_folder_basic): 
            shutil.rmtree(evaluation_folder_basic)
        os.mkdir(evaluation_folder_basic)

        list_triple_files = []  #Pairs (gold,out)
        list_test_base_files = []
        for input_test_file in fold_test_corpora_desc:
            input_test_file = input_test_file.strip()
            basename_file = os.path.basename(input_test_file)
            list_test_base_files.append(basename_file)
            
            # We copy the input KAF file into the gold triple folder doing a mapping
            # of the opinion labels as the input might contain Positive, Negative, Strong
            # and the output of the system just positive and neative (internal mapping)
            
            map_opinion_labels(input_test_file,folder_gold_triples+'/'+basename_file,this_config)
            print 'Processing ',basename_file
            
            input_test_file  = folder_gold_triples+'/'+basename_file 
            gold_triple_file = folder_gold_triples+'/'+basename_file+'.trp'
            
            
            convert_to_triple(arguments.eval_jar_file, input_test_file)
            print '  Created triple gold in',os.path.basename(gold_triple_file)
            
            ##########################
            #For the deluxe version  #
            ##########################
            output_test_file = folder_out_kafs+'/'+basename_file
            output_triple_file = folder_out_kafs+'/'+basename_file+'.trp'
            
            
            tag_file_with_opinions(input_test_file,output_test_file,this_model_folder,remove_existing_opinions=True,include_polarity_strength=False)
            print '  Classified DELUXE in ',os.path.basename(output_test_file)
                        
            convert_to_triple(arguments.eval_jar_file,output_test_file)
            print '  Created triple deluxe system in',os.path.basename(output_triple_file)    
            
            # Run the evaluation     
            evaluate_triples(arguments.eval_jar_file, gold_triple_file, output_triple_file, evaluation_folder)
            print '  Evaluated deluxe on',evaluation_folder
            #########################
            
            ##########################
            #For the basic version   #
            ##########################   
            output_test_file_basic = folder_out_kafs_basic+'/'+basename_file
            output_triple_file_basic = folder_out_kafs_basic+'/'+basename_file+'.trp' 
            
            #Call to the opinion miner basic and generate output_test_file_basic
            run_basic_version(input_test_file,output_test_file_basic)
            print '  Classified BASIC in ',os.path.basename(output_test_file_basic)
        
            convert_to_triple(arguments.eval_jar_file,output_test_file_basic)
            print '  Created triple basic system in',os.path.basename(output_triple_file_basic)
            
            # Run the evaluation     
            evaluate_triples(arguments.eval_jar_file, gold_triple_file, output_triple_file_basic, evaluation_folder_basic)
            print '  Evaluated basic on',evaluation_folder_basic
            #########################  
            
                    

        fold_test_corpora_desc.close()
        
        ## Extract the figures from the excel files
        #DELUXE
        num_files = 0
        over_p_e = over_r_e = over_p_t = over_r_t = over_p_h = over_r_h = over_p_e_t = over_r_e_t = over_p_e_h = over_r_e_h = 0
        for test_file in list_test_base_files:
            expression_eval_file = evaluation_folder+'/'+test_file+'.expression'
            target_eval_file = evaluation_folder+'/'+test_file+'.target'
            holder_eval_file = evaluation_folder+'/'+test_file+'.holder'
            
            prec_exp, rec_exp, _, _, _, _ = extract_figures(expression_eval_file)
            _,_, prec_tar,rec_tar, prec_rel_exp_tar, rec_rel_exp_tar = extract_figures(target_eval_file)
            _,_, prec_hol, rec_hol, prec_rel_exp_hol, rec_rel_exp_hol = extract_figures(holder_eval_file)
            
            over_p_e += prec_exp
            over_r_e += rec_exp
            over_p_t += prec_tar
            over_r_t += rec_tar
            over_p_h += prec_hol 
            over_r_h += rec_hol
            over_p_e_t += prec_rel_exp_tar
            over_r_e_t += rec_rel_exp_tar
            over_p_e_h += prec_rel_exp_hol
            over_r_e_h += rec_rel_exp_hol
            num_files += 1
          
 
        #BASIC
        num_files_basic = 0
        over_p_e_basic = over_r_e_basic = over_p_t_basic = over_r_t_basic = over_p_h_basic = 0
        over_r_h_basic = over_p_e_t_basic = over_r_e_t_basic = over_p_e_h_basic = over_r_e_h_basic = 0
        for test_file in list_test_base_files:
            expression_eval_file = evaluation_folder_basic+'/'+test_file+'.expression'
            target_eval_file = evaluation_folder_basic+'/'+test_file+'.target'
            holder_eval_file = evaluation_folder_basic+'/'+test_file+'.holder'
            
            prec_exp, rec_exp, _, _, _, _ = extract_figures(expression_eval_file)
            _,_, prec_tar,rec_tar, prec_rel_exp_tar, rec_rel_exp_tar = extract_figures(target_eval_file)
            _,_, prec_hol, rec_hol, prec_rel_exp_hol, rec_rel_exp_hol = extract_figures(holder_eval_file)
            
            over_p_e_basic += prec_exp
            over_r_e_basic += rec_exp
            over_p_t_basic += prec_tar
            over_r_t_basic += rec_tar
            over_p_h_basic += prec_hol 
            over_r_h_basic += rec_hol
            over_p_e_t_basic += prec_rel_exp_tar
            over_r_e_t_basic += rec_rel_exp_tar
            over_p_e_h_basic += prec_rel_exp_hol
            over_r_e_h_basic += rec_rel_exp_hol
            num_files_basic += 1       
         
        print
        print '#'*30
        print 'OVERALL DELUXE RESULTS FOR FOLD NUM',num
        print 'Num files:',num_files
        print '  Expression:'
        print '    Precision:',over_p_e*1.0/num_files
        print '    Recall:   ',over_r_e*1.0/num_files
        print '  Target:'
        print '    Precision:',over_p_t*1.0/num_files
        print '    Recall:   ',over_r_t*1.0/num_files
        print '  Holder:'
        print '    Precision:',over_p_h*1.0/num_files
        print '    Recall:   ',over_r_h*1.0/num_files
        print '  Relation:'
        print '    Exp-Tar'
        print '      Prec:',over_p_e_t*1.0/num_files
        print '      Rec: ',over_r_e_t*1.0/num_files
        print '    Exp-Hol'
        print '      Prec:',over_p_e_h*1.0/num_files
        print '      Rec: ',over_r_e_h*1.0/num_files
        print '#'*30
        print 
        all_files +=  num_files
        all_p_e +=over_p_e
        all_r_e += over_r_e
        all_p_t += over_p_t
        all_r_t += over_r_t
        all_p_h += over_p_h
        all_r_h += over_r_h
        all_p_e_t += over_p_e_t
        all_r_e_t += over_r_e_t
        all_p_e_h += over_p_e_h
        all_r_e_h += over_r_e_h
        
        print
        print '#'*30
        print 'OVERALL BASIC RESULTS FOR FOLD NUM',num
        print 'Num files:',num_files_basic
        print '  Expression:'
        print '    Precision:',over_p_e_basic*1.0/num_files_basic
        print '    Recall:   ',over_r_e_basic*1.0/num_files_basic
        print '  Target:'
        print '    Precision:',over_p_t_basic*1.0/num_files_basic
        print '    Recall:   ',over_r_t_basic*1.0/num_files_basic
        print '  Holder:'
        print '    Precision:',over_p_h_basic*1.0/num_files_basic
        print '    Recall:   ',over_r_h_basic*1.0/num_files_basic
        print '  Relation:'
        print '    Exp-Tar'
        print '      Prec:',over_p_e_t_basic*1.0/num_files_basic
        print '      Rec: ',over_r_e_t_basic*1.0/num_files_basic
        print '    Exp-Hol'
        print '      Prec:',over_p_e_h_basic*1.0/num_files_basic
        print '      Rec: ',over_r_e_h_basic*1.0/num_files_basic
        print '#'*30
        print 
        all_files_basic +=  num_files_basic
        all_p_e_basic +=over_p_e_basic
        all_r_e_basic += over_r_e_basic
        all_p_t_basic += over_p_t_basic
        all_r_t_basic += over_r_t_basic
        all_p_h_basic += over_p_h_basic
        all_r_h_basic += over_r_h_basic
        all_p_e_t_basic += over_p_e_t_basic
        all_r_e_t_basic += over_r_e_t_basic
        all_p_e_h_basic += over_p_e_h_basic
        all_r_e_h_basic += over_r_e_h_basic
        break
        #if num==3:
        #    break
    
        
    ## NEXT FOLD!!!   
    print
    print '#'*30
    print 'FINAL DELUXE OVERALL RESULTS FOR ALL FOLDS (microaverage)'
    print 'Total Num files:',all_files
    print 'Total num folds:',num
    print '  Expression:'
    print '    Precision:',all_p_e*1.0/all_files
    print '    Recall:   ',all_r_e*1.0/all_files
    print '  Target:'
    print '    Precision:',all_p_t*1.0/all_files
    print '    Recall:   ',all_r_t*1.0/all_files
    print '  Holder:'
    print '    Precision:',all_p_h*1.0/all_files
    print '    Recall:   ',all_r_h*1.0/all_files
    print '  Relation:'
    print '    Exp-Tar'
    print '      Prec:',all_p_e_t*1.0/all_files
    print '      Rec: ',all_r_e_t*1.0/all_files
    print '    Exp-Hol'
    print '      Prec:',all_p_e_h*1.0/all_files
    print '      Rec: ',all_r_e_h*1.0/all_files
    print '#'*30
    print    
    
    print 'LATEX DELUXE'
    print '\\begin{table}'
    print '\\begin{tabular}{c|c|c|}'
    print '\\hline'
    print 'Type & Precision & Recall\\\\'
    print '\\hline'
    print 'Expression & %.2f & %.2f\\\\' % (all_p_e*1.0/all_files,all_r_e*1.0/all_files)
    print 'Target & %.2f & %.2f \\\\' % (all_p_t*1.0/all_files,all_r_t*1.0/all_files)
    print 'Holder & %.2f & %.2f \\\\' % (all_p_h*1.0/all_files,all_r_h*1.0/all_files)
    print '\\hline'
    print 'Exp-Tar & %.2f & %.2f\\\\' % (all_p_e_t*1.0/all_files,all_r_e_t*1.0/all_files)
    print 'Exp-Hol & %.2f & %.2f\\\\' % (all_p_e_h*1.0/all_files,all_r_e_h*1.0/all_files)
    print '\\hline'
    print '\\end{tabular}'
    print '\\caption{Caption here}'
    print '\\end{table}'


    print
    print '#'*30
    print 'FINAL OVERALL BASIC RESULTS FOR ALL FOLDS (microaverage)'
    print 'Total Num files:',all_files_basic
    print 'Total num folds:',num
    print '  Expression:'
    print '    Precision:',all_p_e_basic*1.0/all_files_basic
    print '    Recall:   ',all_r_e_basic*1.0/all_files_basic
    print '  Target:'
    print '    Precision:',all_p_t_basic*1.0/all_files_basic
    print '    Recall:   ',all_r_t_basic*1.0/all_files_basic
    print '  Holder:'
    print '    Precision:',all_p_h_basic*1.0/all_files_basic
    print '    Recall:   ',all_r_h_basic*1.0/all_files_basic
    print '  Relation:'
    print '    Exp-Tar'
    print '      Prec:',all_p_e_t_basic*1.0/all_files_basic
    print '      Rec: ',all_r_e_t_basic*1.0/all_files_basic
    print '    Exp-Hol'
    print '      Prec:',all_p_e_h_basic*1.0/all_files_basic
    print '      Rec: ',all_r_e_h_basic*1.0/all_files_basic
    print '#'*30
    print    
    
    print 'LATEX BASIC'
    print '\\begin{table}'
    print '\\begin{tabular}{c|c|c|}'
    print '\\hline'
    print 'Type & Precision & Recall\\\\'
    print '\\hline'
    print 'Expression & %.2f & %.2f\\\\' % (all_p_e_basic*1.0/all_files_basic,all_r_e_basic*1.0/all_files_basic)
    print 'Target & %.2f & %.2f \\\\' % (all_p_t_basic*1.0/all_files_basic,all_r_t_basic*1.0/all_files_basic)
    print 'Holder & %.2f & %.2f \\\\' % (all_p_h_basic*1.0/all_files_basic,all_r_h_basic*1.0/all_files_basic)
    print '\\hline'
    print 'Exp-Tar & %.2f & %.2f\\\\' % (all_p_e_t_basic*1.0/all_files_basic,all_r_e_t_basic*1.0/all_files_basic)
    print 'Exp-Hol & %.2f & %.2f\\\\' % (all_p_e_h_basic*1.0/all_files_basic,all_r_e_h_basic*1.0/all_files_basic)
    print '\\hline'
    print '\\end{tabular}'
    print '\\caption{Caption here}'
    print '\\end{table}'
     
        
        
        
        
    
    # Proceed with the evaluation
    
