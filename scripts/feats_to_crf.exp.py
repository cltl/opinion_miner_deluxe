#!/usr/bin/env python

# Separator of field values.
separator = '\t'

# Field names of the input data.
# From file extract_feats_from_kaf
# print sentence_id+'\t'+token_id+'\t'+token+'\t'+term_id+'\t'+lemma+'\t'+pos+'\t'+entity_for_token+'\t'+property_for_token+'\t'+class_for_token

#9	wop140	competitor	t141	competitor	NN		negative	O      

#fields = 'sentence_id token_id tok term_id lem pos pol train_pol y'
#fields = 'sentence_id token_id tok term_id lem pos pol y'
fields = 'sentence_id token_id tok term_id lem pos polope polmpqa poltra y'
fields = 'sentence_id token_id tok term_id lem pos polmpqa y'
# Attribute templates.

templates = (
  #(('tok',-4),), (('lem',-4),), (('polmpqa',-4),),
  #(('tok',-3),), (('lem',-3),), (('polmpqa',-3),),
  #(('tok',-2),), (('lem',-2),), (('pos',-2),),(('polmpqa',-2),), #(('poltra',-2),),(('polope',-2),),
  #(('tok',-4),), (('lem',-4),), (('pos',-4),),(('polmpqa',-4),),
  #(('tok',-3),), (('lem',-3),), (('pos',-3),),(('polmpqa',-3),),
  #(('tok',-2),), (('lem',-2),), (('pos',-2),),(('polmpqa',-2),),
  (('tok',-1),), (('lem',-1),), (('pos',-1),),(('polmpqa',-1),), #(('poltra',-1),),(('polope',-1),),
  (('tok',0),), (('lem',0),), (('pos',0),),(('polmpqa',0),), #(('poltra',0),),(('polope',0),),
  (('tok',1),),  (('lem',1),), (('pos',1),),(('polmpqa',1),), #(('poltra',1),),(('polope',1),),
  #(('tok',2),), (('lem',2),), (('pos',2),),(('polmpqa',2),),
  #(('tok',3),), (('lem',3),), (('pos',3),),(('polmpqa',3),),
  #(('tok',4),), (('lem',4),), (('pos',4),),(('polmpqa',4),),
  #(('tok',2),), (('lem',2),), (('pos',2),),(('polmpqa',2),), #(('poltra',2),),(('polope',2),),
  #(('tok',3),), (('lem',3),), (('polmpqa',3),),
  #(('tok',4),), (('lem',4),), (('polmpqa',4),),
  )


templates1234 = (
  (('tok',-1),), (('pos',-1),), (('lem',-1),),(('train_pol',-1),) , (('pol',-1),),
  (('tok',0),), (('pos',0),), (('lem',0),),(('train_pol',0),) , (('pol',0),),
  (('tok',1),), (('pos',1),), (('lem',1),), (('train_pol',1),) , (('pol',1),),
  )
  


templates_default = (
    (('tok', -1), ), (('pos', -1), ), (('lem',  -1), ), (('pol',  -1), ),
    (('tok', 0), ), (('pos', 0), ), (('lem',  0), ), (('pol',  0), ),  
    (('tok', 1), ), (('pos', 1), ), (('lem',  1), ), (('pol',  1), ),
    (('tok',-1),('tok',0)),(('pos',-1),('pos',0)),  (('lem',-1),('lem',0)), (('pol',-1),('pol',0)),
    (('tok',0),('tok',1)),(('pos',0),('pos',1)), (('lem',0),('lem',1)), (('pol',0),('pol',1)),
    )


templates2222 = (
 # (('tok', -5), ), (('lem', -5), ),(('pol', -5), ), (('train_pol',-5),),
  (('tok', -4), ), (('lem', -4), ),(('pol', -4), ),(('train_pol',-4),),
  (('tok', -3), ), (('lem', -3), ),(('pol', -3), ),(('train_pol',-3),),
  (('tok', -2), ), (('lem', -2), ),(('pol', -2), ), (('train_pol',-2),),
  (('tok', -1), ), (('lem', -1), ),(('pol', -1), ), (('train_pol',-1),),
  (('tok', 0), ), (('lem', 0), ),(('pol', 0), ), (('pos', 0),), (('train_pol',0),),
  (('tok', 1), ), (('lem', 1), ),(('pol', 1), ), (('train_pol',1),),
  (('tok', 2), ), (('lem', 2), ),(('pol', 2), ), (('train_pol',2),),
  (('tok', 3), ),  (('lem', 3), ), (('pol', 3), ),(('train_pol',3),),
  (('tok', 4), ),  (('lem', 4), ), (('pol', 4), ),(('train_pol',4),),
  (('tok', +5), ), (('lem', +5), ),(('pol', +5), ),(('train_pol',5),),
  )

templates22 = (
  (('tok', -5), ), (('lem', -5), ),(('pol', -5), ),
  (('tok', -4), ), (('lem', -4), ),(('pol', -4), ),
  (('tok', -3), ), (('lem', -3), ),(('pol', -3), ),
  (('tok', -2), ), (('lem', -2), ),(('pol', -2), ), (('pos', -2),),
  (('tok', -1), ), (('lem', -1), ),(('pol', -1), ), (('pos', -1),),
  (('tok', 0), ), (('lem', 0), ),(('pol', 0), ), (('pos', 0),), 
  (('tok', 1), ), (('lem', 1), ),(('pol', 1), ), (('pos', 1),),
  (('tok', 2), ), (('lem', 2), ),(('pol', 2), ), (('pos', 2),),
  (('tok', 3), ),  (('lem', 3), ), (('pol', 3), ),
  (('tok', 4), ),  (('lem', 4), ), (('pol', 4), ),
  (('tok', +5), ), (('lem', +5), ),(('pol', +5), ),
  )
  
import crfutils

def feature_extractor(X):
    # Apply attribute templates to obtain features (in fact, attributes)
    crfutils.apply_templates(X, templates)
    if X:
	# Append BOS and EOS features manually
        X[0]['F'].append('__BOS__')     # BOS feature
        X[-1]['F'].append('__EOS__')    # EOS feature



def extract_features(inputfile,outputfile):
  fi = open(inputfile,'r')
  fo = open(outputfile,'w')
  crfutils.main(feature_extractor,fields=fields,sep=separator,fi=fi,fo=fo)
  fi.close()
  fo.close()
  

if __name__ == '__main__':
    crfutils.main(feature_extractor, fields=fields, sep=separator)


