#####
import sys
import logging
from operator import itemgetter


def get_min(l):
  min = None
  for ele in l:
    digits = ''
    for c in ele:
      if c.isdigit(): digits+=c
    value = int(digits)
    if min==None or value<min:
      min = value
  return min

#Returns the maximum position from a list of token ids
def get_max(l):
  max = -1
  for ele in l:
    digits = ''
    for c in ele:
      if c.isdigit(): digits+=c
    value = int(digits)
    if value>max:
      max = value
  return max


## Gets the distance in number of tokens between two lisf of ids
def get_distance(list1, list2):
  min_1 = get_min(list1)
  max_1 = get_max(list1)
  min_2 = get_min(list2)
  max_2 = get_max(list2)

  if max_1 < min_2:
    distance = min_2 - max_1
  elif max_2 < min_1:
    distance = min_1 - max_2
  else:
    distance = 0
  return distance

def link_entities_distance(expressions,targets,holders, sentence_for_token):
    triples = []
    weight_crossing_sentence = 200

    for exp_ids, type_exp in expressions:
        sentence_exp = int(sentence_for_token[exp_ids[0]])
    
        final_tar = []
        list_tar_dist = []
        for tar_ids, target_label in targets:
            sentence_tar = int(sentence_for_token[tar_ids[0]])
            dist_tar_exp = get_distance(exp_ids,tar_ids)
            final_distance = dist_tar_exp + weight_crossing_sentence * abs(sentence_exp - sentence_tar)
            list_tar_dist.append((tar_ids,final_distance))
        if len(list_tar_dist) != 0:
            list_tar_dist.sort(key=itemgetter(1))
            final_tar = list_tar_dist[0][0]
                                 
        final_hol = []
        list_hol_dist = []
        for hol_ids, target_label in holders:
            sentence_hol = int(sentence_for_token[hol_ids[0]])
            dist_hol_exp = get_distance(exp_ids,hol_ids)
            final_distance = dist_hol_exp + weight_crossing_sentence * abs(sentence_exp - sentence_hol)
            list_hol_dist.append((hol_ids,final_distance))
        if len(list_hol_dist) != 0:
            list_hol_dist.sort(key=itemgetter(1))
            final_hol = list_hol_dist[0][0]
        
        triples.append((type_exp,exp_ids,final_tar,final_hol))
    return triples             
 