#!/usr/bin/env python

import sys
from subprocess import Popen,PIPE

def run_basic(input_file,output_file):
  cmd = '/home/izquierdo/opener_repos/opinion-detector-basic/core/opinion_detector_basic_multi.py'
  fin = open(input_file,'r')
  fout = open(output_file,'w')
  basic_opinion_miner = Popen(cmd,stdin=fin, stdout=fout,stderr=PIPE,shell=True)
  fin.close()
  basic_opinion_miner.wait()
  fout.close()
  print 'Done'

if __name__ == '__main__':
  input = 'english00001_0123ff23e0d0dc0177f9b71a1928b674.kaf'
  output = 'english00001_0123ff23e0d0dc0177f9b71a1928b674.basic.kaf'
  run_basic(input,output)
  