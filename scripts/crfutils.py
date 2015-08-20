import optparse
import sys



def feature_extractor(X,templates):
    # Apply attribute templates to obtain features (in fact, attributes)
    apply_templates(X, templates)
    
    for t in range(len(X)):
        BOW(X,t,'token',-4,-1)
        BOW(X,t,'token',1,4)
    if X:
        # Append BOS and EOS features manually
        X[0]['F'].append('__BOS__')     # BOS feature
        X[-1]['F'].append('__EOS__')    # EOS feature
        
    

def BOW(X, t, field, begin, end):
    name = 'bow%s[%d..%d]' % (field, begin, end)
    for offset in range(begin, end+1):
        p = t + offset
        if p not in range(0, len(X)):
            continue
        X[t]['F'].append('%s=%s' % (name, X[p][field]))



def extract_features_to_crf(inputfile,outputfile,fields,separator,templates,possible_classes=None):
  fi = open(inputfile,'r')
  fo = open(outputfile,'w')
  #this_main(feature_extractor,templates=templates,fields=fields,sep=separator,fi=fi,fo=fo,possible_classes=possible_classes)
  
  F = fields.split(' ')
  for X in readiter(fi, F, separator):
      feature_extractor(X,templates)
      output_features(fo, X, 'y',possible_classes)

  fi.close()
  fo.close()

def apply_templates(X, templates):

    """
    Generate features for an item sequence by applying feature templates.
    A feature template consists of a tuple of (name, offset) pairs,
    where name and offset specify a field name and offset from which
    the template extracts a feature value. Generated features are stored
    in the 'F' field of each item in the sequence.

    @type   X:      list of mapping objects
    @param  X:      The item sequence.
    @type   template:   tuple of (str, int)
    @param  template:   The feature template.
    """
    name=''
    for template in templates:
        '''
        name=''
        for f, o in template:
          if o<0: o=-1
          elif o>0: o=1
          name += f+'['+str(o)+']'+'|'
        name = name[:-1]  
        '''
        name = '|'.join(['%s[%d]' % (f, o) for f, o in template])
        for t in range(len(X)):
            values = []
            for field, offset in template:
                p = t + offset
                if p not in range(len(X)):
                    values = []
                    break
                value = X[p][field]
                if value != '' and value !='-':
                  values.append(value)
            if values and len(values)>0 :
                X[t]['F'].append('%s=%s' % (name, '|'.join(values)))

def readiter(fi, names, sep=' '):
    """
    Return an iterator for item sequences read from a file object.
    This function reads a sequence from a file object L{fi}, and
    yields the sequence as a list of mapping objects. Each line
    (item) from the file object is split by the separator character
    L{sep}. Separated values of the item are named by L{names},
    and stored in a mapping object. Every item has a field 'F' that
    is reserved for storing features.

    @type   fi:     file
    @param  fi:     The file object.
    @type   names:  tuple
    @param  names:  The list of field names.
    @type   sep:    str
    @param  sep:    The separator character.
    @rtype          list of mapping objects
    @return         An iterator for sequences.
    """
    X = []
    for line in fi:
        line = line.strip('\n')
        if not line:
            yield X
            X = []
        else:
            fields = line.split(sep)
            if len(fields) < len(names):
                raise ValueError(
                    'Too few fields (%d) for %r\n%s' % (len(fields), names, line))
            item = {'F': []}    # 'F' is reserved for features.
            for i in range(len(names)):
                item[names[i]] = fields[i]
            X.append(item)

def escape(src):
    """
    Escape colon characters from feature names.

    @type   src:    str
    @param  src:    A feature name
    @rtype          str
    @return         The feature name escaped.
    """
    return src.replace(':', '__COLON__')

def output_features(fo, X, field='', possible_classes=None):
    """
    Output features (and reference labels) of a sequence in CRFSuite
    format. For each item in the sequence, this function writes a
    reference label (if L{field} is a non-empty string) and features.

    @type   fo:     file
    @param  fo:     The file object.
    @type   X:      list of mapping objects
    @param  X:      The sequence.
    @type   field:  str
    @param  field:  The field name of reference labels.
    """
    for t in range(len(X)):
        if field:
            this_class = X[t][field]   ### B-positive
            general_class = this_class[2:]
            if possible_classes is not None and general_class not in possible_classes:
                this_class = 'O'                
            fo.write('%s' % this_class)
        for a in X[t]['F']:
            if isinstance(a, str):
                fo.write('\t%s' % escape(a))
            else:
                fo.write('\t%s:%f' % (escape(a[0]), a[1]))
        fo.write('\n')
    fo.write('\n')

def to_crfsuite(X):
    """
    Convert an item sequence into an object compatible with crfsuite
    Python module.

    @type   X:      list of mapping objects
    @param  X:      The sequence.
    @rtype          crfsuite.ItemSequence
    @return        The same sequence in crfsuite.ItemSequence type.
    """
    import crfsuite
    xseq = crfsuite.ItemSequence()
    for x in X:
        item = crfsuite.Item()
        for f in x['F']:
            if isinstance(f, str):
                item.append(crfsuite.Attribute(escape(f)))
            else:
                item.append(crfsuite.Attribute(escape(f[0]), f[1]))
        xseq.append(item)
    return xseq

'''
def this_main(feature_extractor, templates,fields='w pos y', sep=' ',fi = sys.stdin,fo = sys.stdout,possible_classes=None):

    # Parse the command-line arguments.
    parser = optparse.OptionParser(usage="""usage: %prog [options]
This utility reads a data set from STDIN, and outputs attributes to STDOUT.
Each line of a data set must consist of field values separated by SEPARATOR
characters. The names and order of field values can be specified by -f option.
The separator character can be specified with -s option. Instead of outputting
attributes, this utility tags the input data when a model file is specified by
-t option (CRFsuite Python module must be installed)."""
        )
    parser.add_option(
        '-t', dest='model',
        help='tag the input using the model (requires "crfsuite" module)'
        )
    parser.add_option(
        '-f', dest='fields', default=fields,
        help='specify field names of input data [default: "%default"]'
        )
    parser.add_option(
        '-s', dest='separator', default=sep,
        help='specify the separator of columns of input data [default: "%default"]'
        )
    (options, args) = parser.parse_args()

    # The fields of input: ('w', 'pos', 'y) by default.
    F = options.fields.split(' ')

    if not options.model:
        # The generator function readiter() reads a sequence from a 
        for X in readiter(fi, F, options.separator):
            feature_extractor(X,templates)
            output_features(fo, X, 'y',possible_classes)

    else:
        # Create a tagger with an existing model.
        import crfsuite
        tagger = crfsuite.Tagger()
        tagger.open(options.model)

        # For each sequence from STDIN.
        for X in readiter(fi, F, options.separator):
            # Obtain features.
            feature_extractor(X,templates)
            xseq = to_crfsuite(X)
            yseq = tagger.tag(xseq)
            for t in range(len(X)):
                v = X[t]
                fo.write('\t'.join([v[f] for f in F]))
                fo.write('\t%s\n' % yseq[t])
            fo.write('\n')
'''