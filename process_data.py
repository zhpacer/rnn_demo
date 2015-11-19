import sys
import pdb
import re
import nltk
import csv
from itertools import chain

voc_size = 8000
UNK = "UNK"
SB =  "SB"
SE = "SE"

print "Reading CVS file ..."
with open('data/reddit-comments-2015-08.csv','rb') as f:
    reader = csv.reader(f, skipinitialspace=True)
    reader.next()

    #line_set = [nltk.sent_tokenize(x[0].decode('utf-8').lower()) for x in reader]
    #for cur_line in line_set:
        #print "%s\n" % (cur_line)
    sentences = chain(*[nltk.sent_tokenize(x[0].decode('utf-8').lower()) for x in reader])
    sentences = ["%s %s %s" % (SB,x,SE) for x in sentences]

print "Parsed %d sentences.\n" % (len(sentences))
