# -*_ coding: UTF-8 -*-

import sys
import pdb
import re
import nltk
import csv
import operator
import numpy as np
import datetime
from itertools import chain

voc_size = 8000
UNK = "UNK"
SB =  "SB"
SE = "SE"

print 'Reading CVS file ...'

with open('data/reddit-comments-2015-08.csv','rb') as f:
    reader = csv.reader(f, skipinitialspace=True)
    reader.next()

    #line_set = [nltk.sent_tokenize(x[0].decode('utf-8').lower()) for x in reader]
    #for cur_line in line_set:
        #print "%s\n" % (cur_line)
    sentences = chain(*[nltk.sent_tokenize(x[0].decode('utf-8').lower()) for x in reader])
    sentences = ["%s %s %s" % (SB,x,SE) for x in sentences]

print "Parsed %d sentences.\n" % (len(sentences))


tokenized_sents = [nltk.word_tokenize(sent) for sent in sentences]

word_frq = word_frq= nltk.FreqDist(chain(*tokenized_sents))
print "Found %d unique words tokens.\n" % len(word_frq.items())

vocab = word_frq.most_common(voc_size-1)
index_to_word = [x[0] for x in vocab]
index_to_word.append(UNK)
word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])

print "Voc size %d\n" % voc_size
print "The least frequent word in our vocabulary is '%s' and appeared %d times\n" %(vocab[-1][0],vocab[-1][1])

for i,sent in enumerate(tokenized_sents):
    tokenized_sents[i] = [w if w in word_to_index else UNK for w in sent]

print "Example sent: '%s'\n" % sentences[0]
print "Example sent after pre_processing: '%s'\n" % tokenized_sents[0]


X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sents])
Y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sents])


print "X:'%s'\nY:'%s'\n" %(X_train[0],Y_train[0])



def softmax(x):
    #e_x = np.exp(x - np.max(x))
    e_x = np.exp(x)
    out = e_x / e_x.sum()
    return out




class RnnNumpy:
    def __init__(self,word_dim,hidden_dim=100,bptt_trunacate=4):
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_trunacate = bptt_trunacate


        self.U = np.random.uniform(-np.sqrt(1./word_dim),np.sqrt(1./word_dim),(hidden_dim,word_dim))
        self.V = np.random.uniform(-np.sqrt(1./hidden_dim),np.sqrt(1./hidden_dim),(word_dim,hidden_dim))
        self.W = np.random.uniform(-np.sqrt(1./hidden_dim),np.sqrt(1./hidden_dim),(hidden_dim,hidden_dim))

    def forward_propagation(self,x):
        T = len(x)

        s = np.zeros((T+1,self.hidden_dim))
        s[-1] = np.zeros(self.hidden_dim)

        o = np.zeros((T,self.word_dim))

        for t in np.arange(T):
            s[t] = np.tanh(self.U[:,x[t]]+self.W.dot(s[t-1]))
            o[t] = softmax(self.V.dot(s[t]))

        return [o,s]

    def predict(self,x):
        o,s = self.forward_propagation(x)
        return np.argmax(o,axis=1)

    def calc_total_loss(self,x,y):
        L = 0
        for i in np.arange(len(y)):
            o,s = self.forward_propagation(x[i])
            #print "o = "
            #print o.shape
            #np.arange(len_yi),get the size of y_i in o,y_i get the wanted item of o
            correct_word_predictions = o[np.arange(len(y[i])),y[i]]
            #print "len: %d" % (len(y[i]))
            #print  y[i]
            #print "correct_prediction"
            #print correct_word_predictions
            #print "\n"
            L += -1 * np.sum(np.log(correct_word_predictions))
        return L
    def calc_loss(self,x,y):

        N = np.sum(len(y_i) for y_i in y)
        return self.calc_total_loss(x,y)/N

    def bptt(self,x,y):
        T = len(y)
        o,s = self.forward_propagation(x)

        dLdU = np.zeros(self.U.shape)
        dLdV = np.zeros(self.V.shape)
        dLdW = np.zeros(self.W.shape)

        delta_o = o
        #get the output diff
        delta_o[np.arange(len(y)),y] -= 1.

        for t in np.arange(T)[::-1]:
            dLdV += np.outer(delta_o[t],s[t].T)

            delta_t = self.V.T.dot(delta_o[t]) * (1 - (s[t] ** 2))

            for bptt_step in np.arange(max(0,t-self.bptt_trunacate),t+1)[::-1]:

                dLdW += np.outer(delta_t,s[bptt_step-1])
                dLdU[:,x[bptt_step]] += delta_t

                delta_t = self.W.T.dot(delta_t) * (1-s[bptt_step-1]**2)
        return [dLdU,dLdV,dLdW]


    def gradient_check(self,x,y,h=0.001,error_threshold=0.01):

        bptt_gradients = self.bptt(x,y)
        model_parameter = ['U','V','W']

        for pidx,pname in enumerate(model_parameter):

            parameter = operator.attrgetter(pname)(self)
            print "Performing gradient check for parameter %s with size %d." % (pname,np.prod(parameter.shape))

            it = np.nditer(parameter,flags=['multi_index'],op_flags=['readwrite'])

            while not it.finished:
                ix = it.multi_index

                original_value = parameter[ix]

                parameter[ix] = original_value + h
                gradplus = self.calc_total_loss([x],[y])
                parameter[ix] = original_value - h
                gradminus = model.calc_total_loss([x],[y])
                estimated_gradient = (gradplus - gradminus)/(2*h)

                parameter[ix] = original_value

                backprop_gradient = bptt_gradients[pidx][ix]
                relative_error = np.abs(backprop_gradient-estimated_gradient)/(np.abs(backprop_gradient)+np.abs(estimated_gradient))

                if relative_error > error_threshold:
                    print "Gradient check ERROR: parameter=%s ix=%s " %(pname,ix)
                    print "+h loss %f" % gradplus
                    print "-h loss %f " % gradminus
                    print "estimated_gradient %f  backprop_gradient %f" % (estimated_gradient,backprop_gradient)
                    print "Relative Error: %f" % relative_error
                    return
                it.iternext()
            print "Gradient check for parameter %s passed" % (pname)

    def numpy_sdg_step(self,x,y,learning_rate):

        dLdU,dLdV,dLdW = self.bptt(x,y)

        self.U -= learning_rate * dLdU
        self.V -= learning_rate * dLdV
        self.W -= learning_rate * dLdW


    def train_with_sgd(self,X_train,Y_train,learning_rate=0.005,nepoch=100,evaluate_loss_after=5):

        losses = []

        num_examples_seen = 0

        for epoch in range(nepoch):

            if(epoch % evaluate_loss_after == 0):
                loss = self.calc_loss(X_train,Y_train)
                losses.append((num_examples_seen,loss))
                time = datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
                print "%s: Loss after num_examples_seen=%d epoch=%d: %f" % (time,num_examples_seen,epoch,loss)

                if(len(losses) > 1 and losses[-1][1] > losses[-2][1]):
                    learning_rate = learning_rate * 0.5
                    print "Setting learning_rate to %f" % learning_rate
                sys.stdout.flush()

            for i in range(len(Y_train)):
                self.numpy_sdg_step(X_train[i],Y_train[i],learning_rate)
                num_examples_seen += 1


    def generate_sentence(self):
        new_sentence = [word_to_index[SB]]

        while  not new_sentence[-1] == word_to_index[SE]:
            next_word_probs,s = self.forward_propagation(new_sentence)
            sample_word = word_to_index[UNK]

            while sample_word == word_to_index[UNK]:
                #print next_word_probs[-1]
                samples = np.random.multinomial(1,next_word_probs[-1])
                print "multinomial: "
                print samples
                sample_word = np.argmax(samples)
            new_sentence.append(sample_word)
        sentence_str = [index_to_word[x] for x in new_sentence[1:-1]]
        return sentence_str








#RnnNumpy.forward_propagation = forward_propagation

np.random.seed(10)
model = RnnNumpy(voc_size)
o,s = model.forward_propagation(X_train[10])
print X_train[10]
print o.shape
print o
"""
predictions = model.predict(X_train[10])
print predictions.shape
print predictions
"""
print "Expected: %f" % np.log(voc_size)
print "Actual loss: %f" % model.calc_loss(X_train[:1000],Y_train[:1000])

"""
grad_check_vocab_size = 100
np.random.seed(10)
model = RnnNumpy(grad_check_vocab_size,10,bptt_trunacate=1000)
model.gradient_check([0,1,2,3],[1,2,3,4])
"""


np.random.seed(10)
model = RnnNumpy(voc_size)
model.train_with_sgd(X_train[:100],Y_train[:100],nepoch=10, evaluate_loss_after=1)

num_sentences = 10
sentence_min_length = 7

for i in range(num_sentences):
    sent = []

    while len(sent) < sentence_min_length:
        sent = model.generate_sentence()
    print " ".join(sent)







