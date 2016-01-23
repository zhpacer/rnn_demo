import sys
import numpy as np
import pdb
import re
from process_data import *





class RnnNumpy:
    def __init__(self, word_dim, hidden_dim=100, bptt_trunacate=4):
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_trunacate = bptt_trunacate

        #U is hidden_dim * word_dim,  weight is uniform distribution over [-sqrt(1/word_dim),sqrt(1/word_dim)]
        self.U = np.random.uniform(-np.sqrt(1./word_dim),np.sqrt(1./word_dim),(hidden_dim,word_dim))

        #V is word_dim * hidden_dim
        self.V = np.random.uniform(-np.sqrt(1./hidden_dim),np.sqrt(1./hidden_dim),(word_dim,hidden_dim))
        #W is hidden_dim * hidden_dim
        self.W = np.random.uniform(-np.sqrt(1./hidden_dim),np.sqrt(1./hidden_dim),(hidden_dim,hidden_dim))

    def softmax(self,x):
        #e_x = np.exp(x - np.max(x))
        e_x = np.exp(x)
        out = e_x / e_x.sum()
        return out

    def forward_propagation(self,x):
        T = len(x)

        s = np.zeros((T+1,self.hidden_dim))
        s[-1] = np.zeros(self.hidden_dim)

        o = np.zeros((T,self.word_dim))

        for t in np.arange(T):
            s[t] = np.tanh(self.U[:,x[t]]+self.W.dot(s[t-1]))
            o[t] = self.softmax(self.V.dot(s[t]))

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


    def generate_sentence(self,t_data):
        word_to_index = t_data.get_word_to_index()
        index_to_word = t_data.get_index_to_word()
        SB = t_data.sb
        SE = t_data.se
        UNK = t_data.unk_token
        new_sentence = [word_to_index[SB]]

        while not new_sentence[-1] == word_to_index[SE]:
            next_word_probs,s = self.forward_propagation(new_sentence)
            sample_word = word_to_index[UNK]

            while sample_word == word_to_index[UNK]:
                #print next_word_probs[-1]
                samples = np.random.multinomial(1,next_word_probs[-1])
                #print "multinomial: "
                #print samples
                sample_word = np.argmax(samples)
            new_sentence.append(sample_word)
        sentence_str = [index_to_word[x] for x in new_sentence[1:-1]]
        return sentence_str



