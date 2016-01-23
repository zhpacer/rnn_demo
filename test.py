import os

from rnn import *
from process_data import *

def test_rnn():
  #RnnNumpy.forward_propagation = forward_propagation
  t_data = RNNTokenizer("data\\reddit-comments-2015-08.csv")
  X_train, Y_train = t_data.tokenize_data()
  np.random.seed(10)




  model = RnnNumpy(t_data.voc_size)
  o,s = model.forward_propagation(X_train[10])
  print X_train[10]
  print o.shape
  print o

  print "Expected: %f" % np.log(t_data.voc_size)
  print "Actual loss: %f" % model.calc_loss(X_train[:1000],Y_train[:1000])
  np.random.seed(10)
  model = RnnNumpy(t_data.voc_size)
  model.train_with_sgd(X_train[:100],Y_train[:100],nepoch=10, evaluate_loss_after=1)


  num_sentences = 1
  sentence_min_length = 7
  for i in range(num_sentences):
    sent = []
    while len(sent) < sentence_min_length:
        sent = model.generate_sentence(t_data)
    print " ".join(sent)

"""
predictions = model.predict(X_train[10])
print predictions.shape
print predictions
"""

"""
grad_check_vocab_size = 100
np.random.seed(10)
model = RnnNumpy(grad_check_vocab_size,10,bptt_trunacate=1000)
model.gradient_check([0,1,2,3],[1,2,3,4])
"""


if __name__ == '__main__':
    test_rnn()






