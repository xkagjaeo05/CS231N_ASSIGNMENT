import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  
  num_train = X.shape[0]
  num_classes = W.shape[1]
  Z = np.zeros((num_train, num_classes))
  for i in range(num_train):
    Z[i,:] = np.dot(X[i,:],W)
    Z[i,:] = np.exp(Z[i,:])
    
    correct_class_score = Z[i,:][y[i]]
    loss += -np.log(correct_class_score/np.sum(Z[i,:]) )
    
    for k in range(num_classes):
        specific_scores= Z[i,:][k]             ## dZ equals with comparative score for that class of total score.
                                               ## And we need to subtract 1 if it's our target score.
        if k == y[i]:
            Q = (specific_scores/np.sum(Z[i,:]) )- 1
        else:
            Q = (specific_scores/np.sum(Z[i,:]) )

        dW[:,k] += np.dot(X[i,:], Q) 
  loss = (loss / num_train) + (1/2) *reg * np.sum(W*W)
  dW = (dW / num_train )+ reg*W
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  
  num_train = X.shape[0]
  Z = np.dot(X, W)
  Z2 = np.exp(Z)
  total_score = np.sum(Z2, axis=1)
  correct_class_score = Z2[range(X.shape[0]), y]
  
  loss_origin = -np.log( (correct_class_score/total_score).reshape(-1,1) )
  loss = np.sum(loss_origin)
  loss = (loss / num_train) + (1/2) * reg * np.sum(W*W)
    
    
  specific_score = Z2/total_score.reshape(-1,1)
  specific_score[range(X.shape[0]), y] -= 1
  
  dW = np.dot(X.T, specific_score)
  dW = (dW / num_train) + reg * W
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

