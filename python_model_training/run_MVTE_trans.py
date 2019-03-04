import numpy as np
import time
import os
#from VTEmultilayer import VTE,write_params_to_txt

import cPickle
 
from MVTE import MVTE,write_params_to_txt
import theano.tensor as T

np.random.seed(42)

hu_encoder = [  200   ]
hu_transitioner = [   50]
hu_decoder = [   200 ]
n_latent = 20
continuous = True
n_epochs = 500

if False:
    print "Loading Freyface data"
    # Retrieved from: http://deeplearning.net/data/mnist/mnist.pkl.gz
    f = open('freyfaces.pkl', 'rb')
    x = cPickle.load(f)
    f.close()
    x_train = x[:1500]
    x_valid = x[1500:]
else:
   # print "Loading MNIST data"
    # Retrieved from: http://deeplearning.net/data/mnist/mnist.pkl.gz
  #  f = gzip.open('mnist.pkl.gz', 'rb')
   # (x_train, t_train), (x_valid, t_valid), (x_test, t_test) = cPickle.load(f)
   f = open('/home/jubu/Python/data_trans.pkl', 'rb')       
   data = cPickle.load(f)
   x_train = data['x_train']
   y_train = data['y_train']
   x_valid = data['x_valid']
   y_valid = data['y_valid']
   x_test  = data['x_test']
   y_test  = data['y_test']
   f.close()
   f = open('/home/jubu/Python/data2_trans.pkl', 'rb')       
   data = cPickle.load(f)
   x_train = np.concatenate((x_train,data['x_train']))
   y_train = np.concatenate((y_train,data['y_train']))
   x_valid = np.concatenate((x_valid,data['x_valid']))
   y_valid = np.concatenate((y_valid,data['y_valid']))
   x_test  = np.concatenate((x_test,data['x_test']))
   y_test  = np.concatenate((y_test,data['y_test']))
   f.close()
   f = open('/home/jubu/Python/data_trans3.pkl', 'rb')       
   data = cPickle.load(f)
   x_train = np.concatenate((x_train,data['x_train']))
   y_train = np.concatenate((y_train,data['y_train']))
   x_valid = np.concatenate((x_valid,data['x_valid']))
   y_valid = np.concatenate((y_valid,data['y_valid']))
   x_test  = np.concatenate((x_test,data['x_test']))
   y_test  = np.concatenate((y_test,data['y_test']))
 
   f.close()
filename =  "_mt_"  
for l in hu_encoder:    
    filename = filename + str(l) + "_"   
filename = filename + str(n_latent) + "_"
for l in hu_transitioner:    
    filename = filename + str(l) + "_"
filename = filename + str(n_latent) + "_"
for l in hu_decoder:    
    filename = filename + str(l) + "_"
if continuous:
    filename = filename + "c.pkl"
else:
    filename = filename + "d.pkl"
    
    

path = "./models/"
mmm = []
print "instantiating model"
for xxx in [ 0.1 ]:
    model = MVTE(continuous, hu_encoder, hu_transitioner,  hu_decoder, n_latent, x_train, y_train, multi = xxx, learning_rate=0.0001, beta = 0.8)
    
     
    batch_order = np.arange(int(model.N / model.batch_size))
    epoch = 0
    LB_list = []
    
    
    loading = False
    if os.path.isfile(path + "params" + filename) and True :
        print "Restarting from earlier saved parameters!"
        model.load_parameters(path, filename)
        LB_list = np.load(path + "LB_list.npy")
        #epoch = len(LB_list)
        #write_params_to_txt('params.txt', model)
   
    print "iterating"
    old_LB = 10000.
    LB     = 0.
    while epoch < n_epochs and (old_LB - LB)*(old_LB - LB) > 0.0000001:
        batch_order = np.arange(int(model.N / model.batch_size))
        old_LB = LB
        epoch += 1
        start = time.time()
        np.random.shuffle(batch_order)
        LB = 0.
    
        for batch in batch_order:
            batch_LB = model.update(batch, epoch)
            LB += batch_LB
    
        LB /= len(batch_order)
        
        LB_list = np.append(LB_list, LB)
        print "Epoch {0} finished. LB: {1}, time: {2}".format(epoch, LB, time.time() - start)
        np.save(path + "LB_list.npy", LB_list)
        model.save_parameters(path, filename)
    mmm.append(model)

valid_LB = model.likelihood(np.float32(x_valid), np.float32(y_valid))
print "LB on validation set: {0}".format(valid_LB)

 

