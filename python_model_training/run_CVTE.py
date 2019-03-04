import numpy as np
import time
import os
from CVTE import CVTE,write_params_to_txt
import cPickle
 

import theano.tensor as T
np.random.seed(42)


# lazy parameters
hu_encoder      = [  1500 ,500  ]
hu_transitioner = [   100]
hu_decoder      = [  500, 1500 ]
n_latent        = 50
continuous      = True
n_epochs        = 100
loading 	= True
model_path 	= "./models/"
data_path	= ""


# load data
f = open(data_path + 'data.pkl', 'rb')       
data = cPickle.load(f)      
x_train = data['x_train']
y_train = data['y_train']
x_valid = data['x_valid']
y_valid = data['y_valid']
x_test  = data['x_test']
y_test  = data['y_test']
f.close()
for f in [2,3,4]:
   f = open(data_path + 'data' + str(f) + '.pkl', 'rb')       
   data = cPickle.load(f)
   x_train = np.concatenate((x_train,data['x_train']))
   y_train = np.concatenate((y_train,data['y_train']))
   x_valid = np.concatenate((x_valid,data['x_valid']))
   y_valid = np.concatenate((y_valid,data['y_valid']))
   x_test  = np.concatenate((x_test,data['x_test']))
   y_test  = np.concatenate((y_test,data['y_test']))
   f.close()
  
   
   
# construct name of model 
filename =  "_vt_"  
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


def shuffle(x , y ):
    Tx = np.arange(x.shape[0])
    np.random.shuffle(Tx)
    return x[Tx], y[Tx]
     
    


 
print "instantiating model"
for iteration in range(40):
    print "Iteration " + str(iteration)

    x_train, y_train = shuffle(x_train, y_train)
    model = CVTE(continuous, hu_encoder, hu_transitioner,  hu_decoder, n_latent, x_train, y_train, batch_size=100, multi = 0.01, learning_rate=0.0001, beta = 0.8)
     
    batch_order = np.arange(int(model.N / model.batch_size))
    epoch = 0
    LB_list = []
    
    if os.path.isfile(model_path + "params" + filename) and loading :
        print "Restarting from earlier saved parameters!"
        model.load_parameters(model_path, filename)
        LB_list = np.load(model_path + "LB_list.npy")
         
         
     
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
    np.save(model_path + "LB_list.npy", LB_list)
    model.save_parameters(model_path, filename)
    model.save_parameters(model_path, str(iteration) + filename)
     

valid_LB = model.likelihood(np.float32(x_valid), np.float32(y_valid))
print "LB on validation set: {0}".format(valid_LB)

 

