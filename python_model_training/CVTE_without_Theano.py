from __future__ import division
from __future__ import print_function

import numpy as np
import time

"""
This class implements the CVTE in numpy only, no GPU involved.
See CVTE.py for details. 
"""



import cPickle
from collections import OrderedDict
 
epsilon = 1e-8



def tanh(x):
    
    return np.tanh(x )
    
def write_params_to_txt(name, model):
    params = model.params
    pkeys = params.keys()
    L_enc = model.L_encoder
    L_dec = model.L_decoder
    L_tran = model.L_transitioner
    L_lat  = model.n_latent
    cont  = model.continuous
    with open(name, 'w') as f:
        print(str(len(pkeys) / 2), file = f)
    f.close()
    with open(name, 'a') as f:
        
        print(str(L_enc) , file = f)
        print(str(L_dec) , file = f)
        print(str(L_tran) , file = f)
        print(str(L_lat) , file = f)
        print(str(cont.numerator) , file = f)
        for k in pkeys:
            print(k)
            param = np.array(params[k])
            print(param.shape)
            if len(param.shape)<2:
                param = np.reshape(param,(1,param.shape[0]))
             
            [rows,cols] = param.shape
            print(k , file = f)
            print(str(rows), file = f)
            print(str(cols) , file = f)
            for r in range(rows):
                for c in range(cols):
                    if r == 0 and c == 0:
                        numbers = str(param[r][c]) 
                    else:
                        numbers =  ' ' + str(param[r][c]) 
                    print(numbers , end="", file = f)
            
            
    f.close()
    

 


class CVTE:
    """This class implements the Conditional Variational Temporal Encoder"""
    def __init__(self, continuous, hu_encoder, hu_transition, hu_decoder, n_latent, x_train, y_train, multi = 0.0, beta = 1, b1=0.95, b2=0.999, batch_size=100, learning_rate=0.001, lam=0):
        self.continuous = continuous
        self.hu_encoder = hu_encoder
        self.hu_decoder = hu_decoder
        self.n_latent = n_latent
        self.multi = multi
        [self.N, self.features] = x_train.shape
        [self.N, self.features_y] = y_train.shape

        self.prng = np.random.RandomState(42)

        self.b1 = b1
        self.b2 = b2
        self.learning_rate = learning_rate
        self.lam = lam
        self.beta = beta
        
        self.L_encoder = len(hu_encoder)
        self.L_transitioner = len(hu_transition)
        self.L_decoder = len(hu_decoder)
        
        self.batch_size = batch_size

         
        

        self.params = OrderedDict()
        
        """--------------------------------------------------------------------
           encoder
           --------------------------------------------------------------------"""
        layer = 0
        W_name = 'W_xh' + str(layer) 
        b_name = 'b_xh' + str(layer)
        W_xh   = np.random.normal(0,0.1,(self.features, hu_encoder[layer]))
        b_xh   = np.random.normal(0,0.1,(hu_encoder[layer]))
        self.params.update({W_name : W_xh, b_name : b_xh})
        del W_xh,b_xh
        
        
        for layer in range(1,self.L_encoder):
            W_name = 'W_h' + str(layer-1) + 'h' + str(layer) 
            b_name = 'b_h' + str(layer-1) + 'h' + str(layer)
            W_xh = np.random.normal(0,0.1,(hu_encoder[layer-1], hu_encoder[layer]))
            b_xh = np.random.normal(0,0.1,(hu_encoder[layer]))
            self.params.update({W_name : W_xh, b_name : b_xh})
            del W_xh,b_xh
            
        W_name = 'W_h' + str(layer) + 'mu' 
        b_name = 'b_h' + str(layer) + 'mu'
        W_hmu = np.random.normal(0,0.1,(hu_encoder[-1], n_latent))
        b_hmu = np.random.normal(0,0.1,(n_latent))
        self.params.update({W_name: W_hmu, b_name : b_hmu})
        
        W_name = 'W_h' + str(layer) + 'sigma' 
        b_name = 'b_h' + str(layer) + 'sigma'
        W_hsigma = np.random.normal(0,0.1,(hu_encoder[-1], n_latent))
        b_hsigma = np.random.normal(0,0.1,(n_latent))
        self.params.update({W_name: W_hsigma, b_name : b_hsigma})
        """--------------------------------------------------------------------
           transition
           --------------------------------------------------------------------"""         
        layer = 0
        W_name = 'W_mut' + str(layer) 
        b_name = 'b_mut' + str(layer)
        W_xh   = np.random.normal(0,0.1,(n_latent, hu_transition[layer]))
        b_xh   = np.random.normal(0,0.1,(hu_transition[layer]))
        self.params.update({W_name: W_xh, b_name: b_xh})
        del W_xh,b_xh
        
        for layer in range(1,self.L_transitioner):
            W_name = 'W_t' + str(layer-1) + 't' + str(layer) 
            b_name = 'b_t' + str(layer-1) + 't' + str(layer)
            W_xh = np.random.normal(0,0.1,(hu_transition[layer-1], hu_transition[layer]))
            b_xh = np.random.normal(0,0.1,(hu_transition[layer]))
            self.params.update({W_name : W_xh, b_name : b_xh})
            del W_xh,b_xh
            
        W_name = 'W_t' + str(layer) + 'mu' 
        b_name = 'b_t' + str(layer) + 'mu'
        W_hmu = np.random.normal(0,0.1,(hu_transition[-1], n_latent))
        b_hmu = np.random.normal(0,0.1,(n_latent))
        self.params.update({W_name: W_hmu, b_name : b_hmu})
        
        W_name = 'W_t' + str(layer) + 'sigma' 
        b_name = 'b_t' + str(layer) + 'sigma'
        W_hsigma = np.random.normal(0,0.1,(hu_transition[-1], n_latent))
        b_hsigma = np.random.normal(0,0.1,())
        self.params.update({W_name: W_hsigma, b_name : b_hsigma})
           
        
        """--------------------------------------------------------------------
           decoder
           --------------------------------------------------------------------"""
        layer = 0
        W_name = 'W_mud' + str(layer) 
        b_name = 'b_mud' + str(layer)
        W_xh   = np.random.normal(0,0.1,(n_latent, hu_decoder[layer]))
        b_xh   = np.random.normal(0,0.1,(hu_decoder[layer]))
        self.params.update({W_name: W_xh, b_name: b_xh})
        del W_xh,b_xh
        
        for layer in range(1,self.L_decoder):
            W_name = 'W_d' + str(layer-1) + 'd' + str(layer) 
            b_name = 'b_d' + str(layer-1) + 'd' + str(layer)
            W_xh = np.random.normal(0,0.1,(hu_decoder[layer-1], hu_decoder[layer]))
            b_xh = np.random.normal(0,0.1,(hu_decoder[layer]))
            self.params.update({W_name : W_xh, b_name : b_xh})
            del W_xh,b_xh


        if self.continuous:
            W_name = 'W_d' + str(layer) + 'xmu' 
            b_name = 'b_d' + str(layer) + 'xmu'
            W_hxmu = np.random.normal(0,0.1,(hu_decoder[-1], self.features_y))
            b_hxmu = np.random.normal(0,0.1,(self.features_y))
            self.params.update({W_name : W_hxmu, b_name : b_hxmu})
            
            W_name = 'W_d' + str(layer) + 'xsigma' 
            b_name = 'b_d' + str(layer) + 'xsigma'
            W_hxsig = np.random.normal(0,0.1,(hu_decoder[-1], self.features_y))
            b_hxsig = np.random.normal(0,0.1,(self.features_y))
            self.params.update({W_name : W_hxsig, b_name : b_hxsig})
        else:
            W_name = 'W_d' + str(layer) + 'x' 
            b_name = 'b_d' + str(layer) + 'x'
            W_hx = np.random.normal(0,0.1,(hu_decoder[-1], self.features_y))
            b_hx = np.random.normal(0,0.1,(self.features_y))

            self.params.update({W_name: W_hx, b_name: b_hx})

      
        print('Model initialized')
        
        


    def encoder(self, x):
        layer = 0
        h_encoder = tanh(np.dot(x, self.params['W_xh'+str(layer)]) + self.params['b_xh'+str(layer)])
        for layer in range(1,self.L_encoder):
            W_name = 'W_h' + str(layer-1) + 'h' + str(layer) 
            b_name = 'b_h' + str(layer-1) + 'h' + str(layer)
            h_encoder = tanh(np.dot(h_encoder, self.params[W_name]) + self.params[b_name])
            
        mu = np.dot(h_encoder, self.params['W_h' + str(layer) + 'mu']) + self.params['b_h' + str(layer) + 'mu']
        log_sigma = np.dot(h_encoder, self.params['W_h' + str(layer) + 'sigma']) + self.params['b_h' + str(layer) + 'sigma']
        return mu, log_sigma
        
    def transitioner(self, z):
        layer = 0
        h_transition = tanh(np.dot(z, self.params['W_mut'+str(layer)]) + self.params['b_mut'+str(layer)])
        for layer in range(1,self.L_transitioner):
            W_name = 'W_t' + str(layer-1) + 't' + str(layer) 
            b_name = 'b_t' + str(layer-1) + 't' + str(layer)
            h_transition = tanh(np.dot(h_transition, self.params[W_name]) + self.params[b_name])
            
        mu = np.dot(h_transition, self.params['W_t' + str(layer) + 'mu']) + self.params['b_t' + str(layer) + 'mu']
        log_sigma = np.dot(h_transition, self.params['W_t' + str(layer) + 'sigma']) + self.params['b_t' + str(layer) + 'sigma']
        return mu, log_sigma
    



    def decoder(self, x, z, y = -1):
        layer = 0
        h_decoder = tanh(np.dot(z, self.params['W_mud'+str(layer)]) + self.params['b_mud'+str(layer)])
        for layer in range(1,self.L_decoder):
            W_name = 'W_d' + str(layer-1) + 'd' + str(layer) 
            b_name = 'b_d' + str(layer-1) + 'd' + str(layer)
            h_decoder = tanh(np.dot(h_decoder, self.params[W_name]) + self.params[b_name])

        if self.continuous:
            reconstructed_y = np.dot(h_decoder, self.params['W_d' + str(layer) + 'xmu']) + self.params['b_d' + str(layer) + 'xmu' ]
            log_sigma_decoder = np.dot(h_decoder, self.params['W_d' + str(layer) + 'xsigma' ]) + self.params['b_d' + str(layer) + 'xsigma' ]
            if y == -1:
                logpyz = (-(0.5 * np.log(2 * np.pi) + 0.5 * log_sigma_decoder) -
                          0.5 * ((x - reconstructed_y)**2 / np.exp(log_sigma_decoder))).sum(axis=1)
            else: 
                logpyz = (-(0.5 * np.log(2 * np.pi) + 0.5 * log_sigma_decoder) -
                          0.5 * ((y - reconstructed_y)**2 / np.exp(log_sigma_decoder))).sum(axis=1)
        else:
            reconstructed_y = np.sigmoid(np.dot(h_decoder, self.params['W_d' + str(layer) + 'x']) + self.params['b_d' + str(layer) + 'x' ])
            if y == -1:
                logpyz = - 0
            else:
                logpyz = - 0

        return reconstructed_y, logpyz
        
    def sampler(self, mu, log_sigma):  
        
        eps = np.random.normal(0,5, (mu.shape))
        # Reparametrize
        z = mu + np.exp(0.5 * log_sigma) * eps
        return z   
    
    
        
    def decoder_mu_sig(self,z):
        
        layer = 0
        
        h_decoder = tanh(np.dot(z, self.params['W_mud'+str(layer)]) + self.params['b_mud'+str(layer)])
        for layer in range(1,self.L_decoder):
            W_name = 'W_d' + str(layer-1) + 'd' + str(layer) 
            b_name = 'b_d' + str(layer-1) + 'd' + str(layer)
            h_decoder = tanh(np.dot(h_decoder, self.params[W_name]) + self.params[b_name])

        if self.continuous:
            reconstructed_y = np.dot(h_decoder, self.params['W_d' + str(layer) + 'xmu']) + self.params['b_d' + str(layer) + 'xmu' ]
            log_sigma_decoder = np.dot(h_decoder, self.params['W_d' + str(layer) + 'xsigma' ]) + self.params['b_d' + str(layer) + 'xsigma' ]
            log_sigma_decoder = np.exp(log_sigma_decoder)
        else:
            reconstructed_y = np.sigmoid(np.dot(h_decoder, self.params['W_d' + str(layer) + 'x']) + self.params['b_d' + str(layer) + 'x' ])
            log_sigma_decoder = 0

        return reconstructed_y, log_sigma_decoder


    

    def save_parameters(self, path, filename):
        """Saves all the parameters in a way they can be retrieved later"""
        cPickle.dump({name: p  for name, p in self.params.items()}, open(path + "/params" + filename, "wb"))
        

    def load_parameters(self, path, filename):
        """Load the variables in a shared variable safe way"""
        p_list = cPickle.load(open(path + "params" + filename, "rb"))
        for name in p_list.keys():
            self.params[name] = (p_list[name])
            
