from __future__ import division
from __future__ import print_function
import numpy as np
import theano
import theano.tensor as T
import theano.sandbox.rng_mrg as RNG_MRG
import cPickle
from collections import OrderedDict
 
epsilon = 1e-8


theano_random = RNG_MRG.MRG_RandomStreams(seed=23455)

def tanh(x):
    
    return T.tanh(x )
    
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
            param = np.array(params[k].eval())
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
    def __init__(self, continuous, hu_encoder, hu_transition, hu_decoder, n_latent, 
                 x_train, y_train, multi = 0.0, beta = 1, b1=0.95, b2=0.999, batch_size=100, learning_rate=0.001, lam=0, evaluations = 1):
        
        self.continuous = continuous
        self.hu_encoder = hu_encoder
        self.hu_decoder = hu_decoder
        self.n_latent   = n_latent
        self.multi      = multi
        self.evaluations= evaluations
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

        sigma_init = 0.01
        
        if "gpu" in theano.config.device:
            self.srng = theano.sandbox.cuda.rng_curand.CURAND_RandomStreams(seed = np.random.randint(100000000) )
        else:
            self.srng = T.shared_randomstreams.RandomStreams(seed = np.random.randint(100000000))

        create_weight = lambda dim_input, dim_output: self.prng.normal(0, sigma_init, (dim_input, dim_output)).astype(theano.config.floatX)
        create_bias   = lambda dim_output: np.zeros(dim_output).astype(theano.config.floatX)
        
        self.params = OrderedDict()
        
        """--------------------------------------------------------------------
           encoder
           --------------------------------------------------------------------"""
        layer = 0
        W_name = 'W_xh' + str(layer) 
        b_name = 'b_xh' + str(layer)
        W_xh   = theano.shared(create_weight(self.features, hu_encoder[layer]), name=W_name)
        b_xh   = theano.shared(create_bias(hu_encoder[layer]), name=b_name)
        self.params.update({W_name : W_xh, b_name : b_xh})
        del W_xh,b_xh
        
        for layer in range(1,self.L_encoder):
            W_name = 'W_h' + str(layer-1) + 'h' + str(layer) 
            b_name = 'b_h' + str(layer-1) + 'h' + str(layer)
            W_xh = theano.shared(create_weight(hu_encoder[layer-1], hu_encoder[layer]), name=W_name)
            b_xh = theano.shared(create_bias(hu_encoder[layer]), name=b_name)
            self.params.update({W_name : W_xh, b_name : b_xh})
            del W_xh,b_xh
            
        W_name = 'W_h' + str(layer) + 'mu' 
        b_name = 'b_h' + str(layer) + 'mu'
        W_hmu = theano.shared(create_weight(hu_encoder[-1], n_latent), name=W_name)
        b_hmu = theano.shared(create_bias(n_latent), name=b_name)
        self.params.update({W_name: W_hmu, b_name : b_hmu})
        
        W_name = 'W_h' + str(layer) + 'sigma' 
        b_name = 'b_h' + str(layer) + 'sigma'
        W_hsigma = theano.shared(create_weight(hu_encoder[-1], n_latent), name=W_name)
        b_hsigma = theano.shared(create_bias(n_latent), name=b_name)
        self.params.update({W_name: W_hsigma, b_name : b_hsigma})
        """--------------------------------------------------------------------
           transition
           --------------------------------------------------------------------"""         
        layer = 0
        W_name = 'W_mut' + str(layer) 
        b_name = 'b_mut' + str(layer)
        W_xh   = theano.shared(create_weight(n_latent, hu_transition[layer]), name=W_name)
        b_xh   = theano.shared(create_bias(hu_transition[layer]), name=b_name)
        self.params.update({W_name: W_xh, b_name: b_xh})
        del W_xh,b_xh
        
        for layer in range(1,self.L_transitioner):
            W_name = 'W_t' + str(layer-1) + 't' + str(layer) 
            b_name = 'b_t' + str(layer-1) + 't' + str(layer)
            W_xh = theano.shared(create_weight(hu_transition[layer-1], hu_transition[layer]), name=W_name)
            b_xh = theano.shared(create_bias(hu_transition[layer]), name=b_name)
            self.params.update({W_name : W_xh, b_name : b_xh})
            del W_xh,b_xh
            
        W_name = 'W_t' + str(layer) + 'mu' 
        b_name = 'b_t' + str(layer) + 'mu'
        W_hmu = theano.shared(create_weight(hu_transition[-1], n_latent), name=W_name)
        b_hmu = theano.shared(create_bias(n_latent), name=b_name)
        self.params.update({W_name: W_hmu, b_name : b_hmu})
        
        W_name = 'W_t' + str(layer) + 'sigma' 
        b_name = 'b_t' + str(layer) + 'sigma'
        W_hsigma = theano.shared(create_weight(hu_transition[-1], n_latent), name=W_name)
        b_hsigma = theano.shared(create_bias(n_latent), name=b_name)
        self.params.update({W_name: W_hsigma, b_name : b_hsigma})
           
        
        """--------------------------------------------------------------------
           decoder
           --------------------------------------------------------------------"""
        layer = 0
        W_name = 'W_mud' + str(layer) 
        b_name = 'b_mud' + str(layer)
        W_xh   = theano.shared(create_weight(n_latent, hu_decoder[layer]), name=W_name)
        b_xh   = theano.shared(create_bias(hu_decoder[layer]), name=b_name)
        self.params.update({W_name: W_xh, b_name: b_xh})
        del W_xh,b_xh
        
        for layer in range(1,self.L_decoder):
            W_name = 'W_d' + str(layer-1) + 'd' + str(layer) 
            b_name = 'b_d' + str(layer-1) + 'd' + str(layer)
            W_xh = theano.shared(create_weight(hu_decoder[layer-1], hu_decoder[layer]), name=W_name)
            b_xh = theano.shared(create_bias(hu_decoder[layer]), name=b_name)
            self.params.update({W_name : W_xh, b_name : b_xh})
            del W_xh,b_xh


        if self.continuous:
            W_name = 'W_d' + str(layer) + 'xmu' 
            b_name = 'b_d' + str(layer) + 'xmu'
            W_hxmu = theano.shared(create_weight(hu_decoder[-1], self.features_y), name=W_name)
            b_hxmu = theano.shared(create_bias(self.features_y), name=b_name)
            self.params.update({W_name : W_hxmu, b_name : b_hxmu})
            
            W_name = 'W_d' + str(layer) + 'xsigma' 
            b_name = 'b_d' + str(layer) + 'xsigma'
            W_hxsig = theano.shared(create_weight(hu_decoder[-1], self.features_y), name=W_name)
            b_hxsig = theano.shared(create_bias(self.features_y), name=b_name)
            self.params.update({W_name : W_hxsig, b_name : b_hxsig})
        else:
            W_name = 'W_d' + str(layer) + 'x' 
            b_name = 'b_d' + str(layer) + 'x'
            W_hx = theano.shared(create_weight(hu_decoder[-1], self.features_y), name=W_name)
            b_hx = theano.shared(create_bias(self.features_y), name=b_name)

            self.params.update({W_name: W_hx, b_name: b_hx})

        # Adam parameters
        self.m = OrderedDict()
        self.v = OrderedDict()

        for key, value in self.params.items():
                
                self.m[key] = theano.shared(np.zeros_like(value.get_value()).astype(theano.config.floatX), name='m_' + key)
                self.v[key] = theano.shared(np.zeros_like(value.get_value()).astype(theano.config.floatX), name='v_' + key)

        x_train = theano.shared(x_train.astype(theano.config.floatX), name="x_train" )
        y_train = theano.shared(y_train.astype(theano.config.floatX), name="y_train" )
        print('Model initialized')
        self.update, self.likelihood, self.encode, self.decode = self.create_gradientfunctions(x_train, y_train)
        


    def encoder(self, x):
	"""encoder x -> z1"""
        layer = 0
        h_encoder = tanh(T.dot(x, self.params['W_xh'+str(layer)]) + self.params['b_xh'+str(layer)].dimshuffle('x', 0))
        for layer in range(1,self.L_encoder):
            W_name = 'W_h' + str(layer-1) + 'h' + str(layer) 
            b_name = 'b_h' + str(layer-1) + 'h' + str(layer)
            h_encoder = tanh(T.dot(h_encoder, self.params[W_name]) + self.params[b_name].dimshuffle('x', 0))
            
        mu = T.dot(h_encoder, self.params['W_h' + str(layer) + 'mu']) + self.params['b_h' + str(layer) + 'mu'].dimshuffle('x', 0)
        log_sigma = T.dot(h_encoder, self.params['W_h' + str(layer) + 'sigma']) + self.params['b_h' + str(layer) + 'sigma'].dimshuffle('x', 0)
        return mu, log_sigma
        
    def transitioner(self, z):
	"""transitioner z1 -> z2"""
        layer = 0
        h_transition = tanh(T.dot(z, self.params['W_mut'+str(layer)]) + self.params['b_mut'+str(layer)].dimshuffle('x', 0))
        for layer in range(1,self.L_transitioner):
            W_name = 'W_t' + str(layer-1) + 't' + str(layer) 
            b_name = 'b_t' + str(layer-1) + 't' + str(layer)
            h_transition = tanh(T.dot(h_transition, self.params[W_name]) + self.params[b_name].dimshuffle('x', 0))
            
        mu = T.dot(h_transition, self.params['W_t' + str(layer) + 'mu']) + self.params['b_t' + str(layer) + 'mu'].dimshuffle('x', 0)
        log_sigma = T.dot(h_transition, self.params['W_t' + str(layer) + 'sigma']) + self.params['b_t' + str(layer) + 'sigma'].dimshuffle('x', 0)
        return mu, log_sigma
    

    
    def decoder(self, x, z, y = -1):
	"""decoder z2 -> y"""
        layer = 0
        h_decoder = tanh(T.dot(z, self.params['W_mud'+str(layer)]) + self.params['b_mud'+str(layer)].dimshuffle('x', 0))
        for layer in range(1,self.L_decoder):
            W_name = 'W_d' + str(layer-1) + 'd' + str(layer) 
            b_name = 'b_d' + str(layer-1) + 'd' + str(layer)
            h_decoder = tanh(T.dot(h_decoder, self.params[W_name]) + self.params[b_name].dimshuffle('x', 0))

        if self.continuous:
            reconstructed_y = T.dot(h_decoder, self.params['W_d' + str(layer) + 'xmu']) + self.params['b_d' + str(layer) + 'xmu' ].dimshuffle('x', 0)
            log_sigma_decoder = T.dot(h_decoder, self.params['W_d' + str(layer) + 'xsigma' ]) + self.params['b_d' + str(layer) + 'xsigma' ]
            if y == -1:
                logpyz = (-(0.5 * np.log(2 * np.pi) + 0.5 * log_sigma_decoder) -
                          0.5 * ((x - reconstructed_y)**2 / T.exp(log_sigma_decoder))).sum(axis=1)
            else: 
                logpyz = (-(0.5 * np.log(2 * np.pi) + 0.5 * log_sigma_decoder) -
                          0.5 * ((y - reconstructed_y)**2 / T.exp(log_sigma_decoder))).sum(axis=1)
        else:
            reconstructed_y = T.nnet.sigmoid(T.dot(h_decoder, self.params['W_d' + str(layer) + 'x']) + self.params['b_d' + str(layer) + 'x' ].dimshuffle('x', 0))
            if y == -1:
                logpyz = - T.nnet.binary_crossentropy(reconstructed_y, x).sum(axis=1)
            else:
                logpyz = - T.nnet.binary_crossentropy(reconstructed_y, y).sum(axis=1)

        return reconstructed_y, logpyz
        
    def sampler(self, mu, log_sigma):  
	"""sampling from normal"""
        eps = self.srng.normal(mu.shape)
        # Reparametrize
        z = mu + T.exp(0.5 * log_sigma) * eps
        return z

    
    
        
    def decoder_mu_sig(self,z):
        """decoder returns mean and variance"""
        layer = 0
        h_decoder = tanh(T.dot(z, self.params['W_mud'+str(layer)]) + self.params['b_mud'+str(layer)].dimshuffle('x', 0))
        for layer in range(1,self.L_decoder):
            W_name = 'W_d' + str(layer-1) + 'd' + str(layer) 
            b_name = 'b_d' + str(layer-1) + 'd' + str(layer)
            h_decoder = tanh(T.dot(h_decoder, self.params[W_name]) + self.params[b_name].dimshuffle('x', 0))

        if self.continuous:
            reconstructed_y = T.dot(h_decoder, self.params['W_d' + str(layer) + 'xmu']) + self.params['b_d' + str(layer) + 'xmu' ].dimshuffle('x', 0)
            log_sigma_decoder = T.dot(h_decoder, self.params['W_d' + str(layer) + 'xsigma' ]) + self.params['b_d' + str(layer) + 'xsigma' ]
            log_sigma_decoder = log_sigma_decoder.exp()
        else:
            reconstructed_y = T.nnet.sigmoid(T.dot(h_decoder, self.params['W_d' + str(layer) + 'x']) + self.params['b_d' + str(layer) + 'x' ].dimshuffle('x', 0))
            log_sigma_decoder = 0

        return reconstructed_y, log_sigma_decoder


    def create_gradientfunctions(self, x_train, y_train):
        """compute loss and create gradient""" 
        x = T.matrix("x")
        y = T.matrix("y")

        epoch = T.scalar("epoch")
         
        noise = theano_random.normal(avg=0, std=0.1, size=x.shape, dtype=theano.config.floatX)
        
        for idx in range(self.evaluations):
            mu_zt_1, log_sigmazt_1 = self.encoder(x + self.multi*noise)
            zt_1 = self.sampler(mu_zt_1, log_sigmazt_1)
            mu_zt, log_sigmazt = self.transitioner(zt_1)
            zt = self.sampler(mu_zt, log_sigmazt)
            reconstructed_y, logpyz = self.decoder(x,zt,y)
            
            # Expectation of (logpz - logqz_x) over logqz_x is equal to KLD (see appendix B):
            KLDzt_1 = 0.5 * T.sum(1 + log_sigmazt_1 - mu_zt_1**2 - T.exp(log_sigmazt_1), axis=1)
            KLDzt   = 0.5 * T.sum(1 + log_sigmazt   - mu_zt**2   - T.exp(log_sigmazt),   axis=1)
            
            if idx == 0:
            # Average over batch dimension
                logpx = T.mean(logpyz +  self.beta*(KLDzt + KLDzt_1))    
            else:
                logpx += T.mean(logpyz +  self.beta*(KLDzt + KLDzt_1))    

        logpx = logpx / np.float(self.evaluations)
        # Compute all the gradients
        gradients = T.grad(logpx , self.params.values())

        # Adam implemented as updates
        updates = self.get_adam_updates(gradients, epoch)

        batch = T.iscalar('batch')

        givens = {
            x: x_train[batch*self.batch_size:(batch+1)*self.batch_size, :],
            y: y_train[batch*self.batch_size:(batch+1)*self.batch_size, :]
        }

        # Define a bunch of functions for convenience
        update = theano.function([batch, epoch], logpx, updates=updates, givens=givens)
        likelihood = theano.function([x,y], logpx)
        encode     = theano.function([x], zt_1)
        
        decode     = theano.function([zt], reconstructed_y)

        return update, likelihood, encode, decode 

    def transform_data(self, x_train):
	"""transform x to z"""
        transformed_x = np.zeros((self.N, self.n_latent))
        batches       = np.arange(int(self.N / self.batch_size))

        for batch in batches:
            batch_x = x_train[batch*self.batch_size:(batch+1)*self.batch_size, :]
            transformed_x[batch*self.batch_size:(batch+1)*self.batch_size, :] = self.encode(batch_x)

        return transformed_x

    def save_parameters(self, path, filename):
        """Saves all the parameters in a way they can be retrieved later"""
        cPickle.dump({name: p.get_value() for name, p in self.params.items()}, open(path + "/params" + filename, "wb"))
        cPickle.dump({name: m.get_value() for name, m in self.m.items()}, open(path + "/m" + filename, "wb"))
        cPickle.dump({name: v.get_value() for name, v in self.v.items()}, open(path + "/v" + filename, "wb"))

    def load_parameters(self, path, filename):
        """Load the variables in a shared variable safe way"""
        p_list = cPickle.load(open(path + "params" + filename, "rb"))
        m_list = cPickle.load(open(path + "m" + filename, "rb"))
        v_list = cPickle.load(open(path + "v" + filename, "rb"))
        
        

        for name in p_list.keys():
            self.params[name].set_value(p_list[name].astype(theano.config.floatX))
            self.m[name].set_value(m_list[name].astype(theano.config.floatX))
            self.v[name].set_value(v_list[name].astype(theano.config.floatX))

    def get_adam_updates(self, gradients, epoch):
	"""Adam optimizer"""
        updates = OrderedDict()
        gamma = T.sqrt(1 - self.b2**epoch) / (1 - self.b1**epoch)

        values_iterable = zip(self.params.keys(), self.params.values(), gradients, 
                              self.m.values(), self.v.values())

        for name, parameter, gradient, m, v in values_iterable:
            new_m = self.b1 * m + (1. - self.b1) * gradient
            new_v = self.b2 * v + (1. - self.b2) * (gradient**2)

            updates[parameter] = parameter + self.learning_rate * gamma * new_m / (T.sqrt(new_v) + epsilon)

            if 'W' in name:
                # MAP on weights (same as L2 regularization)
                updates[parameter] -= self.learning_rate * self.lam * (parameter * np.float32(self.batch_size / self.N))

            updates[m] = new_m
            updates[v] = new_v

        return updates
