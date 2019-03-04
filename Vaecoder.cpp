#include "Vaecoder.h"
#include <map>
#include <iostream>
#include <Dense>
#include <fstream>
#include <string>
#include <stdio.h>
#include <stdexcept>
#include <math.h>

#include <boost/lexical_cast.hpp>


using namespace std;
using namespace Eigen;

double tanh(double input){
    return tanh(input);
}

double setzero(double input){

    if (input > 0.0000000000001 || input < -0.0000000000001)
        return input;
    else
        return 0.0;
}

double sigmoid(double input){
        return 1 / (1 + exp(-input));
}



/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
* Vaecoder() :  feed-forward VAE structure
*  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
Vaecoder::Vaecoder(std::string file_path){
    load_parameters(file_path);
    srand (time(NULL));
}

Vaecoder::Vaecoder(){
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
* load_parameters() :  parameters need to be saved as described below
*  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
void Vaecoder::load_parameters(std::string file_path){

    /* Input file structure
    *  # number of all layers (int)
    *  # number of encoder layers (int)
    *  # number of decoder layers (int)
    *  continuous output of decoder (bool 0/1)
    *
    *  Layer structure:
    *  W_name (string)
    *  # rows of W (int)
    *  # cols of W (int)
    *  W entries (row of doubles)
    *  b_name (string)
    *  b entries (row of doubles)
    */


    double N_layers = 0;
    int cc = 0;
    int n_rows,n_cols;
    std::string name;

    std::ifstream  text_filestream(file_path.c_str());
    text_filestream >> N_layers;
    text_filestream >> L_encoder;
    text_filestream >> L_decoder;
    text_filestream >> L_transitioner;
    text_filestream >> N_hidden;
    text_filestream >> continuous;
    while (cc < N_layers*2 ) {
        cc++;
        text_filestream >> name;
        text_filestream >> n_rows;
        text_filestream >> n_cols;
        MatrixXd mat_W(n_rows, n_cols);
        for (int i = 0; i < n_rows; i++)
            for (int j = 0; j < n_cols; j++)
                text_filestream >> mat_W(i, j);
        params.insert(std::pair<std::string, MatrixXd>(name, mat_W));

    }

}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
* encode() :  encodes data point into first latent variable (mu, sigma)
*  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
mu_sigma_struct Vaecoder::encode(MatrixXd data_input){

    int layer = 0;
    if(data_input.cols() != params["W_xh" + boost::lexical_cast<std::string>(layer)].rows())
    {
       std::cout << data_input.rows()  << "X" << data_input.cols()  << " vs " <<  params["W_xh" + boost::lexical_cast<std::string>(layer)].rows() << std::endl;
        throw std::invalid_argument( "Wrong dimensions!" );

    }
    MatrixXd h_encoder = ((data_input*params["W_xh" + boost::lexical_cast<std::string>(layer)]) + params["b_xh" + boost::lexical_cast<std::string>(layer)].replicate(data_input.rows(), 1)).unaryExpr(std::ptr_fun(tanh));
    for (int layer = 1; layer < L_encoder; layer++){
        string W_name = "W_h" + boost::lexical_cast<std::string>(layer - 1) + "h" + boost::lexical_cast<std::string>(layer );
        string b_name = "b_h" + boost::lexical_cast<std::string>(layer - 1) + "h" + boost::lexical_cast<std::string>(layer);
        if(h_encoder.cols() != params[W_name].rows())
        {
            throw std::invalid_argument( "Wrong dimensions!" );
        }
        h_encoder = ((h_encoder*params[W_name]) + params[b_name].replicate(data_input.rows(), 1)).unaryExpr(std::ptr_fun(tanh));
    }
    layer = L_encoder - 1;
    string W_name = "W_h" + boost::lexical_cast<std::string>(layer) + "mu";
    string b_name = "b_h" + boost::lexical_cast<std::string>(layer) + "mu";
    MatrixXd mu = ((h_encoder*params[W_name]) + params[b_name].replicate(data_input.rows(), 1)).unaryExpr(std::ptr_fun(setzero));
    W_name = "W_h" + boost::lexical_cast<std::string>(layer) + "sigma";
    b_name = "b_h" + boost::lexical_cast<std::string>(layer) + "sigma";
    MatrixXd sigma = ((h_encoder*params[W_name]) + params[b_name].replicate(data_input.rows(), 1)).unaryExpr(std::ptr_fun(setzero));
    mu_sigma_struct output;
    output.mu = mu;
    output.sigma = sigma;
    return output;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
* transition() :   encodes encoded data point into second latent variable (mu, sigma)
*  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
mu_sigma_struct Vaecoder::transition(MatrixXd encoded_input){
    int layer = 0;
    if(encoded_input.cols() != params["W_mut" + boost::lexical_cast<std::string>(layer)].rows())
    {
        throw std::invalid_argument( "Wrong dimensions!" );

    }
    MatrixXd h_encoder = ((encoded_input*params["W_mut" + boost::lexical_cast<std::string>(layer)]) + params["b_mut" + boost::lexical_cast<std::string>(layer)].replicate(encoded_input.rows(), 1)).unaryExpr(std::ptr_fun(tanh));
    for (int layer = 1; layer < L_transitioner; layer++){
        string W_name = "W_t" + boost::lexical_cast<std::string>(layer - 1) + "t" + boost::lexical_cast<std::string>(layer);
        string b_name = "b_t" + boost::lexical_cast<std::string>(layer - 1) + "t" + boost::lexical_cast<std::string>(layer);
        h_encoder = ((h_encoder*params[W_name]) + params[b_name].replicate(encoded_input.rows(), 1)).unaryExpr(std::ptr_fun(tanh));
    }
    layer = L_transitioner - 1;
    string W_name = "W_t" + boost::lexical_cast<std::string>(layer) + "mu";
    string b_name = "b_t" + boost::lexical_cast<std::string>(layer) + "mu";
    MatrixXd mu = ((h_encoder*params[W_name]) + params[b_name].replicate(encoded_input.rows(), 1)).unaryExpr(std::ptr_fun(setzero));
    W_name = "W_t" + boost::lexical_cast<std::string>(layer) + "sigma";
    b_name = "b_t" + boost::lexical_cast<std::string>(layer) + "sigma";
    MatrixXd sigma = ((h_encoder*params[W_name]) + params[b_name].replicate(encoded_input.rows(), 1)).unaryExpr(std::ptr_fun(setzero));
    mu_sigma_struct output;
    output.mu = mu;
    output.sigma = sigma;
    return output;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
* decode() :   decodes transitioned data point into output distribution (mu, sigma)
*  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
mu_sigma_struct Vaecoder::decode(MatrixXd encoded_input){
    int layer = 0;
    MatrixXd h_encoder = ((encoded_input*params["W_mud" + boost::lexical_cast<std::string>(layer)]) + params["b_mud" + boost::lexical_cast<std::string>(layer)].replicate(encoded_input.rows(), 1)).unaryExpr(std::ptr_fun(tanh));
    for (int layer = 1; layer < L_decoder; layer++){
        string W_name = "W_d" + boost::lexical_cast<std::string>(layer - 1) + "d" + boost::lexical_cast<std::string>(layer);
        string b_name = "b_d" + boost::lexical_cast<std::string>(layer - 1) + "d" + boost::lexical_cast<std::string>(layer);
        h_encoder = ((h_encoder*params[W_name]) + params[b_name].replicate(encoded_input.rows(), 1)).unaryExpr(std::ptr_fun(tanh));
    }
    layer = L_decoder - 1;
    MatrixXd mu;
    MatrixXd sigma;
    if (continuous){
        string W_name = "W_d" + boost::lexical_cast<std::string>(layer) + "xmu";
        string b_name = "b_d" + boost::lexical_cast<std::string>(layer) + "xmu";
        mu = ((h_encoder*params[W_name]) + params[b_name].replicate(encoded_input.rows(), 1)).unaryExpr(std::ptr_fun(setzero));
        W_name = "W_d" + boost::lexical_cast<std::string>(layer) + "xsigma";
        b_name = "b_d" + boost::lexical_cast<std::string>(layer) + "xsigma";
        sigma = ((h_encoder*params[W_name]) + params[b_name].replicate(encoded_input.rows(), 1)).unaryExpr(std::ptr_fun(setzero));
    }
    else{
        string W_name = "W_d" + boost::lexical_cast<std::string>(layer) + "x";
        string b_name = "b_d" + boost::lexical_cast<std::string>(layer) + "x";
        mu = ((h_encoder*params[W_name]) + params[b_name].replicate(encoded_input.rows(), 1)).unaryExpr(std::ptr_fun(sigmoid));
        sigma = mu*0;
    }
    mu_sigma_struct output;
    output.mu = mu;
    output.sigma = sigma;
    return output;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
* e_s_t_s_d() :   encode - sample - transition - sample - decode 
*  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
mu_sigma_struct Vaecoder::e_s_t_s_d(MatrixXd data_input){
    mu_sigma_struct encoded = encode(data_input);
    MatrixXd enc_sample = sample(encoded.mu, encoded.sigma);
    mu_sigma_struct transitioned = transition(enc_sample);
    MatrixXd tra_sample = sample(transitioned.mu, transitioned.sigma);
    mu_sigma_struct decoded = decode(tra_sample);
    return decoded;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
* e_t_s_d() :   encode -  transition - sample - decode 
*  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
mu_sigma_struct Vaecoder::e_t_s_d(MatrixXd data_input){
    mu_sigma_struct encoded = encode(data_input);
    mu_sigma_struct transitioned = transition(encoded.mu);
    MatrixXd tra_sample = sample(transitioned.mu, transitioned.sigma);
    mu_sigma_struct decoded = decode(tra_sample);
    return decoded;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
* e_t_d() :   encode -  transition - decode 
*  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
mu_sigma_struct Vaecoder::e_t_d(MatrixXd data_input){
    mu_sigma_struct encoded = encode(data_input);
    mu_sigma_struct transitioned = transition(encoded.mu);
    mu_sigma_struct decoded = decode(transitioned.mu);
    return decoded;
}


/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
* sample() : sample from normal distribution (mu, sigma)
*  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
MatrixXd  Vaecoder::sample(MatrixXd mu_vec, MatrixXd log_sig_mu){
    MatrixXd sample = MatrixXd::Zero(mu_vec.rows(), mu_vec.cols());

    for(int i = 0; i < mu_vec.rows();  i++){
         for(int j = 0; j < mu_vec.cols(); j++){

            sample(i,j)= mu_vec(i,j) + exp(0.5*log_sig_mu(i,j))*sample_normal(); //Generate numbers;
         }
    }
    return sample;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
* sample_normal() : sample from normal distribution (0, 1)
*  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
double Vaecoder::sample_normal() {

    double u = ((double) rand() / (RAND_MAX)) * 2 - 1;
    double v = ((double) rand() / (RAND_MAX)) * 2 - 1;
    double r = u * u + v * v;
    if (r == 0 || r > 1) return sample_normal();
    double c = sqrt(-2 * log(r) / r);
    return u * c;
}





