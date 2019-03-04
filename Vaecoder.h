#include <map>
#include <iostream>
#include <Dense>
#include <iostream>
#include <fstream>
#include <string>
#include <stdio.h>
#include <math.h>
#include <boost/lexical_cast.hpp>


using namespace Eigen;


struct mu_sigma_struct{
        MatrixXd mu;
        MatrixXd sigma;
};

class Vaecoder
{
public:
        Vaecoder(std::string file_path);
        Vaecoder();
        void load_parameters(std::string file_path);
        mu_sigma_struct decode(MatrixXd data_input);
        mu_sigma_struct transition(MatrixXd encoded_input);
        mu_sigma_struct encode(MatrixXd encoded_input);
        MatrixXd        sample(MatrixXd mu_vec, MatrixXd log_sig_mu);
        mu_sigma_struct e_s_t_s_d(MatrixXd data_input); // samples twice
        mu_sigma_struct e_t_s_d(MatrixXd data_input); // samples only last
        mu_sigma_struct e_t_d(MatrixXd data_input); // does pure prediction with mu
        std::map<std::string, MatrixXd> params;
        double sample_normal();

private:
        int L_encoder;
        int L_decoder;
        int L_transitioner;
        int N_hidden;
        bool continuous;





};


