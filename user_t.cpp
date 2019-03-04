#include "user_t.h"

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
* This file contains helper functions that mostly get, set or print values.
*  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

user_t::user_t()
{
    pose_data = MatrixXd::Zero(1,1);
    user_limb_length = MatrixXd::Zero(1,1);
    init = false;
}

user_t::user_t(MatrixXd pose , MatrixXd  limb_length, bool initt){
    pose_data = pose;
    user_limb_length = limb_length;
    init = initt;
}



MatrixXd  user_t::get_pose(){return pose_data;}

MatrixXd  user_t::get_limb(){return user_limb_length;}

double    user_t::get_limb_idx(int i){return user_limb_length(0,i);}

bool      user_t::get_init(){return init;}

void  user_t::set_pose(MatrixXd pose){pose_data = pose;}

void  user_t::set_limb(MatrixXd  limb_length){user_limb_length = limb_length;}

void  user_t::set_limb_idx(int idx, double value){user_limb_length(0,idx) = value;}

void  user_t::set_init(bool initt){init = initt;}

void  user_t::print_pose_dim(){ std::cout << "Dim " << pose_data.rows() << " " << pose_data.cols() << std::endl;}

void  user_t::print_pose_ij(int i, int j){ if(i < pose_data.rows() && j < pose_data.cols()) std::cout << "elem " << pose_data(i,j) << std::endl;}

void  user_t::print_limb(){
    std::cout << "limb ";
    for( int i = 0; i < user_limb_length.cols(); i ++ )
        std::cout << user_limb_length(0,i) << " ";
    std::cout << " " << std::endl;}

void  user_t::save_pose_data(string file_name)
{
    ofstream myfile (file_name.c_str());
    if (!myfile.is_open())
    {
        std::cout << "Creating file " << file_name << std::endl;
        myfile.open (file_name.c_str());
    }
    myfile <<  pose_data.cols();
    myfile << " ";
    for(int i_row = 0; i_row < pose_data.rows(); i_row ++)
        for(int i_col = 0; i_col < pose_data.cols(); i_col ++){
            myfile << pose_data(i_row, i_col);
            myfile << " ";
        }
    myfile.close();
    std::cout << "Saved pose data to file " << file_name << std::endl;

}

void  user_t::save_limb_data(string file_name)
{
    ofstream myfile (file_name.c_str());
    if (!myfile.is_open())
    {
        std::cout << "Creating file " << file_name << std::endl;
        myfile.open (file_name.c_str());
    }
    myfile <<  user_limb_length.cols();
    myfile << " ";
    for(int i_row = 0; i_row < user_limb_length.cols(); i_row ++)
    {
            myfile << user_limb_length(0, i_row);
            myfile << " ";
        }
    myfile.close();
    std::cout << "Saved limb data to file " << file_name << std::endl;

}

