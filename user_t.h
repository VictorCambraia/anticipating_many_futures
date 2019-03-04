#ifndef USER_T_H
#define USER_T_H

#include <XnCppWrapper.h>
#include <Dense>
#include <iostream>
#include <fstream>
using namespace std;


using Eigen::MatrixXd;

class user_t
{
public:
    user_t();
    user_t(MatrixXd pose_data, MatrixXd user_limb_length, bool init);
    Eigen::MatrixXd pose_data;
    Eigen::MatrixXd user_limb_length;
    bool init;
    MatrixXd  get_pose();
    MatrixXd  get_limb();
    double    get_limb_idx(int i);
    bool      get_init();
    void  print_limb();
    void  set_pose(MatrixXd pose_data);
    void  set_limb(MatrixXd user_limb_length);
    void  set_limb_idx(int idx, double value);
    void  set_init(bool init);
    void  print_pose_dim( );
    void  print_pose_ij(int i, int j);
    void  save_pose_data(string file_name);
    void  save_limb_data(string file_name);

};

#endif // USER_T_H
