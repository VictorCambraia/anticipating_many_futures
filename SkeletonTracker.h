#ifndef SKELETONTRACKER_H
#define SKELETONTRACKER_H


#include <XnCppWrapper.h>
#include <Dense>
#include <vector>
#include <ctime>
#include "timer.h"
#include "skeleton_helper.h"
#include "user_t.h"

using std::vector;
using Eigen::MatrixXd;



class SkeletonTracker
{
public:
    // constructor
    SkeletonTracker(int n_users_t, int n_joints, int keep_in_mem, int rounds);
    SkeletonTracker(std::string filename);
    SkeletonTracker( );
    // public variables
    vector<XnSkeletonJoint> joint_names;
    vector<  MatrixXd >     skel_joint_pos;
    MatrixXd                skel_limb_length;
    vector <user_t>         all_user_data;
    int n_keep_in_mem;      // the number of samples to base predictions on
    node_t skeleton_tree;   // tree of joints in hierarchical manner (shoulder is parent of elbow etc)

    // public methods
    void update_pose_all();                                             // coordinates the update of all users -
    MatrixXd get_pose(XnUserID user, int i_user, bool normalize);         // gathers data for a specific user
    MatrixXd get_pose_out( int i_user, bool normalize);                     // gathers data for a specific user out
    MatrixXd norm_unnorm_pose(int user, node_t node, MatrixXd unnorm_pose, MatrixXd orientation, MatrixXd translation);     // center - scale down - orientation
    MatrixXd unnorm_norm_pose(int user, node_t node, MatrixXd norm_pose,   MatrixXd orientation, MatrixXd translation);    // orientation - scale up - decenter
    MatrixXd get_joint_data(XnUserID user, XnSkeletonJoint eJoint );   // returns 3D pos of joint
    user_t get_user( int user);   // returns user
    void save_data(std::string fileName, int user);               // saves data



private:
    // private variables
    int    n_users;         // number of users
    bool   init;            // is true if init_counter > sec_T_pose
    double init_counter;    // counts how long user has been in T pose
    double sec_T_pose;      // how long should the user be in T pose
    Timer  timer;           // returns elapsed time
    double time;            // counts time between frames
    MatrixXd pose_to_change;


    int n_rounds;           // how many times should be saved
    int c_rounds;           // tracks rounds

    // private methods
    void get_T_position(XnUserID user, int iuser);                          // updates user limb lengths until init == true
    void compute_limb_length(node_t node, int user, MatrixXd data);
    MatrixXd scale_pose_up(node_t node,int user, MatrixXd small_pose);                            // used by unnorm_norm_pose to scale data up
    MatrixXd scale_pose_down(node_t node, int user, MatrixXd large_pose);                          // used by norm_unnorm_pose to scale data down
    MatrixXd translate_joint(node_t node, MatrixXd translation, MatrixXd to_translate);  // translates joint data and all its childrens data
    void append_data_to_matrix(int user, MatrixXd data);                    // appends data to data
    MatrixXd convert_XnOrientation_2_matrix(XnSkeletonJointOrientation orientation);
    MatrixXd convert_XnPosition_2_matrix(XnSkeletonJointPosition position);


};

#endif // SKELETONTRACKER_H
