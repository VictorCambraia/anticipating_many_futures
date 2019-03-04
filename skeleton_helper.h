
#include "user_t.h"

// OpenNI
 #include <XnCppWrapper.h>
 #include <Dense>

 // std
 #include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <stdio.h>


using std::vector;
using Eigen::MatrixXd;




struct node_t {
 int my_pos;
 vector <node_t> children;
} ;


std::vector<XnSkeletonJoint> getAllJointNames(int n_joints);
MatrixXd  getLimbLength(int n_joints);
MatrixXd  loadLimbLength(std::string filename);
std::vector<MatrixXd> getJointPositions(int n_joints);
node_t create_skeleton_tree(int n_joints);
vector<int> traverse_tree(node_t root, vector <int> tree);
double euclidean_distance(MatrixXd one, MatrixXd two);



