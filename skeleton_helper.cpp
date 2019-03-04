#include "skeleton_helper.h"



/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*  getAllJointNames() : returns vector of joint names for full body
*  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
std::vector<XnSkeletonJoint> getAllJointNames(int n_joints){
 std::vector<XnSkeletonJoint> v;
 v.reserve(n_joints);
 v.push_back(XN_SKEL_TORSO);
 v.push_back(XN_SKEL_NECK);
 v.push_back(XN_SKEL_HEAD);
 v.push_back(XN_SKEL_LEFT_SHOULDER);
 v.push_back(XN_SKEL_RIGHT_SHOULDER);
 v.push_back(XN_SKEL_LEFT_ELBOW);
 v.push_back(XN_SKEL_RIGHT_ELBOW);
 v.push_back(XN_SKEL_LEFT_HAND);
 v.push_back(XN_SKEL_RIGHT_HAND);
 if(n_joints == 17){
     v.push_back(XN_SKEL_RIGHT_HIP);
     v.push_back(XN_SKEL_LEFT_KNEE);
     v.push_back(XN_SKEL_RIGHT_KNEE);
     v.push_back(XN_SKEL_LEFT_HIP);
     v.push_back(XN_SKEL_RIGHT_ANKLE);
     v.push_back(XN_SKEL_LEFT_ANKLE);
     v.push_back(XN_SKEL_LEFT_FOOT);
     v.push_back(XN_SKEL_RIGHT_FOOT);
 }
 return v;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*  getJointPositions() : returns vector of positions of custom skeleton
*  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
std::vector<MatrixXd> getJointPositions(int n_joints){
  std::vector<MatrixXd > v;
  v.reserve(n_joints);
  MatrixXd vt(1,3);
  vt << 0.,0.,1.; v.push_back(vt);             // 0 torso
  vt << 0.,1.,1.; v.push_back(vt);             // 1 neck
  vt << 0.,1.5,1.; v.push_back(vt);            // 2 head
  vt << -0.5,1.,1.; v.push_back(vt);           // 3 left shoulder
  vt << 0.5,1.,1.; v.push_back(vt);            // 4 right shoulder
  vt << -0.5,0.5,1.; v.push_back(vt);          // 5 left elbow
  vt << 0.5,0.5,1.; v.push_back(vt);           // 6 right elbow
  vt << -0.5,0.2,1; v.push_back(vt);           // 7 left hand
  vt << 0.5,0.2,1.; v.push_back(vt);           // 8 right hand
  if (n_joints == 17){
      vt << -0.5,0.,1.; v.push_back(vt);       // 9 left hip
      vt << 0.5,0.,1.; v.push_back(vt);        // 10 right hip
      vt << -0.5,-0.5,1.; v.push_back(vt);     // 11 left knee
      vt << 0.5,-0.5,1.; v.push_back(vt);      // 12 right knee
      vt << -0.5,-1.,1.; v.push_back(vt);      // 13 left ankle
      vt << 0.5,-1.,1.; v.push_back(vt);       // 14 right ankle
      vt << -0.5,-1.,0.5; v.push_back(vt);     // 15 left foot
      vt << 0.5,-1,0.5; v.push_back(vt);       // 16 right foot
  }
  return v;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*  getLimbLength() : returns vector of limb lengths of custum skeleton
*  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
MatrixXd getLimbLength(int n_joints){
   int p = 1;
   if(n_joints == 17) p++;
   MatrixXd v = MatrixXd::Zero(1,n_joints - p);

   v(0,0) = (1.5);             // torso - neck
   v(0,1) = (0.3);             // neck  - head
   v(0,2) = (0.5);             // neck  - left shoulder
   v(0,3) = (0.5);             // neck  - right shoulder
   v(0,4) = (0.7);             // left shoulder  - left elbow
   v(0,5) = (0.7);             // right shoulder - right elbow
   v(0,6) = (0.5);             // left elbow     - left hand
   v(0,7) = (0.5);             // right elbow    - right hand
   if (n_joints == 17){
       v(0,8) = (0.5);         // torso - left hip
       v(0,9) = (0.5);         // torso - right hip
       v(0,10) = (0.8);         // left hip    - left knee
       v(0,11) = (0.8);         // right hip   - right knee
       v(0,12) = (0.8);         // left knee   - left ankle
       v(0,13) = (0.8);         // right knee  - right ankle
       v(0,14) = (0.2);         // left ankle  - left foot
       v(0,15) = (0.2);         // right ankle - right foot
   }
   return v;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*  loadLimbLength() : loads limb length from file
*  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
MatrixXd loadLimbLength(std::string fileName){
   std::ifstream  text_filestream(fileName.c_str());
   int n;
   text_filestream >> n;
   MatrixXd v = MatrixXd::Zero(1,n);

   for(int i = 0; i < n; i++)
       text_filestream >> v(0,i);
   return v;
}



/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*  create_skeleton_tree() : returns root of skeleton tree
*  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
node_t create_skeleton_tree(int n_joints) {

   node_t root; root.my_pos = 0;
   node_t neck; neck.my_pos = 1;
   node_t head; head.my_pos = 2;
   node_t lshoulder; lshoulder.my_pos = 3;
   node_t rshoulder; rshoulder.my_pos = 4;
   node_t lelbow; lelbow.my_pos = 5;
   node_t relbow; relbow.my_pos = 6;
   node_t lhand; lhand.my_pos   = 7;
   node_t rhand; rhand.my_pos   = 8;

   lelbow.children.push_back(lhand);
   relbow.children.push_back(rhand);
   lshoulder.children.push_back(lelbow);
   rshoulder.children.push_back(relbow);
   neck.children.push_back(head);
   neck.children.push_back(lshoulder);
   neck.children.push_back(rshoulder);
   root.children.push_back(neck);

    if (n_joints == 17){
        node_t lhip; lhip.my_pos     = 9;
        node_t rhip; rhip.my_pos     = 10;
        node_t lknee; lknee.my_pos   = 11;
        node_t rknee; rknee.my_pos   = 12;
        node_t lankle; lankle.my_pos = 13;
        node_t rankle; rankle.my_pos = 14;
        node_t lfoot; lfoot.my_pos   = 15;
        node_t rfoot; rfoot.my_pos   = 16;
        lankle.children.push_back(lfoot);
        rankle.children.push_back(rfoot);
        lknee.children.push_back(lankle);
        rknee.children.push_back(rankle);
        lhip.children.push_back(lknee);
        rhip.children.push_back(rknee);
        root.children.push_back(lhip);
        root.children.push_back(rhip);
    }
    return root;
}
/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
* traverse_tree() : traverses tree
*  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
vector <int> traverse_tree(node_t root, vector <int> tree){
    if (root.children.empty())
        return tree;
     else{
        tree.push_back(root.my_pos);
        for(int child = 0; child < root.children.size(); child++){
            tree =  traverse_tree(root.children[child],tree);
        }
    }
    return tree;
}

double euclidean_distance(MatrixXd one, MatrixXd two){

    if (one.cols() != two.cols() || one.rows() != two.rows()){
        throw "Dimension missmatch in euclidean_distance!";
    }

    return (double) (one-two).squaredNorm();


}
