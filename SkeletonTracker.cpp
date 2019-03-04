#include "SkeletonTracker.h"
#include "eigen_extras.h"


extern xn::UserGenerator g_UserGenerator;
extern xn::DepthGenerator g_DepthGenerator;

extern XnBool g_bDrawBackground;
extern XnBool g_bDrawPixels;
extern XnBool g_bDrawSkeleton;
extern XnBool g_bPrintID;
extern XnBool g_bPrintState;

extern XnBool g_bPrintFrameID;
extern XnBool g_bMarkJoints;




/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*  SkeletonTracker() : constructor
*  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
SkeletonTracker::SkeletonTracker(int n_users_t, int n_joints, int keep_in_mem, int rounds)
{

    n_users = n_users_t;
    time          = 0;
    init          = false;
    init_counter  = 0;
    sec_T_pose    = 100;
    n_keep_in_mem = keep_in_mem;
    n_rounds      = rounds;
    c_rounds      = 0;

    if(n_joints != 9 && n_joints != 17){
        std::cout << "Number of joints not recognized, initializing default <full skeleton>" << std::endl;
        n_joints = 17;
    }

    joint_names      = getAllJointNames(n_joints);
    skel_joint_pos   = getJointPositions(n_joints);
    skel_limb_length = getLimbLength(n_joints);
    skeleton_tree    = create_skeleton_tree(n_joints);
    user_t temp_user(MatrixXd::Zero(1,n_joints*3+14), getLimbLength(n_joints), false);

    for(int i_user = 0; i_user < n_users; i_user ++) // push empty users
        all_user_data.push_back(temp_user);

}

SkeletonTracker::SkeletonTracker(std::string file_name)
{

    n_users = 1;
    time          = 0;
    init          = true;
    init_counter  = 0;
    sec_T_pose    = 100;
    n_keep_in_mem = -1;
    n_rounds      = -1;
    c_rounds      = -1;
    skel_limb_length =  loadLimbLength(file_name);
    int n_joints  = 17;

    if(skel_limb_length.cols() == 8)
        n_joints  = 9;
    joint_names      = getAllJointNames(n_joints);
    skel_joint_pos   = getJointPositions(n_joints);
    skeleton_tree    = create_skeleton_tree(n_joints);
    user_t temp_user(MatrixXd::Zero(1,n_joints*3+14),skel_limb_length, true);

    for(int i_user = 0; i_user < n_users; i_user ++) // push empty users
        all_user_data.push_back(temp_user);

}

SkeletonTracker::SkeletonTracker( )
{

}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*  get_T_position() : Runs for sec_T_pose ms and saves the lengths of limbs
*                     of user. Once done, it is assumed that the limb length
*                     does not vary significantly.
*  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
void SkeletonTracker::get_T_position(XnUserID user , int iuser){

    if (init_counter > sec_T_pose){
        init = true;
        std::cout << "I assume you got into the T pose! " << std::endl;
        init_counter = 0.0;
    }
    else{
        std::cout << "Get into T pose! Only " << sec_T_pose - init_counter << " ms left! "  << std::endl;
        MatrixXd temp_pose           = get_pose(user, iuser, false);
        temp_pose                    = resize2DMatrix(temp_pose, temp_pose.cols()*temp_pose.rows()/3,3);
        MatrixXd global_translation  =  get_joint_data(  user, XN_SKEL_TORSO) ;
        temp_pose = translate_joint(skeleton_tree, global_translation, temp_pose);
        compute_limb_length(skeleton_tree, iuser, temp_pose);


    }

}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*  compute_limb_length() :  Goes through the skeletal hierarchy and computes
*                           the length between parent and child vector as the
*                           euclidean distance + 1
* *  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
void SkeletonTracker:: compute_limb_length(node_t node, int user, MatrixXd data){
    int pa = node.my_pos;
    int ch;
    if (!node.children.empty()){
        for(int i_joint = 0 ; i_joint < node.children.size(); i_joint++)
        {
            ch = node.children[i_joint].my_pos;
            double t = (data.row(pa)-data.row(ch)).norm() + 0.0000001;
            all_user_data[user].set_limb_idx(ch-1,t);
            compute_limb_length(node.children[i_joint],  user,  data);
        }
    }



}

void SkeletonTracker::update_pose_all()  // coordinates the update of all users
{
    XnUserID aUsers[n_users];
    XnUInt16 nUsers = n_users;
    g_UserGenerator.GetUsers(aUsers, nUsers);

    for(int i_user = 0; i_user < n_users; i_user ++)
    {
        if(g_UserGenerator.GetSkeletonCap().IsTracking(aUsers[i_user])){
            if(!init){  // while not initialized get the T pose
                if(time == 0) timer.reset();
                time = timer.elapsed();
                init_counter = init_counter + time;
                get_T_position(aUsers[i_user], i_user);

            }else{ // if initialized, gather data
                time = timer.elapsed(); timer.reset();
                MatrixXd temp_pose = get_pose(aUsers[i_user],i_user, true);
                MatrixXd pose(temp_pose.rows(), temp_pose.cols()+2);
                pose << temp_pose, time, 1.0;
                temp_pose          = resize2DMatrix(temp_pose, temp_pose.cols()*temp_pose.rows()/3,3);
                append_data_to_matrix(i_user, pose);

                init_counter = init_counter + time;


                if (init_counter > 60 && n_rounds > 0 && c_rounds < n_rounds){
                    std::cout << "------------------------------------------------------------------------------------------------------------- " << c_rounds << std::endl;
                    std::cout << "saving round " << c_rounds << std::endl;
                    all_user_data[i_user].print_pose_dim();
                    string file_name = "limb_u_" + boost::lexical_cast<std::string>(i_user) + ".txt";
                    all_user_data[i_user].save_limb_data(file_name);
                    file_name = "pose_" + boost::lexical_cast<std::string>(c_rounds) +  "_u_" +  boost::lexical_cast<std::string>(i_user) + ".txt";
                    all_user_data[i_user].save_pose_data(file_name);
                    all_user_data[i_user].set_pose(MatrixXd::Zero(1,9*3+14));

                    init_counter = 0.0;
                    c_rounds++;
                }
            }
        }
    }
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
* get_pose() : gathers pose data for a specific user, scaled down
*  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
MatrixXd SkeletonTracker::get_pose(XnUserID user, int i_user, bool normalize)                             
{
    XnSkeletonJointOrientation orientation;
    g_UserGenerator.GetSkeletonCap().GetSkeletonJointOrientation(user, XN_SKEL_TORSO, orientation);
    MatrixXd global_orientation = convert_XnOrientation_2_matrix(orientation);
    MatrixXd global_orientation_inv = global_orientation.inverse();

    MatrixXd global_translation =  get_joint_data(  user, XN_SKEL_TORSO) ;

    int N = joint_names.size();
    MatrixXd pose(N,3);
    for (int i_joint = 0; i_joint < N; i_joint ++)
    {
        pose.row(i_joint) = get_joint_data( user, joint_names[i_joint]);
    }
    if (normalize)
        pose = norm_unnorm_pose(i_user, skeleton_tree, pose , global_orientation_inv, global_translation);
    pose = resize2DMatrix(pose,1,N*3);
    global_orientation = resize2DMatrix(global_orientation,1,9);
    MatrixXd temp_pose(1, pose.cols() + global_orientation.cols() + global_translation.cols());
    temp_pose << pose, global_orientation, global_translation;

    return temp_pose;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
* get_pose() : gathers pose data for a specific user, scaled up
*  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
MatrixXd SkeletonTracker::get_pose_out(  int i_user, bool normalize)                              
{
    XnUserID aUsers[n_users];
    XnUInt16 nUsers = n_users;
    g_UserGenerator.GetUsers(aUsers, nUsers);
    XnUserID user = aUsers[i_user];
    XnSkeletonJointOrientation orientation;
    g_UserGenerator.GetSkeletonCap().GetSkeletonJointOrientation(user, XN_SKEL_TORSO, orientation);
    MatrixXd global_orientation = convert_XnOrientation_2_matrix(orientation);
    MatrixXd global_orientation_inv = global_orientation.inverse();

    MatrixXd global_translation =  get_joint_data(  user, XN_SKEL_TORSO) ;

    int N = joint_names.size();
    MatrixXd pose(N,3);
    for (int i_joint = 0; i_joint < N; i_joint ++)
    {
        pose.row(i_joint) = get_joint_data( user, joint_names[i_joint]);
    }
    if (normalize)
        pose = norm_unnorm_pose(i_user, skeleton_tree, pose , global_orientation_inv, global_translation);
    pose = resize2DMatrix(pose,1,N*3);
    global_orientation = resize2DMatrix(global_orientation,1,9);
    MatrixXd temp_pose(1, pose.cols() + global_orientation.cols() + global_translation.cols());
    temp_pose << pose, global_orientation, global_translation;

    return temp_pose*1000;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
* get_joint_data() : returns 3D pos of joint
*  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

MatrixXd SkeletonTracker::get_joint_data(XnUserID user, XnSkeletonJoint eJoint) // 
{
    XnSkeletonJointPosition position;
    g_UserGenerator.GetSkeletonCap().GetSkeletonJointPosition(user, eJoint, position);
    MatrixXd joint_position = convert_XnPosition_2_matrix(position)  / 1000.;
    return joint_position;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
* norm_unnorm_pose() :  normalize (scale down, translate to zero)
*  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
MatrixXd SkeletonTracker::norm_unnorm_pose(int user, node_t node, MatrixXd unnorm_pose, MatrixXd orientation, MatrixXd translation)      
{
    MatrixXd norm_pose = scale_pose_down( node,  user, unnorm_pose);
    norm_pose = translate_joint( node,  translation, norm_pose);
    return norm_pose;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
* unnorm_norm_pose() :  normalize (translate back, scale up)
*  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
MatrixXd SkeletonTracker::unnorm_norm_pose(int user, node_t node, MatrixXd norm_pose, MatrixXd orientation, MatrixXd translation)     
{

   MatrixXd unnorm_pose =  translate_joint( node,  translation, norm_pose);
   unnorm_pose = scale_pose_up( node,  user, unnorm_pose);
   return unnorm_pose;

}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
* scale_pose_up() :  scales the limb links to be of length 1 (according to T-pose measurement)
*  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
MatrixXd SkeletonTracker::scale_pose_up(node_t node, int user, MatrixXd small_pose)                            
{
    vector<node_t>stack;
    stack.push_back(node);
    while(!stack.empty()){
        node = stack[0];
        stack.erase(stack.begin());
        int pa = node.my_pos;
        MatrixXd vec1 = small_pose.row(pa);
        for(int i_ch = 0; i_ch < node.children.size(); i_ch ++) {
            int ch = node.children[i_ch].my_pos;
            MatrixXd vec2  = small_pose.row(ch);
            double norm    = (all_user_data[user].get_limb_idx(ch-1)  )  ; // skel_limb_length(0,ch-1);
            MatrixXd translate = vec1 + norm*(vec2 - vec1);
            small_pose = translate_joint(node.children[i_ch],  vec2 - translate, small_pose);
            stack.push_back(node.children[i_ch]);
        }
    }
    return small_pose;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
* scale_pose_down() :  scales the limb links to the original length (according to T-pose measurement)
*  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
MatrixXd SkeletonTracker::scale_pose_down(node_t node, int user, MatrixXd large_pose)                            
    vector<node_t>stack;
    stack.push_back(node);
    while(!stack.empty()){
        node = stack[0];
        stack.erase(stack.begin());
        int pa = node.my_pos;

        MatrixXd vec1 = large_pose.row(pa);
        for(int i_ch = 0; i_ch < node.children.size(); i_ch ++) {
            int ch = node.children[i_ch].my_pos;
            MatrixXd vec2 = large_pose.row(ch);
            double norm    = 1. /  (all_user_data[user].get_limb_idx(ch-1));
            MatrixXd translate = vec1 + norm*(vec2 - vec1);
            large_pose = translate_joint(node.children[i_ch],   vec2 - translate, large_pose);
            stack.push_back(node.children[i_ch]);
        }
    }

    return large_pose;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
* translate_joint() :  translates joint data and all its childrens data
*  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
MatrixXd SkeletonTracker::translate_joint(node_t node, MatrixXd translation, MatrixXd to_translate)    
{
    int pa = node.my_pos;
    to_translate.row(pa) = to_translate.row(pa) - translation;
    if (!node.children.empty()){
        for(int i_ch = 0; i_ch < node.children.size(); i_ch ++) {
            to_translate = translate_joint(node.children[i_ch], translation,  to_translate);
        }
    }
    return to_translate;

}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
* save_data() :  saves data to txt
*  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
void SkeletonTracker::save_data(std::string fileName, int user)  
{
    all_user_data[user].save_pose_data(fileName);
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
* append_data_to_matrix() :  appends current pose to data matrix
*  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
void SkeletonTracker::append_data_to_matrix(int user, MatrixXd data) 
{

  MatrixXd mat = all_user_data[user].get_pose();
  if (n_keep_in_mem < 0 || mat.rows() + data.rows() < n_keep_in_mem)
  {
    mat.conservativeResize(mat.rows()+1,Eigen::NoChange);
    mat.row(mat.rows()-1) = data;
    all_user_data[user].set_pose(mat);

  }
  else{
      if( mat.rows() + data.rows() >= n_keep_in_mem)
          all_user_data[user].set_init(true);
      MatrixXd temp_mat                 = MatrixXd::Zero(n_keep_in_mem, mat.cols());
      int N                             = data.rows();
      temp_mat.topRows(n_keep_in_mem-N) = mat.bottomRows(n_keep_in_mem-N);
      temp_mat.bottomRows(N)            = data;
      all_user_data[user].set_pose(temp_mat);

  }
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
* get_user() :  returns valid user
*  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
user_t SkeletonTracker::get_user( int user){
    if(user < all_user_data.size())
        return all_user_data[user];
    else
         throw std::invalid_argument( "Error: SkeletonTracker::get_user - User does not exist" );
}
/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
* convert_XnOrientation_2_matrix() :  converts XnSkeletonJointOrientation to eigen matrix
*  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
MatrixXd SkeletonTracker::convert_XnOrientation_2_matrix(XnSkeletonJointOrientation orientation)
{
    XnFloat* m = orientation.orientation.elements;
    MatrixXd rotation(3,3);
    rotation << m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8];
    return rotation;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
* convert_XnPosition_2_matrix() :  converts XnSkeletonJointPosition to eigen matrix
*  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
MatrixXd SkeletonTracker::convert_XnPosition_2_matrix(XnSkeletonJointPosition position){

    double x = position.position.X;
    double y = position.position.Y;
    double z = position.position.Z;
    MatrixXd pos_xyz(1,3);
    pos_xyz << x,y,z;
    return pos_xyz;

}
