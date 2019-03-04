#include "Integrator.h"


/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
* Integrator() : The integrator class is an interface between the motion prediction network (VAE)
		 and the skeleton tracker.
*  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

Integrator::Integrator(SkeletonTracker skeleton_in, std::string vae_skel_param_path, std::string vae_trans_param_path, bool sampler_var){

    skeleton    = skeleton_in;
    Vaecoder t1(vae_skel_param_path); // all body joints in 3D space
    vae_skel    = t1;
    Vaecoder t2(vae_trans_param_path); // translation in 3D space
    vae_trans   = t2;
    n_samples  = skeleton.n_keep_in_mem;
    sample_var = sampler_var;
    n_joints   = skeleton.joint_names.size();
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
* get_current_frames() :  gets the pose of valid user
*  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
MatrixXd Integrator::get_current_frames(int user){
    user_t curr_user  = skeleton.get_user(user);
    if (curr_user.get_init())
        return curr_user.get_pose();
    else
        throw std::invalid_argument( "Integrator::get_current_frames user is not initialized yet" );
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
* convert_skel_to_vae() :  converts T skeleton frames into a single vector (for the feed-forward network as input)
*  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
MatrixXd Integrator::convert_skel_to_vae(MatrixXd skel_frames){
    // returns matrix (1 x T*(n_joints-1)*3)
    skel_frames = skel_frames.rightCols(skel_frames.cols()-3); // cut the (0,0,0) of root
    skel_frames = skel_frames.leftCols(skel_frames.cols()-14); // cut the orientation (9), translation (3), time (1) and tracked(1)
    skel_frames = resize2DMatrix(skel_frames, 1, skel_frames.rows()*skel_frames.cols());
    return skel_frames;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
* convert_skel_to_trans() :  converts T translation vectors into a single vector (for the feed-forward network as input)
*  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
MatrixXd Integrator::convert_skel_to_trans(MatrixXd skel_frames){
     // returns matrix (1 x T*3)
    skel_frames = skel_frames.leftCols(skel_frames.cols()-2); // cut time (1) and tracked(1)
    skel_frames = skel_frames.rightCols(3); // cut the pose (n_joints*3), orientation (9)
    skel_frames = resize2DMatrix(skel_frames, 1, skel_frames.rows()*skel_frames.cols());
    return skel_frames;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
* convert_vae_to_skel() :  converts a single vector into T skeleton frames  
*  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
MatrixXd Integrator::convert_vae_to_skel(MatrixXd vae_frames){
    // returns matrix (T x n_joints*3)
    vae_frames = resize2DMatrix(vae_frames, n_samples, vae_frames.rows()*vae_frames.cols() / n_samples);
    MatrixXd temp = MatrixXd::Zero(vae_frames.rows(),vae_frames.cols()+3);
    temp.rightCols(vae_frames.cols()) = vae_frames;
    return temp;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
* convert_vae_to_trans() :  converts a single vector into T translation vectors
*  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
MatrixXd Integrator::convert_vae_to_trans(MatrixXd vae_trans){
    // returns matrix (T x  3)
    return resize2DMatrix(vae_trans, n_samples,  3);

}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
* translate_skel_by_pred() :  translate the skeleton in 3D space by predicted vectors
*  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
MatrixXd Integrator::translate_skel_by_pred(MatrixXd skel_frames, MatrixXd pred_trans){

    for(int t = 0; t < n_samples; t++){
        MatrixXd temp = skel_frames.row(t);
        temp          = resize2DMatrix(temp, n_joints, 3);
        temp          = skeleton.unnorm_norm_pose(0,skeleton.skeleton_tree, temp,temp,-pred_trans.row(t));
        skel_frames.row(t)  = resize2DMatrix(temp, 1, temp.rows()*temp.cols());
    }
    return skel_frames;
}


/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
* sample_pred() :  sample different future joint trajectories
*  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
std::vector<mu_sigma_struct> Integrator::sample_pred(int user, int n_pred){

    vector<mu_sigma_struct> samples;
    MatrixXd  frame     = get_current_frames(user);
    MatrixXd skel_trans = convert_skel_to_trans(frame);
    frame               = convert_skel_to_vae(frame);

    for(int p = 0; p < n_pred; p ++){
        mu_sigma_struct sample_tran = sample_trans(skel_trans);
        mu_sigma_struct sample = vae_skel.e_s_t_s_d(frame);
        sample.mu              = convert_vae_to_skel(sample.mu);
        sample_tran.mu         = convert_vae_to_trans(sample_tran.mu);
        sample.mu  = translate_skel_by_pred(sample.mu, sample_tran.mu);
        samples.push_back(sample);
    }
    return samples;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
* sample_trans() :  sample different future translation trajectories
*  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
mu_sigma_struct  Integrator::sample_trans(MatrixXd skel_trans){
    return vae_trans.e_s_t_s_d(skel_trans);
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
* predict() :  predict joint trajectory (decode the latent mean)
*  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
MatrixXd Integrator::predict(int user){
    MatrixXd  frame     = get_current_frames(user);
    MatrixXd skel_trans = convert_skel_to_trans(frame);
    frame               = convert_skel_to_vae(frame);
    mu_sigma_struct pred_skel = vae_skel.e_t_d(frame);
    MatrixXd pred_tran  = predict_trans(skel_trans);

    pred_skel.mu           = convert_vae_to_skel(pred_skel.mu);
    pred_tran              = convert_vae_to_trans(pred_tran);
    pred_skel.mu           = translate_skel_by_pred(pred_skel.mu, pred_tran);
    return pred_skel.mu*1000;

}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
* transform() :  translate skeleton
*  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
MatrixXd Integrator::transform(int user){
    MatrixXd  frame      = get_current_frames(user);
    MatrixXd  skel_trans = convert_skel_to_trans(frame);
    skel_trans           = translate_skel_by_pred(frame, skel_trans);
    return skel_trans;

}
/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
* predict_trans() :  predict translation trajectory (decode the latent mean)
*  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
MatrixXd  Integrator::predict_trans(MatrixXd skel_trans){
    return vae_trans.e_t_d(skel_trans).mu;
}

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
* convert_to_XnPoint3D() :  convert matrix3D to vector of XnPoint3Ds
*  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
vector<vector<XnPoint3D> > Integrator::convert_to_XnPoint3D(MatrixXd matrix3D){
   vector<vector<XnPoint3D> > all_postions;
   for(int t = 0; t < n_samples; t ++)
   {
       vector<XnPoint3D> temp_vec;
       MatrixXd current_mat = resize2DMatrix(matrix3D.row(t), n_joints, 3);
       for(int j = 0; j < n_joints; j++){
            XnPoint3D point;
            point.X  = current_mat(j,0);
            point.Y  = current_mat(j,1);
            point.Z  = current_mat(j,2);
            temp_vec.push_back(point);
       }
       all_postions.push_back(temp_vec);
   }
    return all_postions;
}


