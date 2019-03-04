#ifndef INTEGRATOR_H
#define INTEGRATOR_H


#include <XnCppWrapper.h>
#include <Dense>
#include <vector>
#include "SkeletonTracker.h"
//#include "skeleton_helper.h"
#include "eigen_extras.h"
#include "Vaecoder.h"
#include <iostream>
#include <fstream>
#include <string>
#include <stdio.h>
#include <math.h>

using namespace Eigen;

class Integrator
{
public:
    Integrator(SkeletonTracker skeleton_in, std::string vae_skel_param_path, std::string vae_trans_param_path, bool sampler_var);
    MatrixXd get_current_frames(int user);
    MatrixXd convert_skel_to_vae(MatrixXd skel_frames);
    MatrixXd convert_skel_to_trans(MatrixXd skel_frames);
    MatrixXd convert_vae_to_skel(MatrixXd vae_frames);
    MatrixXd convert_vae_to_trans(MatrixXd vae_trans);
    MatrixXd translate_skel_by_pred(MatrixXd skel_frames, MatrixXd pred_trans);
    MatrixXd transform(int user);

    std::vector <mu_sigma_struct> sample_pred(int user, int n_pred);
    mu_sigma_struct sample_trans(MatrixXd skel_trans);
    MatrixXd predict(int user);
    MatrixXd predict_trans(MatrixXd skel_trans);
    vector<vector<XnPoint3D> > convert_to_XnPoint3D(MatrixXd matrix3D);
    SkeletonTracker skeleton;

private:

    Vaecoder       vae_skel;
    Vaecoder       vae_trans;
    int            n_samples;
    int            n_joints;
    bool           sample_var;
};

#endif // INTEGRATOR_H
