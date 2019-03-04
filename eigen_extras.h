#include<eigen3/Eigen/Dense>
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <boost/lexical_cast.hpp>

using Eigen::MatrixXd;
using Eigen::RowVectorXd;
using Eigen::VectorXd;
using namespace std;


struct  normal_param_t
{
    RowVectorXd mean;
    RowVectorXd std;
};

MatrixXd resize2DMatrix(MatrixXd mat, int N, int M);

MatrixXd ReadMatrixFromFile(string fileName);
void WriteMatrixToFile(string fileName, MatrixXd data);
RowVectorXd SortVectorAscending(RowVectorXd data, RowVectorXd *ids);
RowVectorXd SortVectorDescending(RowVectorXd data, RowVectorXd *ids);
void RegisterVector(RowVectorXd data, char* fileName, double sec);
void RegisterVector(string dataName, RowVectorXd data, char* fileName, double sec);
RowVectorXd ConvertToEigenVector(vector <int> inVector);


MatrixXd NormalizeData(MatrixXd data, normal_param_t params);
VectorXd NormalizeData(VectorXd data, normal_param_t params);
RowVectorXd NormalizeData(RowVectorXd data, normal_param_t params);
RowVectorXd DeNormalize(RowVectorXd normalData, normal_param_t params);
double DeNormalize(double normalData, normal_param_t params);
normal_param_t GetNormalizationParams(MatrixXd data);
void GetVectorNormalParams(VectorXd data, double& mean_data, double& std_data);
