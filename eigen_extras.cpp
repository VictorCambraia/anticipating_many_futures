#include "eigen_extras.h"

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*  resize2DMatrix() : resize a 2D matrix rowwise
*  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

MatrixXd resize2DMatrix(MatrixXd mat, int N, int M){

    if (N*M != mat.rows()*mat.cols()){
        std::cout << "Error in resize2DMatrix - dimension missmatch" << std::endl;
        return mat;
    }

    MatrixXd newMat(N,M);
    int k = 0;
    int l = 0;
    for(int i = 0; i < mat.rows(); i++){
        for(int j = 0; j < mat.cols(); j++){
            if (l == M){ k++; l = 0;}
            newMat(k,l) = mat(i,j);
            l++;
        }

    }
    return newMat;
}


void WriteMatrixToFile(string fileName, MatrixXd data)
{
    int nRow = data.rows();
    int nCol = data.cols();
    ofstream fileWriter;
    fileWriter.open(fileName.c_str());

    fileWriter << nRow << endl;
    fileWriter << nCol << endl;

    for(int i = 0; i < nRow; i++)
        {
            for(int j = 0; j < nCol; j++)
                {
                    fileWriter << data(i,j) << endl;
                }
        }

    fileWriter.close();
}

MatrixXd ReadMatrixFromFile(string fileName)
{
    MatrixXd data;
    ifstream fileReader;
    fileReader.open(fileName.c_str());
    int nRow, nCol;
    string line;

    if (fileReader.is_open())
        {

            getline(fileReader, line);
            nRow = boost::lexical_cast<int>(line);
            getline(fileReader, line);
            nCol = boost::lexical_cast<int>(line);
            data.resize(nRow, nCol);
            for(int i = 0; i < nRow; i++)
                {
                    for(int j = 0; j < nCol; j++)
                        {
                            getline(fileReader, line);
                            data(i,j) = boost::lexical_cast<double>(line);
                        }
                }
        }

    fileReader.close();
    return data;
}

RowVectorXd SortVectorAscending(RowVectorXd data, RowVectorXd *ids)
{
    int N = data.cols();
    *ids = RowVectorXd::Zero(N);
    for(int i = 0; i < N; i++)
        ids->coeffRef(i) = i;
    RowVectorXd sData = data;
    bool swapped = true;
    double temp;
    while (swapped == true)
        {
            swapped = false;
            for (int i = 1; i < N; i++)
                {
                    if( sData(i) < sData(i - 1) )
                        {
                            temp = (*ids)(i);
                            (*ids)(i) = (*ids)(i-1);
                            (*ids)(i-1) = temp;
                            temp = sData(i);
                            sData(i) = sData(i - 1);
                            sData(i - 1) = temp;
                            swapped = true;
                        }
                }
        }
    return sData;
}

RowVectorXd SortVectorDescending(RowVectorXd data, RowVectorXd *ids)
{
    int N = data.cols();
    *ids = RowVectorXd::Zero(N);
    for(int i = 0; i < N; i++)
        ids->coeffRef(i) = i;
    RowVectorXd sData = data;
    bool swapped = true;
    double temp;
    while (swapped == true)
        {
            swapped = false;
            for (int i = 1; i < N; i++)
                {
                    if( sData(i) > sData(i - 1) )
                        {
                            temp = (*ids)(i);
                            (*ids)(i) = (*ids)(i-1);
                            (*ids)(i-1) = temp;
                            temp = sData(i);
                            sData(i) = sData(i - 1);
                            sData(i - 1) = temp;
                            swapped = true;
                        }
                }
        }
    return sData;
}

void RegisterVector(RowVectorXd data, char* fileName, double sec)
{
    int i;
    ofstream fileWriter;
    fileWriter.open(fileName, ios::app);
    fileWriter << sec << endl;
    int nData = data.cols();
    fileWriter << nData << endl;
    for(i = 0; i < nData; i++)
        {
            fileWriter << data(i) << endl;
        }
    fileWriter.close();
}


void RegisterVector(string dataName, RowVectorXd data, char* fileName, double sec)
{
    int i;
    ofstream fileWriter;
    fileWriter.open(fileName, ios::app);
    fileWriter << dataName << endl;
    fileWriter << sec << endl;
    int nData = data.cols();
    fileWriter << nData << endl;
    for(i = 0; i < nData; i++)
        {
            fileWriter << data(i) << endl;
        }
    fileWriter.close();

}

RowVectorXd ConvertToEigenVector(vector <int> inVector)
{
    RowVectorXd outVector(inVector.size());
    for(int i = 0; i < inVector.size(); i++)
        outVector(i) = inVector[i];
    return outVector;
}

// normalizes a matrix row by row
MatrixXd NormalizeData(MatrixXd data, normal_param_t params)
{
    int nData = data.rows();
    int dimData = data.cols();
    MatrixXd normalData(nData, dimData);
    for (int i = 0; i < nData; i++)
        {
            normalData.row(i) = NormalizeData((RowVectorXd)data.row(i), params);
        }
    return normalData;
}

// normalizes a column vector
VectorXd NormalizeData(VectorXd data, normal_param_t params)
{
    int nData = data.rows();
    VectorXd normalData(nData);

    for (int i = 0; i < nData; i++)
        {
            normalData(i) = (data(i) - params.mean(0)) / params.std(0);
        }
    return normalData;
}

// normalizes a row vector
RowVectorXd NormalizeData(RowVectorXd data, normal_param_t params)
{
    return (data - params.mean).cwiseQuotient(params.std);
}

// Denormalizes a vector
RowVectorXd DeNormalize(RowVectorXd normalData, normal_param_t params)
{
    return (normalData.cwiseProduct(params.std) + params.mean);
}

// Denormalizes a double value
double DeNormalize(double normalData, normal_param_t params)
{
    return (normalData * (params.std(0)) + params.mean(0));
}

// Returns columnwise normalization parameters
normal_param_t GetNormalizationParams(MatrixXd data)
{
    normal_param_t res;
    res.mean.resize(data.cols());
    res.std.resize(data.cols());
    double m, s;

    for (int d = 0; d < data.cols(); d++)
        {
            GetVectorNormalParams(data.col(d), m, s);
            res.mean(d) = m;
            res.std(d) = s;
        }
    return res;
}

// Returns vector normalization parameters
void GetVectorNormalParams(VectorXd data, double& mean_data, double& std_data)
{
    int nData = data.rows();
    double mean = 0;
    double mean_squared = 0;

    for (int i = 0; i < nData; i++)
        {
            mean += data(i);
            mean_squared += data(i) * data(i);
        }
    mean /= nData;
    mean_squared /= nData;

    mean_data = mean;
    std_data = sqrt(mean_squared - mean * mean);
}



