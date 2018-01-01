//
// Created by buyi on 17-10-6.
//

#include "Feature_base.h"
#include "Solve_Fundamental.h"

using namespace std;



int main(int argc, char **argv)
{
    if(argc!=3)
    {
        cout<< "Usage: FundamentalTest ImageA ImageB" << endl;
        return 0;
    }

    cv::Mat ImageA, ImageB;
    ImageA = cv::imread(argv[1]);
    ImageB = cv::imread(argv[2]);

    if(!ImageA.data || !ImageB.data)
    {
        cout<< "Invalid Images"<<endl;
    }

    vector<cv::Point2f> PointsA, PointsB;
    CV_MVG::Feature_base::Extract_GoodPoints(ImageA, ImageB, PointsA, PointsB);

    Eigen::Matrix3d F_Mat;
    cv::Mat mask;
    CV_MVG::Solve_F_Matrix Fundamental_solver;
    F_Mat = Fundamental_solver.Solve(PointsA, PointsA, CV_MVG::Solve_F_Matrix::FM_8Point, &mask);
    cout<< F_Mat <<endl;

    cv::Mat tF_Mat;
    tF_Mat = cv::findFundamentalMat(PointsA, PointsB, cv::FM_8POINT, 3, 0.99);
    cout<< tF_Mat <<endl;

    return 0;
}