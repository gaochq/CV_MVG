//
// Created by buyi on 17-11-5.
//

#include "Feature_base.h"
#include "Solve_Homography.h"

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
    CV_MVG::Solve_H_Matrix Homography_solver;
    F_Mat = Homography_solver.Solve(PointsA, PointsB, CV_MVG::Solve_H_Matrix::HM_DLT_Ransac, mask);
    cout<< F_Mat <<endl;

    cv::Mat tF_Mat;
    tF_Mat = cv::findHomography(PointsA, PointsB, CV_RANSAC, 3);
    cout<< tF_Mat <<endl;

    return 0;
}