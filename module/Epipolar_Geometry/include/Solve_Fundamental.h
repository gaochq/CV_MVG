//
// Created by buyi on 17-10-3.
//

#ifndef CV_MVG_SOLVE_FUNDAMENTAL_H
#define CV_MVG_SOLVE_FUNDAMENTAL_H

#include <iostream>
#include <vector>

#include <Eigen/Dense>


#include "cv.h"
#include "opencv2/core/core.hpp"
#include "opencv2/core/eigen.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"

#include "Epipolar_base.h"

namespace CV_MVG
{
class Solve_F_Matrix
{
public:
    enum Solve_method
    {
        FM_8Point = 0,
        FM_8Point_Ransac = 1,
        FM_7Point = 2
    };

public:
    Solve_F_Matrix(Solve_method method);
    ~Solve_F_Matrix();

    //Solve Funmental matrix using 8 Point method
    vector<Eigen::Matrix3d> EightPoint_F_Matrix(vector<cv::Point2f> PointsA, vector<cv::Point2f> PointsB);
    vector<Eigen::Matrix3d> EightPoint_F_Matrix(vector< std::pair<cv::Point2f, cv::Point2f> > Matches);

    //Solve Funmental matrix using 7 Point method
    vector<Eigen::Matrix3d> SevenPoint_F_Matrix(vector<cv::Point2f> PointsA, vector<cv::Point2f> PointsB);
    vector<Eigen::Matrix3d> SevenPoint_F_Matrix(vector< std::pair<cv::Point2f, cv::Point2f> > Matches);

    //Solve Funmental matrix using ransac
    Eigen::Matrix3d Ransac_F_Matrix(vector<cv::Point2f> PointsA, vector<cv::Point2f> PointsB, cv::Mat &mask, Solve_method method);

    //Final sovle function
    Eigen::Matrix3d solve(vector<cv::Point2f> PointsA, vector<cv::Point2f> PointsB, cv::Mat &mask);
private:
    Solve_method Method;
    float Probe;
    float Outlier_thld;
    int Max_Iteration;

    Epipolar_base epipolar_base;
};
} // namespace CV_MVG




#endif //CV_MVG_SOLVE_FUNDAMENTAL_H
