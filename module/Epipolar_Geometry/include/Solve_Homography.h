//
// Created by buyi on 17-10-6.
//

#ifndef CV_MVG_SOLVE_HOMOGRAPHY_H
#define CV_MVG_SOLVE_HOMOGRAPHY_H

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
class Solve_H_Matrix
{
public:
    enum Solve_method
    {
        HM_DLT = 0,
        HM_DLT_Ransac = 1,
    };

public:
    Solve_H_Matrix();
    ~Solve_H_Matrix();

    // Solve the homography using DLT
    Eigen::Matrix3d DLT_H_Matrix(const vector<cv::Point2f> PointsA, const vector<cv::Point2f> PointsB);
    Eigen::Matrix3d DLT_H_Matrix(const vector<std::pair<cv::Point2f, cv::Point2f> > Matches);

    // Solve the homography using ransac
    Eigen::Matrix3d Ransac_H_Matrix(vector<cv::Point2f> PointsA, vector<cv::Point2f> PointsB, cv::Mat &mask);

    //Final sovle function
    Eigen::Matrix3d Solve(vector<cv::Point2f> PointsA, vector<cv::Point2f> PointsB, Solve_method method, cv::Mat &mask);


private:
    float Probe;
    float Outlier_thld;
    int Max_Iteration;
    Solve_method mMethod;

    Epipolar_base epipolar_base;
};
}

#endif //CV_MVG_SOLVE_HOMOGRAPHY_H
