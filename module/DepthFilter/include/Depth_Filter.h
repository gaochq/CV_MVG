//
// Created by buyi on 18-4-9.
//

#ifndef CV_MVG_DEPTH_FILTER_H
#define CV_MVG_DEPTH_FILTER_H

#include <iostream>
#include <vector>
#include <string>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <sophus/se3.h>

#include "cv.h"
#include "opencv2/core/core.hpp"
#include "opencv2/core/eigen.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"


using namespace std;

namespace CV_MVG
{

class Depth_Filter
{
public:
    Depth_Filter(string &tFilename, Eigen::Vector2f tImgSize, cv::Mat tCamIntrinsic,
                cv::Mat tCamUndistort = cv::Mat(5, 1, CV_32F, cv::Scalar(0)));

    ~Depth_Filter();

    //! Load Images and Poses from file
    void LoadPoseImage();

    //! Search the match point along the epipolar line
    void EpipolarSearch();

    //! Socre the result of patch match
    void NccSocore();

    inline Eigen::Vector3d World2Camera(const)

private:

    string      msFilePath;

    size_t        mHeight;
    size_t        mWidth;

    float         mfx;
    float         mfy;
    float         mcx;
    float         mcy;

    float         minv_fx;
    float         minv_fy;
    float         minv_cx;
    float         minv_cy;


    const cv::Mat       mIntrinsicMat;
    const cv::Mat       mDistortMat;
};


} //namespace CV_MVG




#endif //CV_MVG_DEPTH_FILTER_H
