//
// Created by buyi on 18-4-9.
//

#include <Depth_Filter.h>


namespace CV_MVG
{

Depth_Filter::Depth_Filter(string &tFilename, Eigen::Vector2f tImgSize, cv::Mat tCamIntrinsic,
                           cv::Mat tCamUndistort): msFilePath(tFilename), mDistortMat(tCamUndistort)
{
    mHeight = tImgSize[0];
    mWidth = tImgSize[1];

    mfx = tCamIntrinsic.at<float>(0, 0);
    mfy = tCamIntrinsic.at<float>(1, 1);
    mcx = tCamIntrinsic.at<float>(0, 2);
    mcy = tCamIntrinsic.at<float>(1, 2);

    minv_fx = 1/mfx;
    minv_fy = 1/mfy;
    minv_cx = -mcx*minv_fx;
    minv_cy = -mcy*minv_fy;


}

Depth_Filter::~Depth_Filter()
{

}



} // namespace CV_MVG