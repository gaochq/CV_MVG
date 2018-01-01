//
// Created by buyi on 18-1-1.
//

#include <iostream>
#include <fstream>
#include "Sprase_ImageAlign.h"

using namespace std;

void loadBlenderDepthmap(const std::string file_name, cv::Mat& img)
{
    int mheight = 480;
    int mwidth = 752;
    std::ifstream file_stream(file_name.c_str());
    assert(file_stream.is_open());
    img = cv::Mat(mheight, mwidth, CV_32FC1);
    float * img_ptr = img.ptr<float>();
    float depth;
    for(int y=0; y<mheight; ++y)
    {
        for(int x=0; x<mwidth; ++x, ++img_ptr)
        {
            file_stream >> depth;
            Eigen::Vector3d point = CV_MVG::Frame::Pixel2Camera(cv::Point2f(x, y), 1.0);
            point.normalize();

            // blender:
            Eigen::Vector2d uv(point(0)/point(2), point(1)/point(2));
            *img_ptr = depth * sqrt(uv[0]*uv[0] + uv[1]*uv[1] + 1.0);

            // povray
            // *img_ptr = depth/100.0; // depth is in [cm], we want [m]

            if(file_stream.peek() == '\n' && x != mwidth-1 && y != mheight-1)
                printf("WARNING: did not read the full depthmap!\n");
        }
    }
}


int main(int argc, char **argv)
{
    if(argc!=2)
    {
        cout << "Usage: test_SpraseImg_alignment Path_To_data" <<endl;
        return 0;
    }

    CV_MVG::FramePtr frame_ref_;
    CV_MVG::FramePtr frame_cur_;
    Sophus::SE3 T_ref_wg(Eigen::Quaterniond(0.0000, 0.8227, 0.2149, 0.0000), Eigen::Vector3d(0.1131, 0.1131, 2.0000));
    Sophus::SE3 T_cur_wg(Eigen::Quaterniond(0.0000, 0.8227, 0.2148, 0.0000), Eigen::Vector3d(0.2263, 0.2262, 2.0000));

    CV_MVG::Sprase_ImgAlign mSpraseAlign(4, 0, 30);

    //! Read the first color image and construct the reference frame
    std::string ImgStr1 = string(argv[1]) + "/1.png";
    std::string ImgStr2 = string(argv[1]) + "/1.depth";
    std::string ImgStr3 = string(argv[1]) + "/2.png";

    cv::Mat img1 = cv::imread(ImgStr1, 0);
    if(!img1.data)
    {
        cout<< "Empty Image!" << endl;
        return 0;
    }

    Sophus::SE3 T_w_g = T_ref_wg.inverse();
    frame_ref_.reset(new CV_MVG::Frame(img1));
    frame_ref_->mPose = T_ref_wg.inverse();

    //! Create MapPoint
    cv::Mat depthmap;
    loadBlenderDepthmap(ImgStr2, depthmap);
    cv::goodFeaturesToTrack(frame_ref_->Img_Pyr[0], frame_ref_->mvKps, 400, 0.1, 30);
    for (int i = 0; i < frame_ref_->mvKps.size(); ++i)
    {
        cv::Point2f it = frame_ref_->mvKps[i];
        float depth = depthmap.at<float>(it.y, it.x);
        Eigen::Vector3d pt_pos_cam = CV_MVG::Frame::Pixel2Camera(it, 1.0);
        pt_pos_cam.normalize();
        frame_ref_->mvNormals.push_back(pt_pos_cam);

        Eigen::Vector3d pt_pos_wd = frame_ref_->mPose.inverse()*(pt_pos_cam*depth);
        frame_ref_->mvMapPoints.push_back(pt_pos_wd);
    }

    cout << "Added %d 3d pts to the reference frame" <<  frame_ref_->mvKps.size() << endl;

    cv::Mat img2 = cv::imread(ImgStr3, 0);
    frame_cur_.reset(new CV_MVG::Frame(img2));
    frame_cur_->mPose = frame_ref_->mPose;

    CV_MVG::TicToc tc;
    mSpraseAlign.Run(frame_cur_, frame_ref_);
    //std::cout << tc.toc() << std::endl;

    Sophus::SE3 T_f_gt = frame_cur_->mPose*T_cur_wg;

    cout << tc.toc() << "ms" <<"---"<< "Translation error: " << T_f_gt.translation().norm() << endl;
}