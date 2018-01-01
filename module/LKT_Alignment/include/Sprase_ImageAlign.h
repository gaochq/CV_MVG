//
// Created by buyi on 17-12-30.
//

#ifndef CV_MVG_SPRASE_IMAGEALIGN_H
#define CV_MVG_SPRASE_IMAGEALIGN_H

#include <iostream>
#include <memory>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/eigen.hpp>

#include <sophus/se3.h>
#include "tic_toc.h"

#include "ceres/ceres.h"

namespace CV_MVG
{

static const int mHalf_PatchSize = 4;

class Frame
{
public:
    Frame(cv::Mat tImage)
    {
        Img_Pyr.resize(5);
        Img_Pyr[0] = tImage;
        for (int i = 1; i <= 4; ++i)
        {
            Img_Pyr[i] = cv::Mat(Img_Pyr[i-1].rows/2, Img_Pyr[i-1].cols/2, CV_8U);
            cv::pyrDown(Img_Pyr[i-1], Img_Pyr[i]);
        }
    }

    static Eigen::Vector3d Pixel2Camera(const cv::Point2f &point, const float &depth)
    {
        return Eigen::Vector3d( depth*(point.x - mcx)/mfx,
                                depth*(point.y - mcy)/mfy,
                                depth);
    }

    static Eigen::Vector2d Camera2Pixel(const Eigen::Vector3d &Point)
    {
        return Eigen::Vector2d( mfx*Point[0]/Point[2] + mcx,
                                mfy*Point[1]/Point[2] + mcy);
    }


public:

    constexpr static const float mfx = 315.5;
    constexpr static const float mfy = 315.5;
    constexpr static const float mcx = 376.0;
    constexpr static const float mcy = 240.0;
    constexpr static const float mf = 315.5;

    Sophus::SE3     mPose;

    std::vector<cv::Mat> Img_Pyr;
    std::vector<cv::Point2f> mvKps;
    std::vector<Eigen::Vector3d> mvMapPoints;
    std::vector<Eigen::Vector3d> mvNormals;
};

typedef std::shared_ptr<Frame> FramePtr;

class Sprase_ImgAlign
{
public:
    Sprase_ImgAlign(int tMaxLevel, int tMinLevel, int tMaxIterators);
    ~Sprase_ImgAlign();

    //! Reset
    void Reset();

    int Run(FramePtr tCurFrame, FramePtr tRefFrame);

    //! Compute the jacobian matrix and  reference patch
    void GetJocabianMat(int tLevel);

    //! Compute the jacobian about pixel position to camera transform
    Eigen::Matrix<double, 2,6> GetJocabianBA(Eigen::Vector3d tPoint);

    //! Solve transform with ceres
    void DirectSolver(Sophus::SE3 &tT_c2r, int tLevel);

protected:
    int     mnMaxLevel;
    int     mnMinLevel;
    int     mnMaxIterators;


    const int       mBoarder;
    const int       mPatchArea;

    FramePtr    mCurFrame;
    FramePtr    mRefFrame;

    Sophus::SE3     mT_c2r;

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> mRefPatch;
    Eigen::Matrix<double, Eigen::Dynamic, 6, Eigen::RowMajor> mJocabianPatch;
    Eigen::Matrix<double, 3, Eigen::Dynamic, Eigen::ColMajor> mRefNormals;

    std::vector<bool> mVisible;
};

class DirectSE3_Problem: public ceres::SizedCostFunction<mHalf_PatchSize*mHalf_PatchSize, 6>
{
public:
    DirectSE3_Problem(Eigen::Vector3d tPoint, double *tRefPatch, double *tJocabianPatch,
                      const cv::Mat *tCurImg, const float tScale):
            mMapPoint(tPoint), mRefPatch(tRefPatch), mJacobianPatch(tJocabianPatch),
            mCurImg(tCurImg), mScale(tScale), mnboarder(mHalf_PatchSize-1)
    {

    }

    virtual bool Evaluate(double const* const* parameters, double *residuals, double **jacobians) const
    {
        // parameter: translation--rotation
        Eigen::Matrix<double, 6, 1> tT_c2rArray;
        tT_c2rArray<< parameters[0][0], parameters[0][1], parameters[0][2],
                parameters[0][3], parameters[0][4], parameters[0][5];

        Sophus::SE3 tT_c2r(Sophus::SO3::exp(tT_c2rArray.tail<3>()), tT_c2rArray.head<3>());

        Eigen::Vector3d tCurPoint = tT_c2r*mMapPoint;
        Eigen::Vector2d tCurPix = Frame::Camera2Pixel(tCurPoint)*mScale;

        const double u = tCurPix(0);
        const double v = tCurPix(1);
        const int u_i = floor(u);
        const int v_i = floor(v);
        if(u_i < 0 || v_i < 0 || u_i - mnboarder < 0 || v_i - mnboarder < 0 || u_i + mnboarder >= mCurImg->cols || v_i + mnboarder >= mCurImg->rows)
        {
            Eigen::Map<Eigen::Matrix<double, mHalf_PatchSize*mHalf_PatchSize, 1>> mResiduals(residuals);
            mResiduals.setZero();

            if(jacobians != NULL && jacobians[0] != NULL)
            {
                Eigen::Map<Eigen::Matrix<double, mHalf_PatchSize * mHalf_PatchSize, 6, Eigen::RowMajor>> mJacobians(jacobians[0]);
                mJacobians.setZero();
            }

            return true;
        }

        const double tSubPixU = u - u_i;
        const double tSubPixV = v - v_i;
        const double tC_tl = (1.0 - tSubPixU)*(1.0 - tSubPixV);
        const double tC_tr = tSubPixU*(1.0 - tSubPixV);
        const double tC_bl = (1.0 - tSubPixU)*tSubPixV;
        const double tC_br = tSubPixU*tSubPixV;

        int tStep = mCurImg->step.p[0];
        int tNum = 0;

        for (int i = 0; i < mHalf_PatchSize; ++i)
        {
            uchar *it = mCurImg->data + (v_i + i - 2)*mCurImg->cols + u_i - 2;
            for (int j = 0; j < mHalf_PatchSize; ++j, ++it, ++tNum)
            {
                double tCurPx = tC_tl*it[0] + tC_tr*it[1] + tC_bl*it[tStep] + tC_br*it[tStep+1];
                residuals[tNum] = *(mRefPatch+tNum) - tCurPx;
            }
        }

        if(jacobians!=NULL)
        {
            if (jacobians[0] != NULL)
            {
                Eigen::Map<Eigen::Matrix<double, mHalf_PatchSize * mHalf_PatchSize, 6, Eigen::RowMajor>> mJacobians(jacobians[0]);
                Eigen::Matrix<double, mHalf_PatchSize * mHalf_PatchSize, 6, Eigen::RowMajor> tJacobians(mJacobianPatch);

                mJacobians = tJacobians;
            }
        }

        return true;
    }


protected:
    Eigen::Vector3d     mMapPoint;

    const cv::Mat       *mCurImg;
    const float         mScale;

    double              *mRefPatch;
    double              *mJacobianPatch;

    const int           mnboarder;
};

class PoseLocalParameterization : public ceres::LocalParameterization
{

    virtual bool Plus(const double *T_raw, const double *delta_raw, double *T_plus_delta_raw) const
    {
        Eigen::Map<const Sophus::Vector6d> tOld(T_raw);
        Eigen::Map<const Sophus::Vector6d> tDelta(delta_raw);
        Eigen::Map<Sophus::Vector6d> tNew(T_plus_delta_raw);

        Sophus::SE3 tT_Old(Sophus::SO3::exp(tOld.tail<3>()), tOld.head<3>());
        Sophus::SE3 tT_Delta(Sophus::SO3::exp(tDelta.tail<3>()), tDelta.head<3>());
        Sophus::SE3 tT_New = tT_Old*tT_Delta;

        tNew.block(0, 0, 3, 1) = tT_New.translation();
        tNew.block(3, 0, 3, 1) = tT_New.so3().log();

        return true;
    }

    virtual bool ComputeJacobian(const double *T_raw, double *jacobian_raw) const
    {
        Eigen::Map<Eigen::Matrix<double, 6, 6, Eigen::RowMajor> > J(jacobian_raw);
        J.setIdentity();

        return true;
    }

    virtual int GlobalSize() const { return 6; }

    virtual int LocalSize() const { return 6; }

};

static int Eigenfloor(double x)
{
    return floor(x);
}

}// namespace CV_MVG


#endif //CV_MVG_SPRASE_IMAGEALIGN_H
