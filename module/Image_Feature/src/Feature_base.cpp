//
// Created by buyi on 17-10-6.
//

#include "Feature_base.h"


namespace CV_MVG
{
    Feature_base::Feature_base()
    {

    }

    Feature_base::~Feature_base()
    {

    }

    // Extract the keypoints whose match Hamming distance less than 30
    void Feature_base::Extract_GoodPoints(cv::Mat imageA, cv::Mat imageB, vector<cv::Point2f> &PointsA,
                                          vector<cv::Point2f> &PointsB)
    {
        // extract keypoints and descriptors
        vector<cv::KeyPoint> KeyPointA, KeyPointB;
        cv::ORB orb;

        orb.detect(imageA, KeyPointA);
        orb.detect(imageB, KeyPointB);

        cv::Mat DescriptorsA, DescriptorsB;
        orb.compute(imageA, KeyPointA, DescriptorsA);
        orb.compute(imageB, KeyPointB, DescriptorsB);

        // match the keypoints
        cv::BFMatcher matcher(cv::NORM_HAMMING);
        vector<cv::DMatch> matches, Good_matches;
        matcher.match(DescriptorsA, DescriptorsB, matches);

        // select the keypoints have a smaller distance
        double mini_dist = 1000, max_dist = 0;
        for (int i = 0; i < DescriptorsA.rows ; ++i)
        {
            double dist = matches[i].distance;
            if(dist < mini_dist)  mini_dist = dist;
            if(dist > max_dist)     max_dist =dist;
        }

        for (int j = 0; j < DescriptorsA.rows; ++j)
        {
            if(matches[j].distance <= max(2*mini_dist, 30.0))
                Good_matches.push_back(matches[j]);
        }

        vector<int> PointIndexA, PointIndexB;
        for (int i = 0; i < Good_matches.size(); ++i)
        {
            cv::DMatch it = Good_matches[i];
            PointIndexA.push_back(it.queryIdx);
            PointIndexB.push_back(it.trainIdx);
        }

        cv::KeyPoint::convert(KeyPointA, PointsA, PointIndexA);
        cv::KeyPoint::convert(KeyPointB, PointsB, PointIndexB);
    }
} //
