#ifndef HEADPOS_C_H_
#define HEADPOS_C_H_
#include<ctime>
#include<chrono>
#endif //HEADPOS_C_H_
#ifndef HEADPOS_CXX_H_
#define HEADPOS_CXX_H_
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#endif //HEADPOS_CXX_H_
#ifndef HEADPOS_OPENCV_H_
#define HEADPOS_OPENCV_H_
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#endif //HEADPOS_OPENCV_H_
#ifndef HEADPOS_PCL_H_
#define HEADPOS_PCL_H_
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/io/io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/normal_3d.h>
#include <pcl/surface/gp3.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <boost/thread/thread.hpp>
#endif //HEADPOS_PCL_H_
#ifndef HEADPOS_DLIB_H_
#define HEADPOS_DLIB_H_
#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#endif //HEADPOS_DLIB_H_

typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloud;

std::string shapemodel="/home/lhw/uisee/face-2/shape_predictor_68_face_landmarks.dat";
std::string videoPath="/home/lhw/Videos/2019-12-30-095942.webm";

cv::Mat cameramatrix=(cv::Mat_<double>(3,3)<< 806.2888,0.000000,629.656112,
                                                         0.000000,807.0587,380.321045,
                                                         0.000000,0.000000,1.000000);
cv::Mat distortion = (cv::Mat_<double>(5,1)<< -0.052928,-0.004169,0.001651,0.000625,0.000000);

/**
 * @brief Read 3d face points from .txt to a vector variable.
 * @param filename : file path,contain a set of 3d face points from 3dmm model.
 * @param general_face_shape_points : a vector ,the output of this function
*/
void ReadGeneralFaceShapePoints(const std::string filename,
                                std::vector<double>& general_face_shape_points);
/**
 * @brief Read 3d face keypoints' index from .txt to a vector variable.
 * @param filename : file path,contain a set of 3d face keypoints' index.
 * @param face_keypoints_index : a vector , the output of this function.
 */
void ReadGeneralFaceShapePointsIndex(const std::string filename,
                                     std::vector<int>& face_keypoints_index);
/**
 * @brief normalize 3d face points,and transfer the type(vector) of general_face_shape_points into another type(MatrixXd)
 * @param general_face_shape_points : a vector of raw general 3d face points.
 * @param general_face_shape_points_matrix : a matrixXd of normalized general 3d face points.
 * @note general_face_shape_points_matrix need to assign numbers of cols and rows.
 */
void NormalizeFacePoints(const std::vector<double>& general_face_shape_points,
                         cv::Mat & general_face_shape_points_matrix);
/**
 * @brief decompose essential matrix
 * @param essential_matrix : essential matrix.
 * @param R1 : the output of one of rotation matrix .(there are two different solutions)
 * @param R2 : the output of the other rotation matrix .
 * @param t1 : the output of one of transition vector .
 * @param t2 : the output of the other transition vector .
 */
void DecomposeEssentialMatrix(const cv::Mat essential_matrix,
                              cv::Mat_<double> &R1, cv::Mat_<double> &R2,
                              cv::Mat_<double> &t1, cv::Mat_<double> &t2);
/**
 * @brief using triangulation algorithom, judge this rotation matrix R and transition vector t whether get to positive depth.
 * @param lastframe : last frame
 * @param curframe : current frame
 * @param R : rotation matrix
 * @param t : transition vector
 * @return the ratio of positive depth.
 */
double TestTriangulation(const std::vector<cv::Point2d>& lastframe,
                         const std::vector<cv::Point2d>& curframe,
                         cv::Mat_<double> R, cv::Mat_<double> t);
/**
 * @brief detect 68 face keypoints frome image.
 * @param inputimage : the input of image
 * @param outputimage : the output of image
 * @param detector : dlib's face's detector.
 * @param pose_model : dlib's keypoints' model.
 * @param face_detect_keypoints : the output of detected face keypoints.
 * @note inputimage and outputimage must be different.
 */
void FaceDetection(cv::Mat& inputimage,
                   cv::Mat& outputimage,
                   dlib::frontal_face_detector& detector,
                   dlib::shape_predictor& pose_model,
                   std::vector<cv::Point2d>& face_detect_keypoints);

/**
 * @brief use Epipolar geometry algorithom to solve R and t between two frame.
 * @param lastframe : last frame
 * @param curframe : current frame
 * @return 4x4 transform matrix (R,T)
 */
 /**
  * @brief use Epipolar geometry algorithom to solve R and t between two frame.
  * @param face_keypoint_init
  * @param face_keypoint_cur
  * @param transform_matrix
  * @return 4x4 transform matrix (R,T)
  */
void  SolveRT(const std::vector<cv::Point2d>& face_keypoint_init,//需要提取关键点
                const std::vector<cv::Point2d>& face_keypoint_cur,
                cv::Mat& transform_matrix);
/**
 * @brief transfer from general face points matrix to current pose .
 * @param transform_matrix : transform matrix
 * @param general_face_shape_points_matrix : a matrixXd of normalized general 3d face points.
 */
void TransformGeneralFacePoints(cv::Mat& transform_matrix,
                                cv::Mat& general_face_shape_points_matrix);


class pointcloud{
 public:
    pointcloud(){
        cloud = PointCloud::Ptr(new PointCloud);
    }
    void AddPointsToCloud(const cv::Mat& face_3d_final){
        for(int i=0;i<face_3d_final.rows;i++)//Xx3matrix
        {
            cloud->points.emplace_back(face_3d_final.at<double>(i,0),
                                       face_3d_final.at<double>(i,1),
                                       face_3d_final.at<double>(i,2));
        }
        cloud->height = 1;
        cloud->width = cloud->points.size();
        cloud->is_dense = false;
    }
    PointCloud::Ptr GetCloud(){
        return cloud;
    }
    void ClearPointCloud()
    {
        cloud->points.clear();
    }
private:
    PointCloud::Ptr cloud;
};

class viewer
{
public:
    viewer(pointcloud& cloud_){
        cloud = cloud_;
        view = boost::shared_ptr<pcl::visualization::PCLVisualizer>(new pcl::visualization::PCLVisualizer ("3D Viewer"));
    }
    void AddPointCloud(){
        view->addPointCloud(cloud.GetCloud());
    }
    void UpdatePointCloud(int second){
        view->updatePointCloud<pcl::PointXYZ>(cloud.GetCloud());
        view->spinOnce(second);
    }
private:
    pointcloud cloud;
    boost::shared_ptr<pcl::visualization::PCLVisualizer> view;
};