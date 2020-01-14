#ifndef HEADPOS_ESTIMATION_H_
#define HEADPOS_ESTIMATION_H_
#include "head_pose_estimation.h"
#endif //HEADPOS_ESTIMATION_H_
void ReadGeneralFaceShapePoints(const std::string filename,
                                std::vector<double>& general_face_shape_points){
    std::string line;
    std::ifstream it(filename);
    std::istringstream is(line);
    while(it>>line)
    {
        general_face_shape_points.push_back(stod(line));
    }
}
void ReadGeneralFaceShapePointsIndex(const std::string filename,
                                     std::vector<int>& face_keypoints_index){
    std::string line;
    std::ifstream in(filename);
    while(in>>line)
    {
        face_keypoints_index.push_back(stoi(line));
    }
}
void NormalizeFacePoints(const std::vector<double>& general_face_shape_points,
                         cv::Mat& general_face_shape_points_matrix){
    for(int i=0;i<general_face_shape_points.size()/3;++i)
    {
        general_face_shape_points_matrix.at<double>(i,0)=general_face_shape_points[3*i]/100.0;
        general_face_shape_points_matrix.at<double>(i,1)=general_face_shape_points[3*i+1]/100.0;
        general_face_shape_points_matrix.at<double>(i,2)=general_face_shape_points[3*i+2]/100.0;
    }
    double minx,maxx,miny,maxy,minz,maxz;
    cv::minMaxIdx(general_face_shape_points_matrix.col(0), &minx, &maxx);
    cv::minMaxIdx(general_face_shape_points_matrix.col(1), &miny, &maxy);
    cv::minMaxIdx(general_face_shape_points_matrix.col(2), &minz, &maxz);
    cv::Mat minvector=(cv::Mat_<double>(1,3)<<minx,miny,minz);
    cv::Mat maxvector=(cv::Mat_<double>(1,3)<<maxx,maxy,maxz);
    cv::Mat midvector=(minvector+maxvector)/2.0;
    for(int i=0;i<general_face_shape_points.size()/3;++i)
    {
        general_face_shape_points_matrix.row(i)-=midvector;
    }
}
void DecomposeEssentialMatrix(const cv::Mat essential_matrix,
                              cv::Mat_<double> &R1, cv::Mat_<double> &R2,
                              cv::Mat_<double> &t1, cv::Mat_<double> &t2){
    cv::SVD svd(essential_matrix, cv::SVD::MODIFY_A);
    cv::Matx33d W(0, -1, 0,
                  1, 0, 0,
                  0, 0, 1);
    cv::Matx33d Wt(0, 1, 0,
                   -1, 0, 0,
                   0, 0, 1);
    R1 = svd.u * cv::Mat(W) * svd.vt;
    R2 = svd.u * cv::Mat(Wt) * svd.vt;
    t1 = svd.u.col(2);
    t2 = -svd.u.col(2);
}

double TestTriangulation(const std::vector<cv::Point2d>& lastframe,
                         const std::vector<cv::Point2d>& curframe,
                         cv::Mat_<double> R, cv::Mat_<double> t){
    cv::Mat point;
    cv::Matx34d P = cv::Matx34d(1, 0, 0, 0,
                                0, 1, 0, 0,
                                0, 0, 1, 0);
    cv::Matx34d P1 = cv::Matx34d(R(0, 0), R(0, 1), R(0, 2), t(0),
                                 R(1, 0), R(1, 1), R(1, 2), t(1),
                                 R(2, 0), R(2, 1), R(2, 2), t(2));
    cv::triangulatePoints(P, P1, lastframe, curframe, point);
    int front_count = 0;
    for (int i = 0; i < point.cols; i++)
    {
        double normal_factor = point.col(i).at<float>(3);

        cv::Mat_<double> p_3d_l = cv::Mat(P) * (point.col(i) / normal_factor);
        cv::Mat_<double> p_3d_r = cv::Mat(P1) * (point.col(i) / normal_factor);
        if (p_3d_l(2) > 0 && p_3d_r(2) > 0)
            front_count++;
    }
    return 1.0 * front_count / point.cols;//返回深度为正的点数百分比
}

void FaceDetection(cv::Mat& inputimage,
                   cv::Mat& outputimage,
                   dlib::frontal_face_detector& detector,
                   dlib::shape_predictor& pose_model,
                   std::vector<cv::Point2d>& face_detect_keypoints){
    cv::undistort(inputimage,outputimage,cameramatrix,distortion);
    dlib::cv_image<dlib::bgr_pixel> cimg(inputimage);
    std::vector<dlib::rectangle> faces = detector(cimg);
    std::vector<dlib::full_object_detection> shapes;
    for (unsigned long i = 0; i < faces.size(); ++i)
        shapes.push_back(pose_model(cimg, faces[i]));
    cv::Point2d t;
    if (!shapes.empty()) {
        for (int i = 0; i < 31; i++) {
            t.x=double(shapes[0].part(i).x());
            t.y=double(shapes[0].part(i).y());
            face_detect_keypoints.push_back(t);
            cv::circle(outputimage, t,  2,  cv::Scalar(0,0,255),2);

        }
    }
}
void SolveRT(const std::vector<cv::Point2d>& face_keypoint_init,//需要提取关键点
             const std::vector<cv::Point2d>& face_keypoint_cur,
             cv::Mat& transform_matrix){
    if(face_keypoint_cur.empty()||face_keypoint_init.empty()){
        transform_matrix=cv::Mat::eye(4,4,CV_64F);
        return;
    }
    cv::Mat essential_matrix(3,3,CV_64F), R_(3,3,CV_64F),R(3,3,CV_64F),t(3,1,CV_64F),t_(3,1,CV_64F);
    if(face_keypoint_init == face_keypoint_cur){
        transform_matrix=cv::Mat::eye(4,4,CV_64F);
        return;
    }
    if(face_keypoint_init.size()!=face_keypoint_cur.size()){
        transform_matrix=cv::Mat::eye(4,4,CV_64F);
        return;
    }
    essential_matrix = cv::findEssentialMat(face_keypoint_init,face_keypoint_cur,cameramatrix,cv::RANSAC,0.999,1);
    cv::recoverPose(essential_matrix,face_keypoint_init,face_keypoint_cur,R,t);
//    cv::transpose(R_,R);
//    t = -R_ * t_;
    transform_matrix=(cv::Mat_<double>(4,4)<<R.at<double>(0,0),R.at<double>(0,1),R.at<double>(0,2),t.at<double>(0,0),
                                                 R.at<double>(1,0),R.at<double>(1,1),R.at<double>(1,2),t.at<double>(1,0),
                                                 R.at<double>(2,0),R.at<double>(2,1),R.at<double>(2,2),t.at<double>(2,0),
                                                 0,0,1);
}
void TransformGeneralFacePoints(const cv::Mat& transform_matrix,
                                const cv::Mat& general_face_shape_points_matrix,
                                cv::Mat& general_face_shape_points_matrix_){//Xx4matrix
        for (int i=0; i<general_face_shape_points_matrix_.rows; i++){
            general_face_shape_points_matrix_.at<double>(i,0)=transform_matrix.at<double>(0,0)*
                                                                 general_face_shape_points_matrix.at<double>(i,0)+
                                                                 transform_matrix.at<double>(0,1)*
                                                                 general_face_shape_points_matrix.at<double>(i,1)+
                                                                 transform_matrix.at<double>(0,2)*
                                                                 general_face_shape_points_matrix.at<double>(i,2)+
                                                                 transform_matrix.at<double>(0,3);
            general_face_shape_points_matrix_.at<double>(i,1)=transform_matrix.at<double>(1,0)*
                                                                 general_face_shape_points_matrix.at<double>(i,0)+
                                                                 transform_matrix.at<double>(1,1)*
                                                                 general_face_shape_points_matrix.at<double>(i,1)+
                                                                 transform_matrix.at<double>(1,2)*
                                                                 general_face_shape_points_matrix.at<double>(i,2)+
                                                                 transform_matrix.at<double>(1,3);
            general_face_shape_points_matrix_.at<double>(i,2)=transform_matrix.at<double>(2,0)*
                                                                 general_face_shape_points_matrix.at<double>(i,0)+
                                                                 transform_matrix.at<double>(2,1)*
                                                                 general_face_shape_points_matrix.at<double>(i,1)+
                                                                 transform_matrix.at<double>(2,2)*
                                                                 general_face_shape_points_matrix.at<double>(i,2)+
                                                                 transform_matrix.at<double>(2,3);
        }
}
int main() {
    std::string filename_shape("/home/lhw/uisee/face-2d-2d_final/shapeMU.txt"), filename_index("/home/lhw/uisee/face-2d-2d_final/3DMM(1-31).txt");
    std::vector<double> general_face_shape_points;
    std::vector<int> face_keypoints_index;
    //1.read 3d points and keypoints' index from .txt
    ReadGeneralFaceShapePoints(filename_shape, general_face_shape_points);
    ReadGeneralFaceShapePointsIndex(filename_index, face_keypoints_index);
    //2.3d points normalization
    cv::Mat general_face_shape_points_matrix(cv::Size(3,general_face_shape_points.size()/3),CV_64F);
    NormalizeFacePoints(general_face_shape_points, general_face_shape_points_matrix);
    //3.grab image
    cv::VideoCapture cap = cv::VideoCapture(videoPath);//or index
    if (!cap.isOpened()) {
        cout << "no video!" << endl;
    }
    
    //4.define face detection
    dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
    dlib::shape_predictor pose_model;
    dlib::deserialize(shapemodel) >> pose_model;
    //5.define viewer and point cloud
    pointcloud cloud;
    viewer view(cloud);
    view.AddPointCloud();
    cv::Mat initframe_, initframe, curframe_, curframe;
    std::vector<cv::Point2d> face_keypoint_init;
    std::vector<cv::Point2d> face_keypoint_cur;
    int count(0), cycle(1);
    while (1){
        cap >> curframe_;
        if (count / (20 * cycle) < 1) {
            ++count;
            continue;
        }
        if(curframe_.empty())
            break;
        //6.face detection
        std::vector<cv::Point2d> face_detect_keypoints;
        FaceDetection(curframe_, curframe, detector, pose_model, face_detect_keypoints);
        if (initframe.empty()) {
            face_keypoint_init = face_detect_keypoints;
            initframe=curframe.clone();
            continue;
        }
        face_keypoint_cur = face_detect_keypoints;
        //7.solve transform pose
        cv::Mat transform_matrix;
        SolveRT(face_keypoint_init, face_keypoint_cur,transform_matrix);
        //8.transfer 3d points
        cv::Mat general_face_shape_points_matrix_(cv::Size(3,general_face_shape_points.size()/3),CV_64F);
        TransformGeneralFacePoints(transform_matrix,
                                   general_face_shape_points_matrix,
                                   general_face_shape_points_matrix_);
        //9.add 3d points to pointcloud
        cloud.AddPointsToCloud(general_face_shape_points_matrix_);
        //10.show pointcloud
        view.UpdatePointCloud(10);
        //* 测试图片
        cv::imshow("face", curframe);
        cv::waitKey(10);
        face_keypoint_init = face_keypoint_cur;
        face_keypoint_cur.clear();
        general_face_shape_points_matrix_.release();
        cloud.ClearPointCloud();
        ++count;
        ++cycle;

    }
}