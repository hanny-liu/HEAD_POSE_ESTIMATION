# HEAD_POSE_ESTIMATION
driver's head pose estimation
# 驾驶员头部姿态说明文档
<<<<<<< HEAD
## 驾驶员头部姿态估计算法说明
驾驶员头部姿态估计主要分为驾驶员头部关键点提取，驾驶员头部姿态估计，驾驶员头部3d显示。如下图所示：
1. 驾驶员头部关键点提取  
主要利用dlib库的检测器和训练模型提取人脸68个关键点
2. 驾驶员头部姿态估计  
主要利用2d-2d的对极几何的估计相机相邻两帧的变换情况
3. 驾驶员人脸3d显示  
使用PCL库中3d点云三维显示。
##依赖库安装
主要需要下载pcl,opencv4.,dlib.
1. 安装dlib库：  
cd dlib  
mkdir build  
cmake .. -DUSE_AVX_INSTRUCTION=ON -DUSE_SSE4_INSTRUCTIONS=ON  
cmake --build . --config Release --target install  
sudo make install
2. 安装pcl库：  
sudo add-apt-repository ppa:v-launchpad-jochen-sprickerhof-de/pcl  
sudo apt-get update  
sudo apt-get install libpcl-all
3. 安装opencv  
(1).安装依赖库  
sudo apt-get install build-essential libgtk2.0-dev libvtk5-dev libjpeg-dev libtiff5-dev libjasper-dev libopenexr-dev libtbb-dev  
(2).下载源码  
https://opencv.org/releases/  
(3).编译源码  
cmake ..  
make -j4  
sudo make install  
然后再使用release模式运行程序，程序就能加速运行
##驾驶员头部姿态估计代码说明
1. 读入3d人脸数据与处理  
* 读取标准人脸3d特征点：  
Void ReadGeneralFaceShapePoints(const std::string filename, std::vector<double>& general_face_shape_points);  
传入的是存放人脸特征点的文件，该文件以（x1,y1,z1,x2,y2,z2,…,xn,yn,zn）T的格式保存  
参数general_face_shape_points是用来以vector格式存放3d特征点的变量作为输出。
* 读取标准人脸68个特征点的索引：
Void ReadGeneralFaceShapePointsIndex(const std::string filename, 
std::vector<int>& face_keypoints_index);  
传入两个参数 ：一个是存放人脸68个关键点索引的文件，以（1，2，3，…）T列向量保存。另一个是以vector格式存放68个关键点索引的变量作为输出。
* 数据归一化：  
NormalizeFacePoints(const std::vector<double>& general_face_shape_points, cv::Mat & general_face_shape_points_matrix);  
数据归一化是因为3d人脸的数据的量纲较大，一开始使用Pnp算法估计头部姿态时，为了估计较准确减少跳动需要对数据进行归一化。目前对极几何算法可以不用数据归一化。
2. 分解本质矩阵，求R，t  
Void DecomposeEssentialMatrix(const cv::Mat essential_matrix, cv::Mat_<double> &R1, cv::Mat_<double> &R2, cv::Mat_<double> &t1, cv::Mat_<double> &t2);  
传入的参数：essential_matrix是求出的本质矩阵，R1，R2，t1，t2是分解出的四个解。
3. 利用三角化筛选R，t  
double TestTriangulation(const std::vector<cv::Point2d>& lastframe, const std::vector<cv::Point2d>& curframe, cv::Mat_<double> R, cv::Mat_<double> t);  
参数：lastframe为上一帧检测的人脸关键点，curframe为当前帧检测的人脸关键点，R，t为求解出的最终姿态。  
返回：正深度占比
4. 人脸关键点检测  
void FaceDetection(cv::Mat& inputimage, cv::Mat& outputimage, 
dlib::frontal_face_detector& detector, dlib::shape_predictor& pose_model,
std ::vector<cv::Point2d>& face_detect_keypoints);  
参数：inputimage为输入的图片，outputimage为输出的图片，detector为dlib库中的检测器，pose_model为输入的人脸训练模型，face_detect_keypoints为输出的人脸检测的关键点。
5. 求解本质矩阵  
void  SolveRT(const std::vector<cv::Point2d>& face_keypoint_init, const 
std::vector<cv::Point2d>& face_keypoint_cur, cv::Mat& transform_matrix);  
参数：face_keypoint_init为前一帧识别的关键点，face_keypoint_cur为后一帧识别的关键点，transform_matrix为返回的变换矩阵。
6. 变换3d人脸
void TransformGeneralFacePoints(cv::Mat& transform_matrix,
cv::Mat& general_face_shape_points_matrix);
参数：transform_matrix为传入的变换矩阵，general_face_shape_points_matrix为返回的经变化后的3d人脸特征点。
7. 3d人脸显示  
点云类pointcloud，只需在代码中定义一个点云就可以，在构造函数中会自动为其生成用于显示的点云。  
成员函数：  
void AddPointsToCloud(const cv::Mat& face_3d_final)：通过传入最终变换后的3d人脸特征点，加入到点云变量中。  
PointCloud::Ptr GetCloud()：返回点云变量，主要用于显示的初始化  
void ClearPointCloud()：清除点云中的点。  
显示类viewer，构造函数需要其传入点云变量，以初始化显示类的成员变量。  
void AddPointCloud()：添加点云  
void UpdatePointCloud(int second)：更新点云，传入停留点云的秒数。
