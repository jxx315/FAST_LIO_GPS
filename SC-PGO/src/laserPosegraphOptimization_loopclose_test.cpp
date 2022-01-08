#include <fstream>
#include <math.h>
#include <vector>
#include <mutex>
#include <queue>
#include <thread>
#include <iostream>
#include <string>
#include <optional>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/impl/search.hpp>
#include <pcl/range_image/range_image.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/registration/icp.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/octree/octree_pointcloud_voxelcentroid.h>
#include <pcl/filters/crop_box.h> 
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/registration/ndt.h>      //NDT(正态分布)配准类头文件
#include <pcl/filters/approximate_voxel_grid.h>   //滤波类头文件  （使用体素网格过滤器处理的效果比较好）

#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/NavSatFix.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>

#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>


#include <eigen3/Eigen/Dense>

#include <ceres/ceres.h>

#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot2.h>
#include <gtsam/geometry/Pose2.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/ISAM2.h>

#include "aloam_velodyne/common.h"
#include "aloam_velodyne/tic_toc.h"

#include "scancontext/Scancontext.h"
#include <aloam_velodyne/SaveMap.h>

using namespace gtsam;

using std::cout;
using std::endl;



double ndt_resolution =0;
double ndt_stepSize=0;
double loopFitnessScoreThreshold = 0;
double loopCutRange=0;
int historyKeyframeSearchNum_curr=0;
int historyKeyframeSearchNum_map=0;


double keyframeMeterGap;
double keyframeDegGap, keyframeRadGap;
double translationAccumulated = 1000000.0; // large value means must add the first given frame.
double rotaionAccumulated = 1000000.0; // large value means must add the first given frame.

bool isNowKeyFrame = false; 

Pose6D odom_pose_prev {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}; // init 
Pose6D odom_pose_curr {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}; // init pose is zero 

std::queue<nav_msgs::Odometry::ConstPtr> odometryBuf;
std::queue<sensor_msgs::PointCloud2ConstPtr> fullResBuf;
std::queue<sensor_msgs::NavSatFix::ConstPtr> gpsBuf;
std::queue<std::pair<int, int> > scLoopICPBuf;

std::mutex mBuf;
std::mutex mKF;

double timeLaserOdometry = 0.0;
std_msgs::Header timeLaserOdometr_header;
double timeLaser = 0.0;

pcl::PointCloud<PointType>::Ptr laserCloudFullRes(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr laserCloudMapAfterPGO(new pcl::PointCloud<PointType>());

std::vector<pcl::PointCloud<PointType>::Ptr> keyframeLaserClouds; 
std::vector<Pose6D> keyframePoses;
//std::vector<Pose6D> copy_keyframePoses;

std::vector<Pose6D> keyframePosesUpdated;

std::vector<double> keyframeTimes;
std::vector<double> copy_keyframeTimes;
int recentIdxUpdated = 0;



bool aLoopIsClosed = false;
map<int, int> loopIndexContainer; // from new to old
vector<pair<int, int>> loopIndexQueue;
vector<gtsam::Pose3> loopPoseQueue;
// vector<gtsam::noiseModel::Diagonal::shared_ptr> loopNoiseQueue;
vector<gtsam::noiseModel::Robust::shared_ptr> loopNoiseQueue;


pcl::PointCloud<PointType>::Ptr cloudKeyPoses3D;
pcl::PointCloud<PointType>::Ptr copy_cloudKeyPoses3D;

pcl::KdTreeFLANN<PointType>::Ptr kdtreeHistoryKeyPoses;
bool detectLoopClosureDistance(int *latestID, int *closestID);
void visualizeLoopClosure();

gtsam::NonlinearFactorGraph gtSAMgraph;
bool gtSAMgraphMade = false;
gtsam::Values initialEstimate;
gtsam::ISAM2 *isam;
gtsam::Values isamCurrentEstimate;

noiseModel::Diagonal::shared_ptr priorNoise;
noiseModel::Diagonal::shared_ptr odomNoise;
noiseModel::Base::shared_ptr robustLoopNoise;
noiseModel::Base::shared_ptr robustGPSNoise;

pcl::VoxelGrid<PointType> downSizeFilterScancontext;
SCManager scManager;
double scDistThres, scMaximumRadius;

pcl::VoxelGrid<PointType> downSizeFilterICP;
std::mutex mtxICP;
std::mutex mtxPosegraph;
std::mutex mtxRecentPose;

pcl::PointCloud<PointType>::Ptr laserCloudMapPGO(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr laserCloudMapPGO_SavePcd(new pcl::PointCloud<PointType>());
pcl::VoxelGrid<PointType> downSizeFilterMapPGO;
bool laserCloudMapPGORedraw = true;

bool useGPS = true;
// bool useGPS = false;
sensor_msgs::NavSatFix::ConstPtr currGPS;
bool hasGPSforThisKF = false;
bool gpsOffsetInitialized = false; 
double gpsAltitudeInitOffset = 0.0;
double recentOptimizedX = 0.0;
double recentOptimizedY = 0.0;

ros::Publisher pubMapAftPGO, pubOdomAftPGO, pubPathAftPGO;
ros::Publisher pubLoopScanLocal, pubLoopSubmapLocal;
ros::Publisher pubOdomRepubVerifier;
ros::Publisher pubLoopConstraintEdge;

std::string save_directory;
std::string pgKITTIformat, pgScansDirectory;
std::string odomKITTIformat;
std::fstream pgTimeSaveStream;

std::string padZeros(int val, int num_digits = 6) {
  std::ostringstream out;
  out << std::internal << std::setfill('0') << std::setw(num_digits) << val;
  return out.str();
}

gtsam::Pose3 Pose6DtoGTSAMPose3(const Pose6D& p)
{
    return gtsam::Pose3( gtsam::Rot3::RzRyRx(p.roll, p.pitch, p.yaw), gtsam::Point3(p.x, p.y, p.z) );
} // Pose6DtoGTSAMPose3

void saveOdometryVerticesKITTIformat(std::string _filename)
{
    // ref from gtsam's original code "dataset.cpp"
    std::fstream stream(_filename.c_str(), std::fstream::out);
    for(const auto& _pose6d: keyframePoses) {
        gtsam::Pose3 pose = Pose6DtoGTSAMPose3(_pose6d);
        Point3 t = pose.translation();
        Rot3 R = pose.rotation();
        auto col1 = R.column(1); // Point3
        auto col2 = R.column(2); // Point3
        auto col3 = R.column(3); // Point3

        stream << col1.x() << " " << col2.x() << " " << col3.x() << " " << t.x() << " "
               << col1.y() << " " << col2.y() << " " << col3.y() << " " << t.y() << " "
               << col1.z() << " " << col2.z() << " " << col3.z() << " " << t.z() << std::endl;
    }
}

void saveOptimizedVerticesKITTIformat(gtsam::Values _estimates, std::string _filename)
{
    using namespace gtsam;

    // ref from gtsam's original code "dataset.cpp"
    std::fstream stream(_filename.c_str(), std::fstream::out);

    for(const auto& key_value: _estimates) {
        auto p = dynamic_cast<const GenericValue<Pose3>*>(&key_value.value);
        if (!p) continue;

        const Pose3& pose = p->value();

        Point3 t = pose.translation();
        Rot3 R = pose.rotation();
        auto col1 = R.column(1); // Point3
        auto col2 = R.column(2); // Point3
        auto col3 = R.column(3); // Point3

        stream << col1.x() << " " << col2.x() << " " << col3.x() << " " << t.x() << " "
               << col1.y() << " " << col2.y() << " " << col3.y() << " " << t.y() << " "
               << col1.z() << " " << col2.z() << " " << col3.z() << " " << t.z() << std::endl;
    }
}

void laserOdometryHandler(const nav_msgs::Odometry::ConstPtr &_laserOdometry)
{
	mBuf.lock();
	odometryBuf.push(_laserOdometry);
	mBuf.unlock();
} // laserOdometryHandler

void laserCloudFullResHandler(const sensor_msgs::PointCloud2ConstPtr &_laserCloudFullRes)
{
	mBuf.lock();
	fullResBuf.push(_laserCloudFullRes);
	mBuf.unlock();
} // laserCloudFullResHandler

void gpsHandler(const sensor_msgs::NavSatFix::ConstPtr &_gps)
{
    if(useGPS) {
        mBuf.lock();
        gpsBuf.push(_gps);
        mBuf.unlock();
    }
} // gpsHandler

void initNoises( void )
{
    gtsam::Vector priorNoiseVector6(6);
    priorNoiseVector6 << 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12;
    priorNoise = noiseModel::Diagonal::Variances(priorNoiseVector6);

    gtsam::Vector odomNoiseVector6(6);
    // odomNoiseVector6 << 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4;
    odomNoiseVector6 << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4;
    odomNoise = noiseModel::Diagonal::Variances(odomNoiseVector6);

    double loopNoiseScore = 0.5; // constant is ok...
    gtsam::Vector robustNoiseVector6(6); // gtsam::Pose3 factor has 6 elements (6D)
    robustNoiseVector6 << loopNoiseScore, loopNoiseScore, loopNoiseScore, loopNoiseScore, loopNoiseScore, loopNoiseScore;
    robustLoopNoise = gtsam::noiseModel::Robust::Create(
                    gtsam::noiseModel::mEstimator::Cauchy::Create(1), // optional: replacing Cauchy by DCS or GemanMcClure is okay but Cauchy is empirically good.
                    gtsam::noiseModel::Diagonal::Variances(robustNoiseVector6) );

    double bigNoiseTolerentToXY = 1000000000.0; // 1e9
    double gpsAltitudeNoiseScore = 250.0; // if height is misaligned after loop clsosing, use this value bigger
    gtsam::Vector robustNoiseVector3(3); // gps factor has 3 elements (xyz)
    robustNoiseVector3 << bigNoiseTolerentToXY, bigNoiseTolerentToXY, gpsAltitudeNoiseScore; // means only caring altitude here. (because LOAM-like-methods tends to be asymptotically flyging)
    robustGPSNoise = gtsam::noiseModel::Robust::Create(
                    gtsam::noiseModel::mEstimator::Cauchy::Create(1), // optional: replacing Cauchy by DCS or GemanMcClure is okay but Cauchy is empirically good.
                    gtsam::noiseModel::Diagonal::Variances(robustNoiseVector3) );

} // initNoises

Pose6D getOdom(nav_msgs::Odometry::ConstPtr _odom)
{
    auto tx = _odom->pose.pose.position.x;
    auto ty = _odom->pose.pose.position.y;
    auto tz = _odom->pose.pose.position.z;

    double roll, pitch, yaw;
    geometry_msgs::Quaternion quat = _odom->pose.pose.orientation;
    tf::Matrix3x3(tf::Quaternion(quat.x, quat.y, quat.z, quat.w)).getRPY(roll, pitch, yaw);

    return Pose6D{tx, ty, tz, roll, pitch, yaw}; 
} // getOdom

double pose2yaw(const Pose6D& _p1)
{
    Eigen::Affine3f SE3_p1 = pcl::getTransformation(_p1.x, _p1.y, _p1.z, _p1.roll, _p1.pitch, _p1.yaw);
  
    
    float p1_dx, p1_dy, p1_dz, p1_droll, p1_dpitch, p1_dyaw;

    pcl::getTranslationAndEulerAngles (SE3_p1, p1_dx, p1_dy, p1_dz, p1_droll, p1_dpitch, p1_dyaw);

    #define rad2deg (180/3.1415)
    
    // yaw (0--->180(-180)--->0)
    double yaw_deg = p1_dyaw *rad2deg;
    std::cout << "p1 : " << yaw_deg<< std::endl;
    
    if(yaw_deg<0)
        yaw_deg = 360 + yaw_deg;
    // yaw (0--->360)
    

}

Pose6D diffTransformation(const Pose6D& _p1, const Pose6D& _p2)
{
    Eigen::Affine3f SE3_p1 = pcl::getTransformation(_p1.x, _p1.y, _p1.z, _p1.roll, _p1.pitch, _p1.yaw);
    Eigen::Affine3f SE3_p2 = pcl::getTransformation(_p2.x, _p2.y, _p2.z, _p2.roll, _p2.pitch, _p2.yaw);
    Eigen::Matrix4f SE3_delta0 = SE3_p1.matrix().inverse() * SE3_p2.matrix();
    Eigen::Affine3f SE3_delta; SE3_delta.matrix() = SE3_delta0;
    float dx, dy, dz, droll, dpitch, dyaw;
    pcl::getTranslationAndEulerAngles (SE3_delta, dx, dy, dz, droll, dpitch, dyaw);
    // std::cout << "delta : " << dx << ", " << dy << ", " << dz << ", " << droll << ", " << dpitch << ", " << dyaw << std::endl;

    return Pose6D{double(abs(dx)), double(abs(dy)), double(abs(dz)), double(abs(droll)), double(abs(dpitch)), double(abs(dyaw))};
} // SE3Diff


pcl::PointCloud<PointType>::Ptr local2global(const pcl::PointCloud<PointType>::Ptr &cloudIn, const Pose6D& tf)
{
    pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

    int cloudSize = cloudIn->size();
    cloudOut->resize(cloudSize);

    Eigen::Affine3f transCur = pcl::getTransformation(tf.x, tf.y, tf.z, tf.roll, tf.pitch, tf.yaw);
    
    int numberOfCores = 16;
    #pragma omp parallel for num_threads(numberOfCores)
    for (int i = 0; i < cloudSize; ++i)
    {
        const auto &pointFrom = cloudIn->points[i];
        cloudOut->points[i].x = transCur(0,0) * pointFrom.x + transCur(0,1) * pointFrom.y + transCur(0,2) * pointFrom.z + transCur(0,3);
        cloudOut->points[i].y = transCur(1,0) * pointFrom.x + transCur(1,1) * pointFrom.y + transCur(1,2) * pointFrom.z + transCur(1,3);
        cloudOut->points[i].z = transCur(2,0) * pointFrom.x + transCur(2,1) * pointFrom.y + transCur(2,2) * pointFrom.z + transCur(2,3);
        cloudOut->points[i].intensity = pointFrom.intensity;
    }

    return cloudOut;
}

void pubPath( void )
{
    // pub odom and path 
    nav_msgs::Odometry odomAftPGO;
    nav_msgs::Path pathAftPGO;
    pathAftPGO.header.frame_id = "camera_init";
    mKF.lock(); 
    // for (int node_idx=0; node_idx < int(keyframePosesUpdated.size()) - 1; node_idx++) // -1 is just delayed visualization (because sometimes mutexed while adding(push_back) a new one)
    for (int node_idx=0; node_idx < recentIdxUpdated; node_idx++) // -1 is just delayed visualization (because sometimes mutexed while adding(push_back) a new one)
    {
        const Pose6D& pose_est = keyframePosesUpdated.at(node_idx); // upodated poses
        // const gtsam::Pose3& pose_est = isamCurrentEstimate.at<gtsam::Pose3>(node_idx);

        nav_msgs::Odometry odomAftPGOthis;
        odomAftPGOthis.header.frame_id = "camera_init";
        odomAftPGOthis.child_frame_id = "/aft_pgo";
        odomAftPGOthis.header.stamp = ros::Time().fromSec(keyframeTimes.at(node_idx));
        odomAftPGOthis.pose.pose.position.x = pose_est.x;
        odomAftPGOthis.pose.pose.position.y = pose_est.y;
        odomAftPGOthis.pose.pose.position.z = pose_est.z;
        odomAftPGOthis.pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(pose_est.roll, pose_est.pitch, pose_est.yaw);
        odomAftPGO = odomAftPGOthis;

        geometry_msgs::PoseStamped poseStampAftPGO;
        poseStampAftPGO.header = odomAftPGOthis.header;
        poseStampAftPGO.pose = odomAftPGOthis.pose.pose;

        pathAftPGO.header.stamp = odomAftPGOthis.header.stamp;
        pathAftPGO.header.frame_id = "camera_init";
        pathAftPGO.poses.push_back(poseStampAftPGO);
    }
    mKF.unlock(); 
    pubOdomAftPGO.publish(odomAftPGO); // last pose 
    pubPathAftPGO.publish(pathAftPGO); // poses 

    static tf::TransformBroadcaster br;
    tf::Transform transform;
    tf::Quaternion q;
    transform.setOrigin(tf::Vector3(odomAftPGO.pose.pose.position.x, odomAftPGO.pose.pose.position.y, odomAftPGO.pose.pose.position.z));
    q.setW(odomAftPGO.pose.pose.orientation.w);
    q.setX(odomAftPGO.pose.pose.orientation.x);
    q.setY(odomAftPGO.pose.pose.orientation.y);
    q.setZ(odomAftPGO.pose.pose.orientation.z);
    transform.setRotation(q);
    br.sendTransform(tf::StampedTransform(transform, odomAftPGO.header.stamp, "camera_init", "/aft_pgo"));
} // pubPath

void updatePoses(void)
{
    mKF.lock(); 
    for (int node_idx=0; node_idx < int(isamCurrentEstimate.size()); node_idx++)
    {
        Pose6D& p =keyframePosesUpdated[node_idx];
        p.x = isamCurrentEstimate.at<gtsam::Pose3>(node_idx).translation().x();
        p.y = isamCurrentEstimate.at<gtsam::Pose3>(node_idx).translation().y();
        p.z = isamCurrentEstimate.at<gtsam::Pose3>(node_idx).translation().z();
        p.roll = isamCurrentEstimate.at<gtsam::Pose3>(node_idx).rotation().roll();
        p.pitch = isamCurrentEstimate.at<gtsam::Pose3>(node_idx).rotation().pitch();
        p.yaw = isamCurrentEstimate.at<gtsam::Pose3>(node_idx).rotation().yaw();
    }
    mKF.unlock();

    mtxRecentPose.lock();
    const gtsam::Pose3& lastOptimizedPose = isamCurrentEstimate.at<gtsam::Pose3>(int(isamCurrentEstimate.size())-1);
    recentOptimizedX = lastOptimizedPose.translation().x();
    recentOptimizedY = lastOptimizedPose.translation().y();

    recentIdxUpdated = int(keyframePosesUpdated.size()) - 1;

    mtxRecentPose.unlock();
} // updatePoses

void runISAM2opt(void)
{
    // called when a variable added 
    isam->update(gtSAMgraph, initialEstimate);
    isam->update();
    
    gtSAMgraph.resize(0);
    initialEstimate.clear();

    isamCurrentEstimate = isam->calculateEstimate();
    updatePoses();
}

pcl::PointCloud<PointType>::Ptr transformPointCloud(pcl::PointCloud<PointType>::Ptr cloudIn, gtsam::Pose3 transformIn)
{
    pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

    PointType *pointFrom;

    int cloudSize = cloudIn->size();
    cloudOut->resize(cloudSize);

    Eigen::Affine3f transCur = pcl::getTransformation(
                                    transformIn.translation().x(), transformIn.translation().y(), transformIn.translation().z(), 
                                    transformIn.rotation().roll(), transformIn.rotation().pitch(), transformIn.rotation().yaw() );
    
    int numberOfCores = 8; // TODO move to yaml 
    #pragma omp parallel for num_threads(numberOfCores)
    for (int i = 0; i < cloudSize; ++i)
    {
        pointFrom = &cloudIn->points[i];
        cloudOut->points[i].x = transCur(0,0) * pointFrom->x + transCur(0,1) * pointFrom->y + transCur(0,2) * pointFrom->z + transCur(0,3);
        cloudOut->points[i].y = transCur(1,0) * pointFrom->x + transCur(1,1) * pointFrom->y + transCur(1,2) * pointFrom->z + transCur(1,3);
        cloudOut->points[i].z = transCur(2,0) * pointFrom->x + transCur(2,1) * pointFrom->y + transCur(2,2) * pointFrom->z + transCur(2,3);
        cloudOut->points[i].intensity = pointFrom->intensity;
    }
    return cloudOut;
} // transformPointCloud



void loopFindNearKeyframesCloud_jxx( pcl::PointCloud<PointType>::Ptr& nearKeyframes, const int& key, const int& submap_size, const int& root_idx)
{
    // extract and stacking near keyframes (in global coord)
    nearKeyframes->clear();
    for (int i = -submap_size; i <= submap_size; ++i) {
        int keyNear = key + i;
        if (keyNear < 0 || keyNear >= int(keyframeLaserClouds.size()) )
            continue;

        mKF.lock(); 
        *nearKeyframes += * local2global(keyframeLaserClouds[keyNear], keyframePosesUpdated[root_idx+i]);
        mKF.unlock(); 
    }

    if (nearKeyframes->empty())
        return;

    //再转换到key node下
    //*nearKeyframes = *global2local(nearKeyframes,keyframePosesUpdated[root_idx]);

   
    float Xupper = keyframePosesUpdated[key].x+loopCutRange;
    float Xlower = keyframePosesUpdated[key].x-loopCutRange;

    float Yupper = keyframePosesUpdated[key].y+loopCutRange;
    float Ylower = keyframePosesUpdated[key].y-loopCutRange;

    pcl::PointCloud<PointType>::Ptr cloud_cut(new pcl::PointCloud<PointType>());
    for(int i=0;i<nearKeyframes->size();i++)
    {
        // if(nearKeyframes->points[i].x<40 && nearKeyframes->points[i].x>-40
        // && nearKeyframes->points[i].y<40 && nearKeyframes->points[i].y>-40)
        if(nearKeyframes->points[i].x<Xupper && nearKeyframes->points[i].x>Xlower
        && nearKeyframes->points[i].y<Yupper && nearKeyframes->points[i].y>Ylower)
        {
            cloud_cut->points.emplace_back(nearKeyframes->points[i]);
        }
    }

    // downsample near keyframes
    pcl::PointCloud<PointType>::Ptr cloud_temp(new pcl::PointCloud<PointType>());
    //downSizeFilterICP.setInputCloud(nearKeyframes);
    downSizeFilterICP.setInputCloud(cloud_cut);
    downSizeFilterICP.filter(*cloud_temp);
    *nearKeyframes = *cloud_temp;
} // loopFindNearKeyframesCloud

Eigen::Affine3f pclPointToAffine3f(Pose6D tf)
{ 
    return pcl::getTransformation(tf.x, tf.y, tf.z, tf.roll, tf.pitch, tf.yaw);
}


std::optional<gtsam::Pose3> doICPVirtualRelative( int _loop_kf_idx, int _curr_kf_idx )
{
    // parse pointclouds
    //int historyKeyframeSearchNum = 100; // enough. ex. [-25, 25] covers submap length of 50x1 = 50m if every kf gap is 1m
    pcl::PointCloud<PointType>::Ptr cureKeyframeCloud(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr targetKeyframeCloud(new pcl::PointCloud<PointType>());

    loopFindNearKeyframesCloud_jxx(cureKeyframeCloud, _curr_kf_idx,historyKeyframeSearchNum_curr, _curr_kf_idx);
    loopFindNearKeyframesCloud_jxx(targetKeyframeCloud, _loop_kf_idx, historyKeyframeSearchNum_map, _loop_kf_idx); 

    // std::string curr_node_idx_str = padZeros(_curr_kf_idx);
    // pcl::io::savePCDFileBinary("/home/jiangxx/workspace/" + curr_node_idx_str + ".pcd", *cureKeyframeCloud); // scan 
    // curr_node_idx_str = padZeros(_loop_kf_idx);
    // pcl::io::savePCDFileBinary("/home/jiangxx/workspace/" + curr_node_idx_str + ".pcd", *targetKeyframeCloud); // scan 

    if (cureKeyframeCloud->size() < 300 || targetKeyframeCloud->size() < 1000)
        return std::nullopt;


    Eigen::Matrix4f guess_trans;
    
    // 初始化正态分布(NDT)对象
    pcl::NormalDistributionsTransform<PointType, PointType> ndt;

    // 根据输入数据的尺度设置NDT相关参数

    ndt.setTransformationEpsilon (0.01);   //为终止条件设置最小转换差异
        
    ndt.setStepSize (ndt_stepSize);    // 0.5--->mid70 ok为more-thuente线搜索设置最大步长

    ndt.setResolution (ndt_resolution);   // 4.0 --->mid70 ok   设置NDT网格网格结构的分辨率（voxelgridcovariance）
    //以上参数在使用房间尺寸比例下运算比较好，但是如果需要处理例如一个咖啡杯子的扫描之类更小的物体，需要对参数进行很大程度的缩小

    //设置匹配迭代的最大次数，这个参数控制程序运行的最大迭代次数，一般来说这个限制值之前优化程序会在epsilon变换阀值下终止
    //添加最大迭代次数限制能够增加程序的鲁棒性阻止了它在错误的方向上运行时间过长
    ndt.setMaximumIterations (100);

    pcl::PointCloud<PointType>::Ptr filtered_cloud (new pcl::PointCloud<PointType>);
    pcl::ApproximateVoxelGrid<PointType> approximate_voxel_filter;
    approximate_voxel_filter.setLeafSize (0.11, 0.11, 0.11);
    approximate_voxel_filter.setInputCloud (cureKeyframeCloud);
    approximate_voxel_filter.filter (*filtered_cloud);

    ndt.setInputSource (filtered_cloud);  //源点云
    // Setting point cloud to be aligned to.
    ndt.setInputTarget (targetKeyframeCloud);  //目标点云

    // 设置使用机器人测距法得到的粗略初始变换矩阵结果
       
    Eigen::Matrix4f init_guess ;
     init_guess << 1.0, 0.0, 0.0, 0.0,
                    0.0, 1.0, 0.0, 0.0,
                    0.0, 0.0, 1.0, 0.0,
                    0.0, 0.0, 0.0, 1.0;

    //  init_guess << 1.0, 0.0, 0.0, 3.0,
    //                   0.0, 1.0, 0.0, -3.0,
    //                   0.0, 0.0, 1.0, -3.5,
    //                   0.0, 0.0, 0.0, 1.0;

    // 计算需要的刚体变换以便将输入的源点云匹配到目标点云
    pcl::PointCloud<PointType>::Ptr output_cloud (new pcl::PointCloud<PointType>);
    ndt.align (*output_cloud, init_guess);
    //这个地方的output_cloud不能作为最终的源点云变换，因为上面对点云进行了滤波处理
    std::cout << "Normal Distributions Transform has converged:" << ndt.hasConverged ()
                    << " score: " << ndt.getFitnessScore () << std::endl;

    guess_trans = ndt.getFinalTransformation();

    // ICP Settings
    pcl::IterativeClosestPoint<PointType, PointType> icp;
    icp.setMaxCorrespondenceDistance(150); // giseop , use a value can cover 2*historyKeyframeSearchNum range in meter 
    icp.setMaximumIterations(100);
    icp.setTransformationEpsilon(1e-6);
    icp.setEuclideanFitnessEpsilon(1e-6);
    icp.setRANSACIterations(0);

    // Align pointclouds
    icp.setInputSource(cureKeyframeCloud);
    icp.setInputTarget(targetKeyframeCloud);
    pcl::PointCloud<PointType>::Ptr unused_result(new pcl::PointCloud<PointType>());
    icp.align(*unused_result, guess_trans);
 
    //float loopFitnessScoreThreshold = 0.5; // user parameter but fixed low value is safe. 
    if (icp.hasConverged() == false || icp.getFitnessScore() > loopFitnessScoreThreshold) {
    //if (ndt.hasConverged() == false || ndt.getFitnessScore() > loopFitnessScoreThreshold) {
        std::cout << "[SC loop] ndt fitness test failed (" << ndt.getFitnessScore() << " > " << loopFitnessScoreThreshold << "). Reject this SC loop." << std::endl;
        return std::nullopt;
    } else {
        std::cout << "[SC loop] ndt fitness test passed (" << ndt.getFitnessScore() << " < " << loopFitnessScoreThreshold << "). Add this SC loop." << std::endl;
    }
    
    // loop verification 
    pcl::PointCloud<PointType>::Ptr closed_cloud(new pcl::PointCloud<PointType>());
    pcl::transformPointCloud(*cureKeyframeCloud, *closed_cloud, ndt.getFinalTransformation());

    sensor_msgs::PointCloud2 cureKeyframeCloudMsg;
    pcl::toROSMsg(*closed_cloud, cureKeyframeCloudMsg);
    cureKeyframeCloudMsg.header.frame_id = "camera_init";
    pubLoopScanLocal.publish(cureKeyframeCloudMsg);

    sensor_msgs::PointCloud2 targetKeyframeCloudMsg;
    pcl::toROSMsg(*targetKeyframeCloud, targetKeyframeCloudMsg);
    targetKeyframeCloudMsg.header.frame_id = "camera_init";
    pubLoopSubmapLocal.publish(targetKeyframeCloudMsg);



    // Get pose transformation
    float x, y, z, roll, pitch, yaw;
    Eigen::Affine3f correctionLidarFrame;
    correctionLidarFrame = icp.getFinalTransformation();
    //correctionLidarFrame = ndt.getFinalTransformation ();


    Eigen::Affine3f tWrong = pclPointToAffine3f(keyframePosesUpdated[_curr_kf_idx]);
    Eigen::Affine3f tCorrect = correctionLidarFrame * tWrong;
    pcl::getTranslationAndEulerAngles (tCorrect, x, y, z, roll, pitch, yaw);

    std::cout << ndt.getFinalTransformation ()<<endl;
    //pcl::getTranslationAndEulerAngles (correctionLidarFrame, x, y, z, roll, pitch, yaw);
    gtsam::Pose3 poseFrom = Pose3(Rot3::RzRyRx(roll, pitch, yaw), Point3(x, y, z));
    //gtsam::Pose3 poseTo = Pose3(Rot3::RzRyRx(0.0, 0.0, 0.0), Point3(0.0, 0.0, 0.0));

    auto tf2 =keyframePosesUpdated[_loop_kf_idx];
    gtsam::Pose3 poseTo = gtsam::Pose3(gtsam::Rot3::RzRyRx(double(tf2.roll), double(tf2.pitch), double(tf2.yaw)),
                                  gtsam::Point3(double(tf2.x),    double(tf2.y),     double(tf2.z)));


    std::cout <<"return poseFrom.between(poseTo)"<<std::endl;
    return poseFrom.between(poseTo);
    
} // doICPVirtualRelative

void process_pg()
{
    while(1)
    {
		while ( !odometryBuf.empty() && !fullResBuf.empty() )
        {
            //
            // pop and check keyframe is or not  
            // 
			mBuf.lock();       
            while (!odometryBuf.empty() && odometryBuf.front()->header.stamp.toSec() < fullResBuf.front()->header.stamp.toSec())
                odometryBuf.pop();
            if (odometryBuf.empty())
            {
                mBuf.unlock();
                break;
            }

            // Time equal check
            timeLaserOdometry = odometryBuf.front()->header.stamp.toSec();
            timeLaserOdometr_header = odometryBuf.front()->header;
            timeLaser = fullResBuf.front()->header.stamp.toSec();
            // TODO

            laserCloudFullRes->clear();
            pcl::PointCloud<PointType>::Ptr thisKeyFrame(new pcl::PointCloud<PointType>());
            pcl::fromROSMsg(*fullResBuf.front(), *thisKeyFrame);
            fullResBuf.pop();

            Pose6D pose_curr = getOdom(odometryBuf.front());
            odometryBuf.pop();

            // find nearest gps 
            double eps = 0.1; // find a gps topioc arrived within eps second 
            while (!gpsBuf.empty()) {
                auto thisGPS = gpsBuf.front();
                auto thisGPSTime = thisGPS->header.stamp.toSec();
                if( abs(thisGPSTime - timeLaserOdometry) < eps ) {
                    currGPS = thisGPS;
                    hasGPSforThisKF = true; 
                    break;
                } else {
                    hasGPSforThisKF = false;
                }
                gpsBuf.pop();
            }
            mBuf.unlock(); 

            //
            // Early reject by counting local delta movement (for equi-spereated kf drop)
            // 
            odom_pose_prev = odom_pose_curr;
            odom_pose_curr = pose_curr;
            Pose6D dtf = diffTransformation(odom_pose_prev, odom_pose_curr); // dtf means delta_transform

            double delta_translation = sqrt(dtf.x*dtf.x + dtf.y*dtf.y + dtf.z*dtf.z); // note: absolute value. 
            translationAccumulated += delta_translation;
            rotaionAccumulated += (dtf.roll + dtf.pitch + dtf.yaw); // sum just naive approach.  

            if( translationAccumulated > keyframeMeterGap || rotaionAccumulated > keyframeRadGap ) {
                isNowKeyFrame = true;
                translationAccumulated = 0.0; // reset 
                rotaionAccumulated = 0.0; // reset 
            } else {
                isNowKeyFrame = false;
            }

            if( ! isNowKeyFrame ) 
                continue; 

            if( !gpsOffsetInitialized ) {
                if(hasGPSforThisKF) { // if the very first frame 
                    gpsAltitudeInitOffset = currGPS->altitude;
                    gpsOffsetInitialized = true;
                } 
            }

            //
            // Save data and Add consecutive node 
            //
            pcl::PointCloud<PointType>::Ptr thisKeyFrameDS(new pcl::PointCloud<PointType>());
            downSizeFilterScancontext.setInputCloud(thisKeyFrame);
            downSizeFilterScancontext.filter(*thisKeyFrameDS);

            mKF.lock(); 
            keyframeLaserClouds.push_back(thisKeyFrameDS);
            keyframePoses.push_back(pose_curr);
            keyframePosesUpdated.push_back(pose_curr); // init
            keyframeTimes.push_back(timeLaserOdometry);

            scManager.makeAndSaveScancontextAndKeys(*thisKeyFrameDS);

            laserCloudMapPGORedraw = true;
            mKF.unlock(); 

            const int prev_node_idx = keyframePoses.size() - 2; 
            const int curr_node_idx = keyframePoses.size() - 1; // becuase cpp starts with 0 (actually this index could be any number, but for simple implementation, we follow sequential indexing)
            if( ! gtSAMgraphMade /* prior node */) {
                const int init_node_idx = 0; 
                gtsam::Pose3 poseOrigin = Pose6DtoGTSAMPose3(keyframePoses.at(init_node_idx));
                // auto poseOrigin = gtsam::Pose3(gtsam::Rot3::RzRyRx(0.0, 0.0, 0.0), gtsam::Point3(0.0, 0.0, 0.0));

                mtxPosegraph.lock();
                {
                    // prior factor 
                    gtSAMgraph.add(gtsam::PriorFactor<gtsam::Pose3>(init_node_idx, poseOrigin, priorNoise));
                    initialEstimate.insert(init_node_idx, poseOrigin);
                    // runISAM2opt();          
                }   
                mtxPosegraph.unlock();

                gtSAMgraphMade = true; 

                cout << "posegraph prior node " << init_node_idx << " added" << endl;
            } else /* consecutive node (and odom factor) after the prior added */ { // == keyframePoses.size() > 1 
                gtsam::Pose3 poseFrom = Pose6DtoGTSAMPose3(keyframePoses.at(prev_node_idx));
                gtsam::Pose3 poseTo = Pose6DtoGTSAMPose3(keyframePoses.at(curr_node_idx));

                mtxPosegraph.lock();
                {
                    // odom factor
                    gtSAMgraph.add(gtsam::BetweenFactor<gtsam::Pose3>(prev_node_idx, curr_node_idx, poseFrom.between(poseTo), odomNoise));

                    // gps factor 
                    if(hasGPSforThisKF) {
                        double curr_altitude_offseted = currGPS->altitude - gpsAltitudeInitOffset;
                        mtxRecentPose.lock();
                        gtsam::Point3 gpsConstraint(recentOptimizedX, recentOptimizedY, curr_altitude_offseted); // in this example, only adjusting altitude (for x and y, very big noises are set) 
                        mtxRecentPose.unlock();
                        gtSAMgraph.add(gtsam::GPSFactor(curr_node_idx, gpsConstraint, robustGPSNoise));
                        cout << "GPS factor added at node " << curr_node_idx << endl;
                    }
                    initialEstimate.insert(curr_node_idx, poseTo);                
                    // runISAM2opt();
                }
                mtxPosegraph.unlock();

                if(curr_node_idx % 100 == 0)
                    cout << "posegraph odom node " << curr_node_idx << " added." << endl;
            }
          
        }
        // wait (must required for running the while loop)
        std::chrono::milliseconds dura(2);
        std::this_thread::sleep_for(dura);
    }
} // process_pg

void updateLoopFactor()
{
     if (loopIndexQueue.empty())
            return;

        for (int i = 0; i < (int)loopIndexQueue.size(); ++i)
        {
            int indexFrom = loopIndexQueue[i].first;
            int indexTo = loopIndexQueue[i].second;
            gtsam::Pose3 poseBetween = loopPoseQueue[i];
            //gtsam::noiseModel::Diagonal::shared_ptr noiseBetween = loopNoiseQueue[i];
            //gtsam::noiseModel::Robust::shared_ptr noiseBetween = loopNoiseQueue[i];
            gtSAMgraph.add(BetweenFactor<Pose3>(indexFrom, indexTo, poseBetween, robustLoopNoise));  //robustLoopNoise
        }

        loopIndexQueue.clear();
        loopPoseQueue.clear();
        loopNoiseQueue.clear();
        aLoopIsClosed = true;
}
void performSCLoopClosure_jxx(void)
{
    if( int(keyframePosesUpdated.size()) < 200) // do not try too early 
        return;

    cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
    //每次都copy一遍，好麻烦
    mKF.lock(); 
    for(int i=0;i<keyframePosesUpdated.size();i++)
    {
        PointType thisPose3D;
        thisPose3D.x = keyframePosesUpdated[i].x;
        thisPose3D.y = keyframePosesUpdated[i].y;
        thisPose3D.z = keyframePosesUpdated[i].z;
        thisPose3D.intensity = i; // this can be used as index
        cloudKeyPoses3D->push_back(thisPose3D);
    }
    copy_keyframeTimes = keyframeTimes;
    mKF.unlock(); 

    // find keys
    int loopKeyCur;
    int loopKeyPre;
   
    if (detectLoopClosureDistance(&loopKeyCur, &loopKeyPre) == false)
        return;

    auto relative_pose_optional = doICPVirtualRelative(loopKeyPre, loopKeyCur);
    if(relative_pose_optional) {
        gtsam::Pose3 relative_pose = relative_pose_optional.value();
        // mtxPosegraph.lock();
        // gtSAMgraph.add(gtsam::BetweenFactor<gtsam::Pose3>(loopKeyPre, loopKeyCur, relative_pose, robustLoopNoise));      
        // mtxPosegraph.unlock();

        mKF.lock();
        loopIndexQueue.push_back(make_pair(loopKeyCur, loopKeyPre));
        loopPoseQueue.push_back(relative_pose);
        //loopNoiseQueue.push_back(robustLoopNoise);   //loopNoiseQueue  odomNoise
        mKF.unlock();

        // add loop constriant
        loopIndexContainer[loopKeyCur] = loopKeyPre;
    } 
} // performSCLoopClosure_jxx

bool detectLoopClosureDistance(int *latestID, int *closestID)
{
    int loopKeyCur = cloudKeyPoses3D->size() - 1;
    int loopKeyPre = -1;

    // check loop constraint added before
    auto it = loopIndexContainer.find(loopKeyCur);
    if (it != loopIndexContainer.end())
        return false;

    // find the closest history key frame
    int historyKeyframeSearchRadius = 20;
    std::vector<int> pointSearchIndLoop;
    std::vector<float> pointSearchSqDisLoop;
     cout<<" cloudKeyPoses3D.size()=" <<cloudKeyPoses3D->size()<<endl;   
    kdtreeHistoryKeyPoses->setInputCloud(cloudKeyPoses3D);
    kdtreeHistoryKeyPoses->radiusSearch(cloudKeyPoses3D->back(), historyKeyframeSearchRadius, pointSearchIndLoop, pointSearchSqDisLoop, 0);

    cout<<" kdtreeHistoryKeyPoses->radiusSearch done!,pointSearchIndLoop.size()" <<pointSearchIndLoop.size()<<endl;   
    for (int i = 0; i < (int)pointSearchIndLoop.size(); ++i)
    {
        int id = pointSearchIndLoop[i];
        double historyKeyframeSearchTimeDiff = 50;
        // if (abs(copy_cloudKeyPoses6D->points[id].time - timeLaserInfoCur) > historyKeyframeSearchTimeDiff)
        cout<<" timediff=" <<abs(copy_keyframeTimes[id] - timeLaserOdometry)<<endl; 
        if (abs(copy_keyframeTimes[id] - timeLaserOdometry) > historyKeyframeSearchTimeDiff)
        {
            loopKeyPre = id;
            break;
        }
    }
    cout<<" loopKeyPre=" <<loopKeyPre<<"<--->"<<loopKeyCur<<endl;  
    if (loopKeyPre == -1 || loopKeyCur == loopKeyPre)
        return false;

    *latestID = loopKeyCur;
    *closestID = loopKeyPre;

    return true;
}// detectLoopClosureDistance

void process_lcd(void)
{
    float loopClosureFrequency = 1.0; // can change 
    ros::Rate rate(loopClosureFrequency);
    while (ros::ok())
    {
        rate.sleep();
        //performSCLoopClosure();
        performSCLoopClosure_jxx(); // TODO
        visualizeLoopClosure();
    }
} // process_lcd

void visualizeLoopClosure()
{
        if (loopIndexContainer.empty())
            return;
        
        visualization_msgs::MarkerArray markerArray;
        // loop nodes
        visualization_msgs::Marker markerNode;
        markerNode.header.frame_id = "camera_init";
        markerNode.header.stamp = timeLaserOdometr_header.stamp;
        markerNode.action = visualization_msgs::Marker::ADD;
        markerNode.type = visualization_msgs::Marker::SPHERE_LIST;
        markerNode.ns = "loop_nodes";
        markerNode.id = 0;
        markerNode.pose.orientation.w = 1;
        markerNode.scale.x = 0.3; markerNode.scale.y = 0.3; markerNode.scale.z = 0.3; 
        markerNode.color.r = 0; markerNode.color.g = 0.8; markerNode.color.b = 1;
        markerNode.color.a = 1;
        // loop edges
        visualization_msgs::Marker markerEdge;
        markerEdge.header.frame_id = "camera_init";
        markerEdge.header.stamp = timeLaserOdometr_header.stamp;
        markerEdge.action = visualization_msgs::Marker::ADD;
        markerEdge.type = visualization_msgs::Marker::LINE_LIST;
        markerEdge.ns = "loop_edges";
        markerEdge.id = 1;
        markerEdge.pose.orientation.w = 1;
        markerEdge.scale.x = 0.2;
        markerEdge.color.r = 0.9; markerEdge.color.g = 0.0; markerEdge.color.b = 0;
        markerEdge.color.a = 1;

        for (auto it = loopIndexContainer.begin(); it != loopIndexContainer.end(); ++it)
        {
            int key_cur = it->first;
            int key_pre = it->second;
            geometry_msgs::Point p;
            
            // p.x = keyframePosesUpdated[key_cur].x;
            // p.y = keyframePosesUpdated[key_cur].y;
            // p.z = keyframePosesUpdated[key_cur].z;

            p.x = cloudKeyPoses3D->points[key_cur].x;
            p.y = cloudKeyPoses3D->points[key_cur].y;
            p.z = cloudKeyPoses3D->points[key_cur].z;
            markerNode.points.push_back(p);
            markerEdge.points.push_back(p);
            // p.x = keyframePosesUpdated[key_pre].x;
            // p.y = keyframePosesUpdated[key_pre].y;
            // p.z = keyframePosesUpdated[key_pre].z;

            p.x = cloudKeyPoses3D->points[key_pre].x;
            p.y = cloudKeyPoses3D->points[key_pre].y;
            p.z = cloudKeyPoses3D->points[key_pre].z;
            markerNode.points.push_back(p);
            markerEdge.points.push_back(p);
        }

        markerArray.markers.push_back(markerNode);
        markerArray.markers.push_back(markerEdge);
        pubLoopConstraintEdge.publish(markerArray);
}


void process_viz_path(void)
{
    float hz = 10.0; 
    ros::Rate rate(hz);
    while (ros::ok()) {
        rate.sleep();
        if(recentIdxUpdated > 1) {
            pubPath();
        }
    }
}

void process_isam(void)
{
    float hz = 1; 
    ros::Rate rate(hz);
    while (ros::ok()) {
        rate.sleep();
        if( gtSAMgraphMade ) {
            mtxPosegraph.lock();
            updateLoopFactor();
            runISAM2opt();
            //cout << "running isam2 optimization ..." << endl;
            mtxPosegraph.unlock();

            // saveOptimizedVerticesKITTIformat(isamCurrentEstimate, pgKITTIformat); // pose
            // saveOdometryVerticesKITTIformat(odomKITTIformat); // pose
        }
    }
}

bool save_map_service(aloam_velodyne::SaveMapRequest &req, aloam_velodyne::SaveMapResponse &res)
{
        std::cout << "req.destination" << req.destination << std::endl;
        std::cout << "req.resolution" << req.resolution << std::endl;
        std::cout << "req.utm" << req.utm << std::endl;

        cout << "****************************************************" << endl;
        cout << "start saving map to pcd files ..." << endl;

        /**************** save map ****************/
   
        laserCloudMapPGO_SavePcd->clear();

        mKF.lock(); 
        // for (int node_idx=0; node_idx < int(keyframePosesUpdated.size()); node_idx++) {
       
        for (int node_idx=0; node_idx < recentIdxUpdated; node_idx++) {
            *laserCloudMapPGO_SavePcd += *local2global(keyframeLaserClouds[node_idx], keyframePosesUpdated[node_idx]);
        }
        mKF.unlock(); 
     
        if (1)
        {
            cout << "=========================>saveing map ..." << endl;
            //string file_name = string("scans.pcd");
            //string all_points_dir(string(string(ROOT_DIR) + "PCD/") + file_name);
            pcl::PCDWriter pcd_writer;
            //cout << "current scan saved to /PCD/" << file_name<<endl;
            //pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
            pcd_writer.writeBinary(pgScansDirectory + "Opt_map" + ".pcd", *laserCloudMapPGO_SavePcd);

            //pcl::io::savePCDFileBinary(pgScansDirectory + "map" + ".pcd", *thisKeyFrame); // scan 
        }
        cout << "****************************************************" << endl;
        cout << "saving map completed" << endl;

        res.success = 1;
        return true;
}


void pubMap(void)
{
    int SKIP_FRAMES = 2; // sparse map visulalization to save computations 
    int counter = 0;

    laserCloudMapPGO->clear();

    mKF.lock(); 
    // for (int node_idx=0; node_idx < int(keyframePosesUpdated.size()); node_idx++) {
    for (int node_idx=0; node_idx < recentIdxUpdated; node_idx++) {
        if(counter % SKIP_FRAMES == 0) {
            *laserCloudMapPGO += *local2global(keyframeLaserClouds[node_idx], keyframePosesUpdated[node_idx]);
        }
        counter++;
    }
    mKF.unlock(); 

    downSizeFilterMapPGO.setInputCloud(laserCloudMapPGO);
    downSizeFilterMapPGO.filter(*laserCloudMapPGO);

    sensor_msgs::PointCloud2 laserCloudMapPGOMsg;
    pcl::toROSMsg(*laserCloudMapPGO, laserCloudMapPGOMsg);
    laserCloudMapPGOMsg.header.frame_id = "camera_init";
    pubMapAftPGO.publish(laserCloudMapPGOMsg);
}

void process_viz_map(void)
{
    float vizmapFrequency = 0.1; // 0.1 means run onces every 10s
    ros::Rate rate(vizmapFrequency);
    while (ros::ok()) {
        rate.sleep();
        if(recentIdxUpdated > 1) {
            pubMap();
        }

        // pubMap();
    }
} // pointcloud_viz


int main(int argc, char **argv)
{
	ros::init(argc, argv, "laserPGO");
	ros::NodeHandle nh;

	nh.param<std::string>("save_directory", save_directory, "/"); // pose assignment every k m move 
    pgScansDirectory = save_directory;

	nh.param<double>("keyframe_meter_gap", keyframeMeterGap, 2.0); // pose assignment every k m move 
	nh.param<double>("keyframe_deg_gap", keyframeDegGap, 10.0); // pose assignment every k deg rot 
    keyframeRadGap = deg2rad(keyframeDegGap);

	nh.param<double>("sc_dist_thres", scDistThres, 0.2);  
	nh.param<double>("sc_max_radius", scMaximumRadius, 80.0); // 80 is recommended for outdoor, and lower (ex, 20, 40) values are recommended for indoor 

    nh.param<double>("ndt_resolution", ndt_resolution, 0.2);
    nh.param<double>("ndt_stepSize", ndt_stepSize, 0.2);
    nh.param<double>("loopFitnessScoreThreshold", loopFitnessScoreThreshold, 0.2);
    nh.param<double>("loopCutRange", loopCutRange, 50);
    

    nh.param<int>("historyKeyframeSearchNum_curr", historyKeyframeSearchNum_curr, 50);
    nh.param<int>("historyKeyframeSearchNum_map", historyKeyframeSearchNum_map, 100);



    ros::ServiceServer save_map_service_server;
    save_map_service_server = nh.advertiseService("opt/save_map", &save_map_service);

    ISAM2Params parameters;
    parameters.relinearizeThreshold = 0.01;
    parameters.relinearizeSkip = 1;
    isam = new ISAM2(parameters);
    initNoises();

    scManager.setSCdistThres(scDistThres);
    scManager.setMaximumRadius(scMaximumRadius);

    float filter_size = 0.1; 
    downSizeFilterScancontext.setLeafSize(filter_size, filter_size, filter_size);
    // downSizeFilterICP.setLeafSize(filter_size, filter_size, filter_size);
    downSizeFilterICP.setLeafSize(0.1, 0.1, 0.1);

    double mapVizFilterSize;
	nh.param<double>("mapviz_filter_size", mapVizFilterSize, 0.4); // pose assignment every k frames 
    downSizeFilterMapPGO.setLeafSize(mapVizFilterSize, mapVizFilterSize, mapVizFilterSize);


    cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
    copy_cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
    kdtreeHistoryKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());


	ros::Subscriber subLaserCloudFullRes = nh.subscribe<sensor_msgs::PointCloud2>("/velodyne_cloud_registered_local", 100, laserCloudFullResHandler);
	ros::Subscriber subLaserOdometry = nh.subscribe<nav_msgs::Odometry>("/aft_mapped_to_init", 100, laserOdometryHandler);
	ros::Subscriber subGPS = nh.subscribe<sensor_msgs::NavSatFix>("/gps/fix", 100, gpsHandler);

	pubOdomAftPGO = nh.advertise<nav_msgs::Odometry>("/aft_pgo_odom", 100);
	pubOdomRepubVerifier = nh.advertise<nav_msgs::Odometry>("/repub_odom", 100);
	pubPathAftPGO = nh.advertise<nav_msgs::Path>("/aft_pgo_path", 100);
	pubMapAftPGO = nh.advertise<sensor_msgs::PointCloud2>("/aft_pgo_map", 100);

	pubLoopScanLocal = nh.advertise<sensor_msgs::PointCloud2>("/loop_scan_local", 100);
	pubLoopSubmapLocal = nh.advertise<sensor_msgs::PointCloud2>("/loop_submap_local", 100);

    pubLoopConstraintEdge = nh.advertise<visualization_msgs::MarkerArray>("/lio_sam/mapping/loop_closure_constraints", 1);


	std::thread posegraph_slam {process_pg}; // pose graph construction
	std::thread lc_detection {process_lcd}; // loop closure detection 
	//std::thread icp_calculation {process_icp}; // loop constraint calculation via icp 
	std::thread isam_update {process_isam}; // if you want to call less isam2 run (for saving redundant computations and no real-time visulization is required), uncommment this and comment all the above runisam2opt when node is added. 

	std::thread viz_map {process_viz_map}; // visualization - map (low frequency because it is heavy)
	std::thread viz_path {process_viz_path}; // visualization - path (high frequency)

 	ros::spin();

	return 0;
}
