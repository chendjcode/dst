#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <sensor_msgs/PointCloud2.h>

#include "pcl_ros/point_cloud.h"  // fromROSMsg以及pcl中的各种点云

// #include <pcl/io/pcd_io.h>
#include <pcl/registration/icp.h>
#include <pcl/common/transforms.h>
#include <pcl/console/time.h>   // TicToc

using namespace std;
using namespace sensor_msgs;
using namespace message_filters;

ros::Publisher pub;  // 融合点云发布

vector<Eigen::Matrix4f> calibration; // 标定矩阵，0->1 和 2->1

pcl::console::TicToc tim;  // 计时器

/* calib
 *
 * 雷达编号：0, 1, 2
 *
 * 0 -> 1:
 * x:-12.85, y:-4.59, z:1.55-1.6
 *
 * 2 -> 1:
 * x:-10.3, y:+6.7, z:1.55-1.16
 * */

void calib(const pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud0, const pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud1,
           const pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud2) {
    pcl::IterativeClosestPoint<pcl::PointXYZI, pcl::PointXYZI> icp01;
    icp01.setMaxCorrespondenceDistance (0.05);
    icp01.setInputSource(cloud0);
    icp01.setInputTarget(cloud1);

    Eigen::Matrix4f guess01;
    guess01 << 1, 0, 0, -12.85, 0, 1, 0, -4.59, 0, 0, 1, 0, 0, 0, 0, 1;

    pcl::PointCloud<pcl::PointXYZI> cloud0_final;
    icp01.align(cloud0_final, guess01);

    std::cout << "0号雷达标定，has converged:" << icp01.hasConverged() << " score: " <<
              icp01.getFitnessScore() << std::endl;
    std::cout << icp01.getFinalTransformation() << std::endl;

    pcl::IterativeClosestPoint<pcl::PointXYZI, pcl::PointXYZI> icp21;
    icp21.setMaxCorrespondenceDistance (0.05);
    icp21.setInputSource(cloud2);
    icp21.setInputTarget(cloud1);

    Eigen::Matrix4f guess21;
    guess21 << 1, 0, 0, -10, 0, 1, 0, 7.2, 0, 0, 1, 0.4, 0, 0, 0, 1;

    pcl::PointCloud<pcl::PointXYZI> cloud2_final;
    icp21.align(cloud2_final, guess21);

    std::cout << "2号雷达标定：has converged:" << icp21.hasConverged() << " score: " <<
              icp21.getFitnessScore() << std::endl;
    std::cout << icp21.getFinalTransformation() << std::endl;

    calibration.push_back(icp01.getFinalTransformation());
    calibration.push_back(icp21.getFinalTransformation());

}


void callback(const PointCloud2ConstPtr &velodyne0_data, const PointCloud2ConstPtr &velodyne1_data,
              const PointCloud2ConstPtr &velodyne2_data) {
    tim.tic();
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud0(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud1(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud2(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::fromROSMsg(*velodyne0_data, *cloud0);
    pcl::fromROSMsg(*velodyne1_data, *cloud1);
    pcl::fromROSMsg(*velodyne2_data, *cloud2);

    if (calibration.empty())
        calib(cloud0, cloud1, cloud2);

    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_merge(new pcl::PointCloud<pcl::PointXYZI>);
    Eigen::Matrix4f calib01 = calibration[0];
    Eigen::Matrix4f calib21 = calibration[1];

    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud01(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::transformPointCloud(*cloud0, *cloud01, calib01);

    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud21(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::transformPointCloud(*cloud2, *cloud21, calib21);

    *cloud_merge += *cloud1;
    *cloud_merge += *cloud01;
    *cloud_merge += *cloud21;

    cloud_merge->header = cloud1->header;
    cloud_merge->header.frame_id = "merge";
    pub.publish(cloud_merge);
    tim.toc_print();
}

int main(int argc, char **argv) {
    ros::init(argc, argv, "multi_lidar_calib");

    ros::NodeHandle nh;

    pub = nh.advertise<pcl::PointCloud<pcl::PointXYZI>>("/merge_points", 1);

    message_filters::Subscriber<PointCloud2> velodyne0_sub(nh, "/velodyne0/velodyne0_nodelet_manager_points", 1);
    message_filters::Subscriber<PointCloud2> velodyne1_sub(nh, "/velodyne1/velodyne1_nodelet_manager_points", 1);
    message_filters::Subscriber<PointCloud2> velodyne2_sub(nh, "/velodyne2/velodyne2_nodelet_manager_points", 1);


    typedef sync_policies::ApproximateTime<PointCloud2, PointCloud2, PointCloud2> LidarSyncPolicy;
    // ApproximateTime takes a queue size as its constructor argument, hence LidarSyncPolicy(10)
    Synchronizer<LidarSyncPolicy> sync(LidarSyncPolicy(10), velodyne0_sub, velodyne1_sub, velodyne2_sub);
    sync.registerCallback(boost::bind(&callback, _1, _2, _3));

    ros::spin();

    return 0;
}