//
// Created by cdj on 2020/2/7.
//
#include <ros/ros.h>
// PCL specific includes
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
// #include <pcl/conversions.h>
#include "pcl_ros/point_cloud.h"
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/registration/icp.h>
#include <pcl/console/time.h>   // TicToc


pcl::console::TicToc tim;
ros::Publisher pub;
pcl::PointCloud<pcl::PointXYZ>::Ptr pre_cloud;

// get T from pre to now
Eigen::Matrix<float, 4, 4>
icp(pcl::PointCloud<pcl::PointXYZ>::Ptr &_pre, pcl::PointCloud<pcl::PointXYZ>::Ptr &_now) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr pre(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr now(new pcl::PointCloud<pcl::PointXYZ>);
    // down sampling
    float max = 6.0f, min = 2.0f;
    for (auto p : _pre->points) {
        float distance = sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
        if ((distance < max) && (distance > min))
            pre->push_back(p);
    }
    for (auto p : _now->points) {
        float distance = sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
        if ((distance < max) && (distance > min))
            now->push_back(p);
    }
    // ICP
    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
    icp.setInputSource(pre);
    icp.setInputTarget(now);
    pcl::PointCloud<pcl::PointXYZ> Final;
    icp.align(Final, Eigen::Matrix<float, 4, 4>::Identity());
    return icp.getFinalTransformation();
}

void
cloud_cb(const sensor_msgs::PointCloud2ConstPtr &input) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr _cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr pre_cloud_(new pcl::PointCloud<pcl::PointXYZ>);  // transformed pre frame
    pcl::PointCloud<pcl::PointXYZ>::Ptr motion(new pcl::PointCloud<pcl::PointXYZ>);

    // convert
    pcl::fromROSMsg(*input, *_cloud);

    // down sampling
    pcl::VoxelGrid<pcl::PointXYZ> sor;
    float leafSize = 0.1f;
    sor.setLeafSize(leafSize, leafSize, leafSize);
    sor.setInputCloud(_cloud);
    sor.filter(*cloud);

    // first frame process
    if (!pre_cloud) {
        pre_cloud = cloud;
        return;
    }

    // cloud_ is transformed
    tim.tic();
    pcl::transformPointCloud(*pre_cloud, *pre_cloud_, icp(pre_cloud, cloud));
    auto elapsed = tim.toc();

    // kdtree match
    tim.tic();
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(pre_cloud_);
    for (auto p : cloud->points) {
        int K = 1;
        std::vector<int> pointIdxNKNSearch(K);
        std::vector<float> pointNKNSquaredDistance(K);
        if ( kdtree.nearestKSearch (p, K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0 ) {
            float distance = sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
            float radius = distance < 10.0f ? 0.1f : distance / 10.0f;
            if (pointNKNSquaredDistance[0] > radius)
                motion->push_back(p);
        }
    }
    auto elapsed1 = tim.toc();
    std::cout << "icp time = " << elapsed << " ms, "
              << "kdtree time = " << elapsed1 << " ms, "
              << "motion cloud size = " << motion->size() << std::endl;

    // Publish the data.
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_show(new pcl::PointCloud<pcl::PointXYZRGB>);
    cloud_show->header = cloud->header;
    for (auto p : motion->points) {
        pcl::PointXYZRGB tmp;
        tmp.x = p.x, tmp.y = p.y, tmp.z = p.z;
        tmp.r = 0, tmp.g = 255, tmp.b = 0;
        cloud_show->push_back(tmp);
    }

    pub.publish(cloud_show);

    // set pre frame
    pre_cloud = cloud;
}

int
main(int argc, char **argv) {
    // Initialize ROS
    ros::init(argc, argv, "test");
    ros::NodeHandle nh;

    // Create a ROS subscriber for the input point cloud
    ros::Subscriber sub = nh.subscribe("/velodyne_points", 1, cloud_cb);
    // ros::Subscriber sub = nh.subscribe("/kitti/velo/pointcloud", 1, cloud_cb);

    // Create a ROS publisher for the output point cloud
    pub = nh.advertise<pcl::PointCloud<pcl::PointXYZRGB>>("/tmp", 1);

    // Spin
    ros::spin();
}
