//
// Created by cdj on 2020/9/22.
//
#include <ros/ros.h>
// PCL specific includes
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
// #include <pcl/conversions.h>
#include "pcl_ros/point_cloud.h"
#include <pcl/octree/octree_pointcloud_changedetector.h>
#include <pcl/console/time.h>   // TicToc

using namespace std;


pcl::console::TicToc tim;
ros::Publisher pub;
pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_pre;

/*
 * 帧差，KDTree
 */
void
cloud_cb(const sensor_msgs::PointCloud2ConstPtr &input) {
    tim.tic();  // 开始记录时间
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_now(new pcl::PointCloud<pcl::PointXYZ>);  // 当前帧
    // convert：将ros格式的点云转成pcl格式的点云
    pcl::fromROSMsg(*input, *cloud_now);

    // 第一帧的处理
    if (!cloud_pre) {
        cloud_pre = cloud_now;
        return;
    }

    // Octree resolution - side length of octree voxels
    float resolution = 0.2f;

    // Instantiate octree-based point cloud change detection class
    pcl::octree::OctreePointCloudChangeDetector<pcl::PointXYZ> octree (resolution);

    // Add points from cloudA to octree
    octree.setInputCloud (cloud_pre);
    octree.addPointsFromInputCloud ();

    // Switch octree buffers: This resets octree but keeps previous tree structure in memory.
    octree.switchBuffers ();

    // Add points from cloudB to octree
    octree.setInputCloud (cloud_now);
    octree.addPointsFromInputCloud ();

    std::vector<int> newPointIdxVector;

    // Get vector of point indices from octree voxels which did not exist in previous buffer
    octree.getPointIndicesFromNewVoxels (newPointIdxVector);

    // Publish the data.
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_show(new pcl::PointCloud<pcl::PointXYZRGB>);
    cloud_show->header = cloud_now->header;
    for (int idx : newPointIdxVector) {
        pcl::PointXYZRGB tmp;
        auto p = cloud_now->points[idx];
        tmp.x = p.x, tmp.y = p.y, tmp.z = p.z;
        tmp.r = 0, tmp.g = 255, tmp.b = 0;
        cloud_show->push_back(tmp);
    }

    pub.publish(cloud_show);

    // set pre frame
    cloud_pre = cloud_now;

    cout << "单次处理时间：";
    tim.toc_print();
}

int
main(int argc, char **argv) {
    // Initialize ROS
    ros::init(argc, argv, "test");
    ros::NodeHandle nh;

    // Create a ROS subscriber for the input point cloud
    ros::Subscriber sub = nh.subscribe("/velodyne_points", 1, cloud_cb);

    // Create a ROS publisher for the output point cloud
    pub = nh.advertise<pcl::PointCloud<pcl::PointXYZRGB>>("/motion_points", 1);

    // Spin
    ros::spin();
}
