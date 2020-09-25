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
#include <pcl/octree/octree_search.h>


ros::Publisher pub;
float resolution = 128.0f;
pcl::octree::OctreePointCloudSearch<pcl::PointXYZ> octree (resolution);

void
cloud_cb(const sensor_msgs::PointCloud2ConstPtr &input) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr motion(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(*input, *cloud);

    // std::cout << "after " << cloud_filtered->size() << std::endl;
    if (!octree.getInputCloud()) {
        octree.setInputCloud (cloud);
        octree.addPointsFromInputCloud ();
        return;
    }
    for (auto p : cloud->points) {
        std::vector<int> pointIdxRadiusSearch;
        std::vector<float> pointRadiusSquaredDistance;
        float radius = 0.1f;
        if ( octree.radiusSearch (p, radius, pointIdxRadiusSearch, pointRadiusSquaredDistance) == 0 ) {
            motion->push_back(p);
        }
    }
    std::cout << "motion cloud size = " << motion->size() << std::endl;

    // Publish the data.
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_show(new pcl::PointCloud<pcl::PointXYZRGB>);
    cloud_show->header = cloud->header;
    for (auto p : motion->points) {
        pcl::PointXYZRGB tmp;
        tmp.x = p.x, tmp.y = p.y, tmp.z = p.z;
        tmp.r = 0, tmp.g = 255, tmp.b = 0;
        cloud_show->push_back(tmp);
    }

    if (cloud_show->size() < 500)
        pub.publish(cloud_show);

    std::cout << "cloud size = " << cloud->size() << std::endl;
    octree.setInputCloud (cloud);
    octree.addPointsFromInputCloud ();
}

int
main(int argc, char **argv) {
    // Initialize ROS
    ros::init(argc, argv, "test");
    ros::NodeHandle nh;

    // Create a ROS subscriber for the input point cloud
    ros::Subscriber sub = nh.subscribe("/velodyne_points", 1, cloud_cb);

    // Create a ROS publisher for the output point cloud
    pub = nh.advertise<pcl::PointCloud<pcl::PointXYZRGB>>("/tmp", 1);

    // Spin
    ros::spin();
}
