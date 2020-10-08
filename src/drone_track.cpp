//
// Created by cdj on 2020/10/7.
//
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <sensor_msgs/PointCloud2.h>

#include "pcl_ros/point_cloud.h"  // fromROSMsg以及pcl中的各种点云

// #include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/registration/icp.h>
#include <pcl/common/transforms.h>
#include <pcl/console/time.h>   // TicToc
#include <visualization_msgs/Marker.h>

using namespace std;
using namespace sensor_msgs;
using namespace message_filters;

ros::Publisher pub;  // 融合点云发布

pcl::console::TicToc tim;  // 计时器

Eigen::Vector3f drone_coord (-9.45, -4.82, -0.71);

void callback(const PointCloud2ConstPtr &merge_data) {
    tim.tic();
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::fromROSMsg(*merge_data, *cloud);

    pcl::KdTreeFLANN<pcl::PointXYZI> kdtree;
    kdtree.setInputCloud (cloud);
    pcl::PointXYZI searchPoint;
    searchPoint.x = drone_coord.x(), searchPoint.y = drone_coord.y(), searchPoint.z = drone_coord.z();

    // Neighbors within radius search
    std::vector<int> pointIdxRadiusSearch;
    std::vector<float> pointRadiusSquaredDistance;
    float radius = 0.5;

    if ( kdtree.radiusSearch (searchPoint, radius, pointIdxRadiusSearch, pointRadiusSquaredDistance) > 0 ) {
        float x = 0.0, y = 0.0, z = 0.0;
        // float minx = 100, maxx = -100, miny = 100, maxy = -100, minz = 100, maxz = -100;
        for (int idx : pointIdxRadiusSearch) {
            auto point = cloud->points[idx];
            x += point.x;
            y += point.y;
            z += point.z;
        }
        x /= pointIdxRadiusSearch.size();
        y /= pointIdxRadiusSearch.size();
        z /= pointIdxRadiusSearch.size();
        drone_coord << x, y, z;
        cout << "无人机坐标：" << drone_coord.transpose() << endl;

        visualization_msgs::Marker marker;
        marker.header.frame_id = "merge";
        marker.header.stamp = ros::Time();
        marker.id = 0;
        marker.type = visualization_msgs::Marker::CUBE;
        marker.action = visualization_msgs::Marker::ADD;
        marker.pose.position.x = x;
        marker.pose.position.y = y;
        marker.pose.position.z = z;

        marker.pose.orientation.x = 0.0;
        marker.pose.orientation.y = 0.0;
        marker.pose.orientation.z = 0.0;
        marker.pose.orientation.w = 1.0;

        marker.scale.x = 0.5;
        marker.scale.y = 0.5;
        marker.scale.z = 0.5;

        marker.color.a = 1.0; // Don't forget to set the alpha!
        marker.color.r = 0.0;
        marker.color.g = 1.0;
        marker.color.b = 0.0;
        pub.publish( marker );
    }

    tim.toc_print();
}

int main(int argc, char **argv) {
    ros::init(argc, argv, "drone_track");

    ros::NodeHandle nh;

    pub = nh.advertise<visualization_msgs::Marker>( "drone_marker", 0);

    ros::Subscriber sub = nh.subscribe("/merge_points", 1, callback);

    ros::spin();

    return 0;
}
