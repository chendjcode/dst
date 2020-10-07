//
// Created by cdj on 2020/9/28.
//
#define PCL_NO_PRECOMPILE

#include <ros/ros.h>
// PCL specific includes
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
// #include <pcl/conversions.h>
#include "pcl_ros/point_cloud.h"
#include <pcl/console/time.h>   // TicToc
#include <pcl/octree/octree_search.h>

#include <tuple>
#include <pcl/ModelCoefficients.h>
#include <pcl/segmentation/extract_clusters.h>

#include <pcl/io/pcd_io.h>

using namespace std;

class Node {
private:
    ros::NodeHandle nh;
    ros::Subscriber sub;
    ros::Publisher pub;
    pcl::console::TicToc tim;
    float imax = 0.0;  // 激光强度的最大值
public:
    Node() {
        // Create a ROS subscriber for the input point cloud
        // sub = nh.subscribe("/kitti/velo/pointcloud", 1, &Node::cloud_cb, this);  // kitti
        sub = nh.subscribe("/velodyne_points", 1, &Node::cloud_cb, this);
        // Create a ROS publisher for the output point cloud
        pub = nh.advertise<pcl::PointCloud<pcl::PointXYZRGB>>("/cluster_points", 1);
    };

    // 判断文件或者文件夹是否存在
    bool isPathExist(const std::string &s) {
        struct stat buffer;
        return (stat (s.c_str(), &buffer) == 0);
    }

    // call back
    void cloud_cb(const sensor_msgs::PointCloud2ConstPtr &input) {
        tim.tic();
        pcl::PointCloud<pcl::PointXYZI>::Ptr _cloud(new pcl::PointCloud<pcl::PointXYZI>);
        // convert
        pcl::fromROSMsg(*input, *_cloud);
        // keep header
        auto header = _cloud->header;

        // 聚类
        // Creating the KdTree object for the search method of the extraction
        pcl::search::KdTree<pcl::PointXYZI>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZI>);
        tree->setInputCloud(_cloud);

        std::vector<pcl::PointIndices> cluster_indices;
        pcl::EuclideanClusterExtraction<pcl::PointXYZI> ec;
        ec.setClusterTolerance(0.2); // 聚类 tolerance 距离，单位 m
        ec.setMinClusterSize(15); // 聚类点最小数量
        ec.setMaxClusterSize(500);
        ec.setSearchMethod(tree);
        ec.setInputCloud(_cloud);
        ec.extract(cluster_indices);

        cout << "聚类数量：" << cluster_indices.size() << endl;

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_show(new pcl::PointCloud<pcl::PointXYZRGB>);
        string pcd_dir = "/home/cdj/Data/PCD/" + to_string(header.stamp);
        if (!isPathExist(pcd_dir))
            mkdir(pcd_dir.c_str(), 0777);
        for (const auto &cluster : cluster_indices) {  // 每次遍历一个聚类
            float rgb = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);  // 随机生成颜色
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_tmp(new pcl::PointCloud<pcl::PointXYZRGB>);
            for (auto idx : cluster.indices) {
                auto p = _cloud->points[idx];
                pcl::PointXYZRGB tmp_rgb;
                tmp_rgb.x = p.x, tmp_rgb.y = p.y, tmp_rgb.z = p.z;
                tmp_rgb.r = tmp_rgb.g = tmp_rgb.b = p.intensity;  // PCL中intensity的范围为[0, 255]，Open3D中intensity的范围是[0, 1]
                cloud_show->push_back(tmp_rgb);

                cloud_tmp->push_back(tmp_rgb);
                imax = max(imax, p.intensity);
            }
            string cloud_tmp_name = pcd_dir + "/" + to_string(rgb) + ".pcd";

            // 采集
            // pcl::io::savePCDFileASCII (cloud_tmp_name, *cloud_tmp);

            // 每一簇都发布出去
            cloud_tmp->header = header;
            pub.publish(cloud_tmp);
        }

        cloud_show->header = header;

        // pub.publish(cloud_show);
        tim.toc_print();
    }
};

int main(int argc, char **argv) {
    // Initialize ROS
    ros::init(argc, argv, "plane");

    Node node;

    // Spin
    ros::spin();
}

