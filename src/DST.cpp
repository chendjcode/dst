//
// Created by cdj on 2020/2/20.
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
#include <vector>
#include <pcl/pcl_macros.h>
#include <pcl/filters/voxel_grid.h>
#include <random>

struct PointLidar {
    PCL_ADD_POINT4D;                  // preferred way of adding a XYZ+padding
    float e;    // empty
    float o;    // occupied
    float u;    // unknow
    bool first;
    float conf;
    float cons;
    float unc;
    // PCL_MAKE_ALIGNED_OPERATOR_NEW     // make sure our new allocators are aligned
} EIGEN_ALIGN16;                    // enforce SSE padding for correct memory alignment

POINT_CLOUD_REGISTER_POINT_STRUCT (PointLidar, (float, x, x)(float, y, y)(float, z, z)
        (float, e, e)
        (float, o, o)
        (float, u, u)
        (float, first, first)
        (float, conf, conf)
        (float, cons, cons)
        (float, unc, unc)
)

class Node {
private:
    ros::NodeHandle nh;
    ros::Subscriber sub;
    ros::Publisher pub;
    pcl::PointCloud<PointLidar>::Ptr pre_cloud;
    pcl::console::TicToc tim;
    float resolution;
    // Instantiate octree-based point cloud change detection class
    pcl::octree::OctreePointCloudSearch<PointLidar> octree;
    float lambda_theta;  // the angular resolution of the sensor
    float downSampleLeafSize;
public:
    Node() : resolution(1.0f), octree(resolution), lambda_theta(2.0f), downSampleLeafSize(2) {
        // Create a ROS subscriber for the input point cloud
        // sub = nh.subscribe("/kitti/velo/pointcloud", 1, &Node::cloud_cb, this);
        sub = nh.subscribe("/velodyne_points", 1, &Node::cloud_cb, this);
        // Create a ROS publisher for the output point cloud
        pub = nh.advertise<pcl::PointCloud<pcl::PointXYZRGB>>("/tmp", 1);
    };

    float get_f_theta(PointLidar Q, PointLidar P) {
        auto OQ = Q.getVector3fMap();
        auto OP = P.getVector3fMap();
        float theta = acos(OQ.dot(OP) / (OP.norm() * OQ.norm()));
        return exp(-theta * theta / (2 * lambda_theta * lambda_theta));
    }

    float get_F(PointLidar Q) {
        float x = Q.getVector3fMap().norm();
        float theta_m = 0.0002;
        float theta_r = 0.1;
        float theta_2 = theta_m * theta_m + theta_r * theta_r;
        float F = exp(-x * x / (2 * theta_2)) / sqrt(2 * M_PI * theta_2);
        // std::cout << F << std::endl;
        return F;
    }

    // the degrees of belief for the three possible labels
    void m(PointLidar Q, const pcl::PointCloud<PointLidar>::Ptr &cloud) {
        auto OQ_normalized = Q.getVector3fMap().normalized();
        float OQ_len = Q.getVector3fMap().norm();
        // float F = get_F(Q);
        float F = 1.0f;
        for (auto &P : cloud->points) {
            float e1 = P.e, o1 = P.o, u1 = P.u;
            float e2 = 0, o2 = 0, u2 = 1;

            float OP__len = P.getVector3fMap().dot(OQ_normalized);  // OP__len = OP' length
            if (OQ_len > OP__len) {  // if Q is behind P'
                e2 = 1;
                o2 = 0;
            } else {
                e2 = 0;
                float r = OP__len - OQ_len;
                o2 = exp(-(r * r) / 2);
            }
            float f_theta = get_f_theta(Q, P);
            e2 = f_theta * e2 * F;
            o2 = o2 * F;
            u2 = 1 - e2 - o2;
            // std::cout << e2 << " " << o2 << " " << u2 << std::endl;

            if (P.first) {
                P.e = e2;
                P.o = o2;
                P.u = u2;
                P.first = false;
            } else {
                float K = o1 * e2 + e1 * o2;
                P.e = (e1 * e2 + e1 * u2 + u1 * e2) / (1 - K);
                P.o = (o1 * o2 + o1 * u2 + u1 * o2) / (1 - K);
                P.u = (u1 * u2) / (1 - K);
            }
        }
    }

    // call back
    void cloud_cb(const sensor_msgs::PointCloud2ConstPtr &input) {
        tim.tic();
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<PointLidar>::Ptr cloud_lidar(new pcl::PointCloud<PointLidar>);

        // convert
        pcl::fromROSMsg(*input, *cloud);

        for (auto p : cloud->points) {
            PointLidar tmp{};
            tmp.x = p.x, tmp.y = p.y, tmp.z = p.z, tmp.e = 0, tmp.o = 0, tmp.u = 1, tmp.first = true;
            cloud_lidar->push_back(tmp);
        }

        // DST
        for (int i = 0; i < 4 * 16; i++) {
            auto idx = rand() % cloud_lidar->size();
            auto Q = cloud_lidar->points[idx];
            m(Q, cloud_lidar);
        }

        // first frame process
        if (!pre_cloud) {
            pre_cloud = cloud_lidar;
            return;
        }

        // Add points from cloudB to octree
        octree.setInputCloud(cloud_lidar);
        octree.addPointsFromInputCloud();

        for (auto p : pre_cloud->points) {
            // Neighbors within voxel search
            std::vector<int> pointIdxVec;
            if (octree.voxelSearch(p, pointIdxVec)) {
                for (int i : pointIdxVec) {
                    PointLidar &point = cloud_lidar->points[i];
                    point.conf = p.e * point.o + p.o * point.e;
                    point.cons = p.e * point.e + p.o * point.o + p.u * point.u;
                    point.unc = p.u * (point.e + point.o) + point.u * (p.e + p.o);
                }
            }

        }
//        for (auto p : cloud_lidar->points) {
//            std::cout << p.conf << " " << p.cons << " " << p.unc << std::endl;
//        }

        // Publish the data.
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_show(new pcl::PointCloud<pcl::PointXYZRGB>);
        cloud_show->header = cloud->header;
        for (auto p : cloud_lidar->points) {
            pcl::PointXYZRGB tmp;
            tmp.x = p.x, tmp.y = p.y, tmp.z = p.z;
            tmp.r = p.conf * 255, tmp.g = p.cons * 255, tmp.b = p.unc * 255;
            if (p.conf > p.cons || p.unc > p.cons)
                cloud_show->push_back(tmp);
        }

        pub.publish(cloud_show);

        // set pre frame
        pre_cloud = cloud_lidar;
        tim.toc_print();
    }
};

int main(int argc, char **argv) {
    // Initialize ROS
    ros::init(argc, argv, "test");

    Node node;

    // Spin
    ros::spin();
}

