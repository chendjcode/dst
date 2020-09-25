//
// Created by cdj on 2020/5/4.
//
//
// Created by cdj on 2020/4/6.
//
/*
 * 尽量按照IROS-16论文来，使用两帧来计算 DST， 使用同一帧的不同的 DST 来进行一致性评估；
 * 希望这次可以成功；
 * */
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
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/registration/gicp.h>

#include <tuple>
#include <thread>
#include <mutex>
#include <atomic>

using namespace  std;

struct PointLidar {
    PCL_ADD_POINT4D;                  // preferred way of adding a XYZ+padding
    double e;    // empty
    double o;    // occupied
    double u;    // unknown
    double conf;
    double cons;
    double unc;
    double dist;
    double pox;
    double poy;
    double poz;
    int stamp;
    // PCL_MAKE_ALIGNED_OPERATOR_NEW     // make sure our new allocators are aligned
} EIGEN_ALIGN16;                    // enforce SSE padding for correct memory alignment

POINT_CLOUD_REGISTER_POINT_STRUCT (PointLidar,
                                   (double, x, x)(double, y, y)(double, z, z)
                                           (double, e, e)
                                           (double, o, o)
                                           (double, u, u)
                                           (double, conf, conf)
                                           (double, cons, cons)
                                           (double, unc, unc)
                                           (double, dist, dist)
                                           (double, pox, pox)
                                           (double, poy, poy)
                                           (double, poz, poz)
                                           (int, stamp, stamp)
)

class Node {
private:
    ros::NodeHandle nh;
    ros::Subscriber sub;
    ros::Publisher pub;
    pcl::console::TicToc tim;  // 计时器
    std::vector<pcl::PointCloud<PointLidar>::Ptr> cloud_queue; // 保存了连续的点云
    pcl::PointCloud<PointLidar>::Ptr cloud_global;  // 全局点云
    double ground_remove_octree_resolution;  // 用于去除地面的八叉树的分辨率
    double lambda_theta; // 雷达的垂直分辨率
    int stamp; // 用于标记点云
    int frame_count;
    int Q_count;
public:
    Node() {
        // Create a ROS subscriber for the input point cloud
        // sub = nh.subscribe("/kitti/velo/pointcloud", 1, &Node::cloud_cb, this);  // kitti
        sub = nh.subscribe("/velodyne_points", 1, &Node::cloud_cb, this);
        lambda_theta = 0.4; // 雷达的垂直分辨率 64-0.4, 16-2.0
        // Create a ROS publisher for the output point cloud
        pub = nh.advertise<pcl::PointCloud<pcl::PointXYZRGB>>("/tmp", 1);
        cloud_global = boost::make_shared<pcl::PointCloud<PointLidar>>();
        frame_count = 5; // 连续的帧数
        Q_count = 3;
        stamp = 0; // 用于标记当前帧的 id
        ground_remove_octree_resolution = 1.0;
    };

    // IROS-16，地面去除，据论文说是 novel 的
    void
    ground_remove(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_tmp(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ground(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::octree::OctreePointCloud<pcl::PointXYZ> octree(1.6);
        octree.setInputCloud(cloud);
        octree.addPointsFromInputCloud();
        double h_hat = 0.0;
        double s = 0.09; // 初始值 0.09
        for (auto it = octree.leaf_begin(); it != octree.leaf_end(); ++it) {
            std::vector<int> indices;
            it.getLeafContainer().getPointIndices(indices); // 树叶不可能为空
            auto p = cloud->points[it.getLeafContainer().getPointIndex()]; // 叶子里的一个点（源代码里好像就是 vector 的最后一个值。。。）
            auto r = p.getVector3fMap().norm(); // 点的半径
            if (r > 20.0 || r < 5.0)
                continue;
            double H = p.z;  // 初始化最高和最低值
            double h = H;
            for (auto idx : indices) {
                double z_tmp = cloud->points[idx].z;
                if (z_tmp > H)
                    H = z_tmp;
                if (z_tmp < h)
                    h = z_tmp;
            }
            if (((H - h) < s) && (H < h_hat + s)) {   // 高低差根据论文设为 0.09 m
                // ground
                for (auto idx : indices) {
                    if (cloud->points[idx].z < -1)
                        cloud_ground->push_back(cloud->points[idx]);   // 非地面的
                }
            } else {
                h_hat = H;
                for (auto idx : indices) {
                    cloud_tmp->push_back(cloud->points[idx]);   // 非地面的
                }
            }
        }
        // cloud = cloud_tmp;
        cloud = cloud_ground;
    }

    // call back
    void cloud_cb(const sensor_msgs::PointCloud2ConstPtr &input) {
        tim.tic();
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromROSMsg(*input, *cloud_xyz);
        auto header = cloud_xyz->header;
        this->stamp = header.stamp % 10000;
        // ground remove，顺带着对半径方面做了过滤
        ground_remove(cloud_xyz);

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_show(new pcl::PointCloud<pcl::PointXYZRGB>);
        double height = 0.0;
        for (auto p : cloud_xyz->points) {
            pcl::PointXYZRGB tmp;
            tmp.x = p.x, tmp.y = p.y, tmp.z = p.z;
            tmp.r = 0, tmp.g = 255, tmp.b = 0;
            cloud_show->push_back(tmp);
            height += tmp.z;
        }
        height /= cloud_show->size();
        cout << height << endl;

        cloud_show->header = header;

        pub.publish(cloud_show);
        this->cloud_queue.clear();
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

