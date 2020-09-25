//
// Created by cdj on 2020/3/20.
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
#include <pcl/octree/octree.h>
#include <pcl/octree/octree_search.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/filters/voxel_grid.h>

#include <tuple>

struct PointLidar {
    PCL_ADD_POINT4D;                  // preferred way of adding a XYZ+padding
    float e;    // empty
    float o;    // occupied
    float u;    // unknown
    float conf;
    float cons;
    float unc;
    float theta;
    // PCL_MAKE_ALIGNED_OPERATOR_NEW     // make sure our new allocators are aligned
} EIGEN_ALIGN16;                    // enforce SSE padding for correct memory alignment

POINT_CLOUD_REGISTER_POINT_STRUCT (PointLidar, (float, x, x)(float, y, y)(float, z, z)
        (float, e, e)
        (float, o, o)
        (float, u, u)
        (float, conf, conf)
        (float, cons, cons)
        (float, unc, unc)
        (float, theta, theta)
)

class Node {
private:
    ros::NodeHandle nh;
    ros::Subscriber sub;
    ros::Publisher pub;
    pcl::console::TicToc tim;
    std::vector<pcl::PointCloud<PointLidar>::Ptr> cloud_queue;
    // pcl::PointCloud<PointLidar>::Ptr pre_cloud, now_cloud, nex_cloud;
public:
    Node() {
        // Create a ROS subscriber for the input point cloud
        // sub = nh.subscribe("/kitti/velo/pointcloud", 1, &Node::cloud_cb, this);  // kitti
        sub = nh.subscribe("/velodyne_points", 1, &Node::cloud_cb, this);
        // Create a ROS publisher for the output point cloud
        pub = nh.advertise<pcl::PointCloud<pcl::PointXYZRGB>>("/tmp", 1);
    };

    static void ground_remove(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_tmp(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::octree::OctreePointCloud<pcl::PointXYZ> octree(0.4f); // 设置八叉树的分辨率为 0.4 m
        octree.setInputCloud(cloud);
        octree.addPointsFromInputCloud();
        float h_hat = 0.0;
        float s = 0.09;
        for (auto it = octree.breadth_begin(); it != octree.breadth_end(); ++it) {    // 其实八叉树的树叉从某种意义上来说是没有东西的
            if (it.isBranchNode())  // 如果是树枝，就跳过
                continue;
            std::vector<int> indices;
            it.getLeafContainer().getPointIndices(indices); // 树叶不可能为空
            float H = cloud->points[indices[0]].z;  // 用 vector 中的第一个点来初始化最高和最低值
            float h = H;
            for (auto idx : indices) {
                float z_tmp = cloud->points[idx].z;
                if (z_tmp > H)
                    H = z_tmp;
                if (z_tmp < h)
                    h = z_tmp;
            }
            if (((H - h) < s) && (H < h_hat + s)) {   // 高低差根据论文设为 0.09 m
                // ground
            } else {
                h_hat = H;
                for (auto idx : indices) {
                    cloud_tmp->push_back(cloud->points[idx]);   // 非地面的
                }
            }
        }
        cloud = cloud_tmp;
    }

    static float f_theta(PointLidar &P, PointLidar &Q) {
        auto OP = P.getVector3fMap();
        auto OQ = Q.getVector3fMap();
        float theta = acos(OP.normalized().dot(OQ.normalized())) * 180 / M_PI;
        return exp(-0.5 * pow(theta / 2, 2));
    }

    static std::tuple<float, float, float> m(PointLidar P, PointLidar Q) {
        auto OQ = Q.getVector3fMap();
        auto OP = P.getVector3fMap();
        float OP_scalar_projection = OP.dot(OQ.normalized());
        float OQ_norm = OQ.norm();
        float e, o, u;
        if (OP_scalar_projection < 0) {
            e = 0.0;
            o = 0.0;
            std::cout << "todo\n";
        } else if (OP_scalar_projection < OQ_norm) {
            e = 1.0;
            o = 0.0;
        } else {
            e = 0.0;
            float r = OP_scalar_projection - OQ_norm;  // 因为在体素内选点，所以这个值比较小
            o = exp(-0.5 * pow(r, 2));
        }
        e = e * f_theta(P, Q);
        u = 1 - e - o;
        // std::cout << "test: " << e << " " << o << " " << u << "\n";
        return {e, o, u};
    }

    static void fusion(PointLidar &P, PointLidar Q) {
        float e1 = P.e, o1 = P.o, u1 = P.u;
        auto eou = m(P, Q);
        float e2 = std::get<0>(eou), o2 = std::get<1>(eou), u2 = std::get<2>(eou);
        if (u1 == 1) {  // 针对第一次计算，直接赋eou的计算值
            P.e = e2, P.o = o2, P.u = u2;
            return;
        }
        if (u2 == 1) {  // 当该次计算为 unknow 时，不进行dst（概率融合），直接返回
            return;
        }
        float K = o1 * e2 + e1 * o2;
        float e = (e1 * e2 + e1 * u2 + u1 * e2) / (1 - K);
        float o = (o1 * o2 + o1 * u2 + u1 * o2) / (1 - K);
        float u = (u1 * u2) / (1 - K);
        float sum = e + o + u;
        e /= sum;
        o /= sum;
        u /= sum;
        if (isnan(e) || (isnan(o) || isnan(u))) {
            P.e = round(P.e), P.o = round(P.o), P.u = round(P.u);
            return;
        }
        P.e = e, P.o = o, P.u = u;
        // std::cout << P.e << " " << P.o << " " << P.u << " " << sum << "\n";
    }

    // Neighbors within voxel search
    static std::vector<int>
    octreeVoxelSearch(pcl::octree::OctreePointCloudSearch<PointLidar> &octree, PointLidar &searchPoint) {
        std::vector<int> pointIdxVec;
        double min_x, min_y, min_z, max_x, max_y, max_z;
        octree.getBoundingBox(min_x, min_y, min_z, max_x, max_y, max_z);
        bool isInBox = (searchPoint.x >= min_x && searchPoint.x <= max_x)
                       && (searchPoint.y >= min_y && searchPoint.y <= max_y)
                       && (searchPoint.z >= min_z && searchPoint.z <= max_z);
        if (isInBox && octree.isVoxelOccupiedAtPoint(searchPoint))
            octree.voxelSearch(searchPoint, pointIdxVec);
        return pointIdxVec;
    }

    static void dst(const pcl::PointCloud<PointLidar>::Ptr &cloud) {
        // 要计算dst的点云的八叉树
        pcl::octree::OctreePointCloudSearch<PointLidar> octree(0.2f); // 设置八叉树的分辨率为 0.4 m
        octree.setInputCloud(cloud);
        octree.addPointsFromInputCloud();

        for (auto it = octree.leaf_begin(); it != octree.leaf_end(); it++) {
            std::vector<int> indices;
            it.getLeafContainer().getPointIndices(indices);
            for (auto idx : indices) {
                auto &P = cloud->points[idx];
                for (auto idx_ : indices) {
                    if (idx_ != idx) {
                        auto Q = cloud->points[idx_];
                        fusion(P, Q);
                    }
                }
            }
        }
    }

    static float compute_theta(pcl::PointXYZ &p) {
        auto OP = p.getVector3fMap();
        Eigen::Vector3f OZ(0, 0, 1);
        float theta = acos(OP.normalized().dot(OZ)) * 180 / M_PI;
        return theta;
    }

    static void
    consistency_assessment(pcl::PointCloud<PointLidar>::Ptr &cloud_pre, pcl::PointCloud<PointLidar>::Ptr &cloud_now) {
        pcl::octree::OctreePointCloudSearch<PointLidar> octree_pre(0.2f); // 设置八叉树的分辨率为 0.4 m
        octree_pre.setInputCloud(cloud_pre);
        octree_pre.addPointsFromInputCloud();
        pcl::octree::OctreePointCloudSearch<PointLidar> octree_now(0.2f); // 设置八叉树的分辨率为 0.4 m
        octree_now.setInputCloud(cloud_now);
        octree_now.addPointsFromInputCloud();

        for (auto it = octree_now.leaf_begin(); it != octree_now.leaf_end(); it++) {
            std::vector<int> indices;
            it.getLeafContainer().getPointIndices(indices);
            auto &P = cloud_now->points[indices[0]];  // 取体素中的第一个点来作为代表

            std::vector<int> pointIdxSearch = octreeVoxelSearch(octree_pre, P);
            if (!pointIdxSearch.empty()) {
                auto P_ = cloud_pre->points[pointIdxSearch[0]]; // 我觉得这一个体素内的所有点的eou应该都是一样的
                float E1 = P.e, O1 = P.o, U1 = P.u;
                float E2 = P_.e, O2 = P_.o, U2 = P_.u;
                P.conf = E1 * O2 + O1 * E2;
                P.cons = E1 * E2 + O1 * O2 + U1 * U2;
                P.unc = U1 * (E2 + O2) + U2 * (E1 + O1);
                // std::cout << P.conf << " " << P.cons << " " << P.unc << "\n";
                for (auto idx : indices) {  // 体素中其它的点全都按照第一个点的dst来进行赋值
                    auto &p = cloud_now->points[idx];  // idx instead of indices[idx]
                    p.conf = P.conf, p.cons = P.cons, p.unc = P.unc;
                }
            }
        }
    }

    // call back
    void cloud_cb(const sensor_msgs::PointCloud2ConstPtr &input) {
        tim.tic();
        pcl::PointCloud<pcl::PointXYZ>::Ptr _cloud(new pcl::PointCloud<pcl::PointXYZ>);
        // convert
        pcl::fromROSMsg(*input, *_cloud);
        // keep header
        auto header = _cloud->header;
        // ground remove
        ground_remove(_cloud);
        // change to PointLidar type
        pcl::PointCloud<PointLidar>::Ptr cloud(new pcl::PointCloud<PointLidar>);
        for (auto p : _cloud->points) {
            PointLidar tmp{};
            tmp.x = p.x, tmp.y = p.y, tmp.z = p.z;
            tmp.e = 0.0, tmp.o = 0.0, tmp.u = 1.0;
            tmp.cons = tmp.conf = tmp.unc = 0.0;
            tmp.theta = compute_theta(p);
            cloud->push_back(tmp);
        }

        cloud_queue.push_back(cloud);
        if (cloud_queue.size() < 2) {
            return;
        }

        auto cloud_pre = cloud_queue[0];
        auto cloud_now = cloud_queue[1];

        dst(cloud_pre);
        dst(cloud_now);
        consistency_assessment(cloud_pre, cloud_now);

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_show(new pcl::PointCloud<pcl::PointXYZRGB>);
        for (auto p : cloud_now->points) {
            if ((p.conf > p.cons) || (p.unc > p.cons)) {
                pcl::PointXYZRGB tmp;
                tmp.x = p.x, tmp.y = p.y, tmp.z = p.z;
                tmp.r = 0, tmp.g = 255, tmp.b = 0;
                cloud_show->push_back(tmp);
//                std::cout << p.conf << " " << p.cons << " " << p.unc << "\n";
            }
//            pcl::PointXYZRGB tmp;
//            tmp.x = p.x, tmp.y = p.y, tmp.z = p.z;
//            tmp.r = 255*p.cons, tmp.g = 255*p.conf, tmp.b = 255*p.unc;
//            cloud_show->push_back(tmp);
        }
        cloud_show->header = header;

        pub.publish(cloud_show);
        cloud_queue.erase(cloud_queue.begin());
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







