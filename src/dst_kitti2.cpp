//
// Created by cdj on 2020/3/29.
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
#include <pcl/registration/gicp.h>

#include <tuple>
#include <thread>
#include <mutex>
#include <atomic>

struct PointLidar {
    PCL_ADD_POINT4D;                  // preferred way of adding a XYZ+padding
    float e;    // empty
    float o;    // occupied
    float u;    // unknown
    float conf;
    float cons;
    float unc;
    int seq;
    float dist;
    // PCL_MAKE_ALIGNED_OPERATOR_NEW     // make sure our new allocators are aligned
} EIGEN_ALIGN16;                    // enforce SSE padding for correct memory alignment

POINT_CLOUD_REGISTER_POINT_STRUCT (PointLidar,
                                   (float, x, x)(float, y, y)(float, z, z)
                                           (float, e, e)
                                           (float, o, o)
                                           (float, u, u)
                                           (float, conf, conf)
                                           (float, cons, cons)
                                           (float, unc, unc)
                                           (float, seq, seq)
                                           (float, dist, dist)
)

class Node {
private:
    ros::NodeHandle nh;
    ros::Subscriber sub;
    ros::Publisher pub;
    pcl::console::TicToc tim;
    std::vector<pcl::PointCloud<PointLidar>::Ptr> cloud_queue, cloud_dsts;
    std::vector<PointLidar> Q_points;
    int seq;
public:
    Node() {
        // Create a ROS subscriber for the input point cloud
        sub = nh.subscribe("/kitti/velo/pointcloud", 1, &Node::cloud_cb, this);  // kitti
        // sub = nh.subscribe("/velodyne_points", 1, &Node::cloud_cb, this);
        // Create a ROS publisher for the output point cloud
        pub = nh.advertise<pcl::PointCloud<pcl::PointXYZRGB>>("/tmp", 1);
        seq = 0;
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
        return exp(-0.5 * pow(theta / 0.4, 2));
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
//            std::cout << "todo\n";
        } else if (OP_scalar_projection < OQ_norm) {
            e = 1.0;
            e = e * f_theta(P, Q);
            o = 0.0;
        } else {
            e = 0.0;
            float r = OP_scalar_projection - OQ_norm;  // 因为在体素内选点，所以这个值比较小
            o = exp(-0.5 * pow(r, 2));
        }
        u = 1 - e - o;
        // std::cout << "test: " << e << " " << o << " " << u << "\n";
        return {e, o, u};
    }

    static void fusion(PointLidar &P, PointLidar Q) {
        float e1 = P.e, o1 = P.o, u1 = P.u;
        auto eou = m(P, Q);
        float e2 = std::get<0>(eou), o2 = std::get<1>(eou), u2 = std::get<2>(eou);
        float K = o1 * e2 + e1 * o2;
        if (u1 == 1) {  // 针对第一次计算，直接赋eou的计算值
            P.e = e2, P.o = o2, P.u = u2;
            return;
        }
        if (K == 0)  //TODO 异常
            return;
        float e = (e1 * e2 + e1 * u2 + u1 * e2) / (1 - K);
        float o = (o1 * o2 + o1 * u2 + u1 * o2) / (1 - K);
        float u = (u1 * u2) / (1 - K);
        if (isnan(e) || (isnan(o) || isnan(u))) {
            return;
        }
        P.e = e, P.o = o, P.u = u;
        // std::cout << P.e << " " << P.o << " " << P.u << "\n";
    }

    static void
    robust(pcl::PointCloud<PointLidar>::Ptr &cloud) {
        if (cloud->size() == 0)
            return;
        PointLidar B = cloud->points[0]; // 最远的点
        for (auto &p : cloud->points)
            if (p.dist > B.dist)
                B = p;
        float r_sup = 0.8, r_inf = 0.6;
        for (auto &P : cloud->points) {
            float l = r_sup - (r_sup - r_inf) * (P.dist / B.dist);
            P.e = (P.e > P.o) && (P.e > P.u) ? l : 0;
            P.o = (P.o > P.e) && (P.o > P.u) ? l : 0;
            P.u = 1 - P.e - P.o;
        }
    }

    // 这次计算 dst 的时候，以参考帧的点来进行计算，先进行　gicp
    void
    dst(pcl::PointCloud<PointLidar>::Ptr &cloud, pcl::PointCloud<PointLidar>::Ptr &cloud_ref) {

        this->transform(cloud_ref, cloud);  // 先将 cloud_ref 的坐标系转换到 cloud 下
        this->set_Q_points(cloud_ref);

        pcl::octree::OctreePointCloudSearch<PointLidar> octree(1.0f); // 设置八叉树的分辨率为 0.4 m
        octree.setInputCloud(cloud);
        octree.addPointsFromInputCloud();

        for (auto it = octree.leaf_begin(); it != octree.leaf_end(); it++) {
            std::vector<int> indices;
            it.getLeafContainer().getPointIndices(indices);
            auto P = cloud->points[it.getLeafContainer().getPointIndex()];  // 取体素中的一个点作为代表
            for (auto Q : this->Q_points) { // 仅计算体素中的一个点的dst
                fusion(P, Q);
            }
            for (auto idx : indices) {  // 体素中其它的点全都按照第一个点的dst来进行赋值
                auto &p = cloud->points[idx];  // idx instead of indices[idx]
                p.e = P.e, p.o = P.o, p.u = P.u;
                // std::cout << p.e << " " << p.o << " " << p.u << "\n";
            }
        }
        robust(cloud); // ROS-16 论文提供的方法，用于提高鲁棒性
    }

    static void consistency_relations(PointLidar &P, PointLidar &P_) {
        float E1 = P.e, O1 = P.o, U1 = P.u;
        float E2 = P_.e, O2 = P_.o, U2 = P_.u;
        P.conf = E1 * O2 + O1 * E2;
        P.cons = E1 * E2 + O1 * O2 + U1 * U2;
        P.unc = U1 * (E2 + O2) + U2 * (O1 + O1);
    }

    static void avg_eou(PointLidar &P, PointLidar Q) {
        // std::cout << P.e << " " << P.o << " " << P.u << "\n";
        P.e = (P.e + Q.e) / 2;
        P.o = (P.o + Q.o) / 2;
        P.u = (P.u + Q.u) / 2;
    }

    static void
    consistency_assessment(pcl::PointCloud<PointLidar>::Ptr &cloud_1, pcl::PointCloud<PointLidar>::Ptr &cloud_2) {
        // 这里的一致性评估，是对同一点云进行的，它分别对两个参考帧计算了 dst
        // 一致性评估的结果就保存在 cloud_1 里吧
        for (auto idx = 0; idx < cloud_1->size(); idx++) {
            auto &p = cloud_1->points[idx];
            auto &p_ = cloud_2->points[idx];
            consistency_relations(p, p_);
        }

    }

    void set_Q_points(pcl::PointCloud<PointLidar>::Ptr &cloud) {
        if (!this->Q_points.empty())
            this->Q_points.clear();
        pcl::octree::OctreePointCloudSearch<PointLidar> octree(1.5f); // 设置八叉树的分辨率
        octree.setInputCloud(cloud);
        octree.addPointsFromInputCloud();
        for (auto it = octree.leaf_begin(); it != octree.leaf_end(); it++) {
            int idx = it.getLeafContainer().getPointIndex();
            auto Q = cloud->points[idx];
            if ((Q.dist > 1.0f)) // (Q.dist < 30.0f) &&
                this->Q_points.push_back(Q);
        }
    }

    Eigen::Matrix<float, 4, 4>
    icp(pcl::PointCloud<PointLidar>::Ptr &cloud_from, pcl::PointCloud<PointLidar>::Ptr &cloud_to) {
        pcl::GeneralizedIterativeClosestPoint<PointLidar, PointLidar> icp; // 使用的是 gicp
        icp.setInputSource(cloud_from);
        icp.setInputTarget(cloud_to);
        pcl::PointCloud<PointLidar> Final;
        icp.align(Final);
        return icp.getFinalTransformation();
    }

    // 将　cloud_from 坐标系转换到 cloud_to 的坐标系下
    void
    transform(pcl::PointCloud<PointLidar>::Ptr &cloud_from, pcl::PointCloud<PointLidar>::Ptr &cloud_to) {
        // 从　cloud_from 到　cloud_to : cloud_to = T * cloud_from
        auto T = icp(cloud_from, cloud_to);
        for (auto &p : cloud_from->points) {
            auto p_new = T * p.getVector4fMap();
            p.x = p_new.x();
            p.y = p_new.y();
            p.z = p_new.z();
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
        header.seq = this->seq++;
        // ground remove
        ground_remove(_cloud);
        // change to PointLidar type
        pcl::PointCloud<PointLidar>::Ptr cloud(new pcl::PointCloud<PointLidar>);
        for (auto p : _cloud->points) {
            float dist = p.getVector3fMap().norm();  // 点到中心的距离
            if ((dist > 30))
                continue;
            PointLidar tmp{};
            tmp.x = p.x, tmp.y = p.y, tmp.z = p.z;
            tmp.e = 0.0, tmp.o = 0.0, tmp.u = 1.0;  // e o u 初始化
            tmp.cons = 0.0, tmp.conf = 0.0, tmp.unc = 1.0; // cons conf unc 初始化
            tmp.seq = header.seq;   // 用时间来标
            tmp.dist = dist;
            cloud->push_back(tmp);
        }

        cloud_queue.push_back(cloud);
        if (cloud_queue.size() < 11) {
            return;
        }

        auto cloud_k = cloud_queue[cloud_queue.size() / 2];  // 想不出合适的名字了，这个点云就是要计算 dst 的点云

        this->cloud_dsts.clear();
        for (auto cloud_tmp : this->cloud_queue) {
            if (cloud_tmp == cloud_k)
                continue;
            pcl::PointCloud<PointLidar>::Ptr cloud_copy(new pcl::PointCloud<PointLidar>);
            pcl::copyPointCloud(*cloud_k, *cloud_copy);  // copy == k
            this->dst(cloud_copy, cloud_tmp);
            this->cloud_dsts.push_back(cloud_copy);  // 这些 cloud_tmp 中都包含了和相对帧计算了 dst 之后的 eou
            std::cout << ".";
        }
        std::cout << std::endl;


        // 一致性评估
        auto cloud_result = *this->cloud_dsts.begin();
        for (auto it = this->cloud_dsts.begin() + 1; it != this->cloud_dsts.end(); it++) {
            consistency_assessment(cloud_result, *it);
        }
        for (auto cloud_tmp : this->cloud_dsts)
            consistency_assessment(cloud_k, cloud_tmp);

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_show(new pcl::PointCloud<pcl::PointXYZRGB>);
        for (auto p : cloud_result->points) {
            // std::cout << p.conf << " " << p.cons << " " << p.unc << std::endl;
            if ((p.conf > p.cons) || (p.unc > p.cons)) {
                pcl::PointXYZRGB tmp;
                tmp.x = p.x, tmp.y = p.y, tmp.z = p.z;
                tmp.r = 0, tmp.g = 255, tmp.b = 0;
                cloud_show->push_back(tmp);
            }
        }
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


