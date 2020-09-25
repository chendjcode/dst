//
// Created by cdj on 2020/3/30.
//
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
#include <pcl/kdtree/kdtree_flann.h>
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
    float pox;
    float poy;
    float poz;
    int id;
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
                                           (float, pox, pox)
                                           (float, poy, poy)
                                           (float, poz, poz)
                                           (int, id, id)
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
        if (cloud->empty())
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
    dst(pcl::PointCloud<PointLidar>::Ptr &cloud) {
        pcl::octree::OctreePointCloud<PointLidar> octree(1.0f); // 设置八叉树的分辨率为 0.4 m
        octree.setInputCloud(cloud);
        octree.addPointsFromInputCloud();

        for (auto it = octree.leaf_begin(); it != octree.leaf_end(); it++) {
            std::vector<int> indices;
            it.getLeafContainer().getPointIndices(indices);
            int idx = it.getLeafContainer().getPointIndex();  // 取体素中的一个点作为代表
            auto P = cloud->points[idx];  // 取体素中的一个点作为代表
            for (auto Q : this->Q_points) { // 仅计算体素中的一个点的dst
                fusion(P, Q);
            }
            for (auto idx_ : indices) {  // 体素中其它的点全都按照第一个点的dst来进行赋值
                auto &p = cloud->points[idx_];  // idx instead of indices[idx]t
                p.e = P.e, p.o = P.o, p.u = P.u;
                // std::cout << p.e << " " << p.o << " " << p.u << "\n";
            }
        }
        robust(cloud); // IROS-16
    }

    static void consistency_relations(PointLidar &P, PointLidar &P_) {
        float E1 = P.e, O1 = P.o, U1 = P.u;
        float E2 = P_.e, O2 = P_.o, U2 = P_.u;
        P.conf = E1 * O2 + O1 * E2;
        P.cons = E1 * E2 + O1 * O2 + U1 * U2;
        P.unc = U1 * (E2 + O2) + U2 * (O1 + O1);
    }

    // 变成适合进行 kdtree 搜索的形式
    static void cloud_change_to(pcl::PointCloud<PointLidar>::Ptr &cloud) {
        for (auto &p : cloud->points) {
            point_change_to(p);
        }
    }

    // 变回来
    static void cloud_change_back(pcl::PointCloud<PointLidar>::Ptr &cloud) {
        for (auto &p : cloud->points) {
            point_change_back(p);
        }
    }

    static void point_change_to(PointLidar &p) {
        float x, y, z;
        x = p.x, y = p.y, z = p.z;
        p.x = p.pox, p.y = p.poy, p.z = p.poz;
        p.pox = x, p.poy = y, p.poz = z;
    }

    static void point_change_back(PointLidar &p) {
        float x, y, z;
        x = p.pox, y = p.poy, z = p.poz;
        p.pox = p.x, p.poy = p.y, p.poz = p.z;
        p.x = x, p.y = y, p.z = z;
    }

    std::vector<int>
    kdtree_search(pcl::KdTreeFLANN<PointLidar> &kdtree, PointLidar &p) {
        point_change_to(p);
        // Neighbors within radius search
        std::vector<int> pointIdxRadiusSearch;
        std::vector<float> pointRadiusSquaredDistance;
        float radius = 5.0; // 5 度
        kdtree.radiusSearch (p, radius, pointIdxRadiusSearch, pointRadiusSquaredDistance);
        point_change_back(p); // 再变回来
        return pointIdxRadiusSearch;
    }

    void // k 是要计算的帧，ref 是参考帧
    consistency_assessment(pcl::PointCloud<PointLidar>::Ptr &cloud_k, pcl::PointCloud<PointLidar>::Ptr &cloud_ref) {
        pcl::octree::OctreePointCloud<PointLidar> octree(1.0f); // 设置八叉树的分辨率为 0.4 m
        octree.setInputCloud(cloud_k);
        octree.addPointsFromInputCloud();

        cloud_change_to(cloud_ref);
        pcl::KdTreeFLANN<PointLidar> kdtree;
        kdtree.setInputCloud(cloud_ref);

        for (auto it = octree.leaf_begin(); it != octree.leaf_end(); it++) {
            std::vector<int> indices;
            it.getLeafContainer().getPointIndices(indices);
            int idx = it.getLeafContainer().getPointIndex();  // 取体素中的一个点作为代表
            auto P = cloud_k->points[idx];  // 取体素中的一个点作为代表
            auto idx_s = kdtree_search(kdtree, P); // 返回一定半径内的点
            for (auto idx_ : idx_s) {
                auto P_ = cloud_ref->points[idx_];
                consistency_relations(P, P_);  // 这里的 P_ 是在参考帧最近的一个点
            }
            for (auto idx_ : indices) {  // 体素中其它的点全都按照第一个点的dst来进行赋值
                auto &p = cloud_k->points[idx_];  // idx instead of indices[idx]t
                p.cons = P.cons, p.conf = P.conf, p.unc = P.unc;
                // std::cout << p.cons << " " << p.conf << " " << p.unc << "\n";
            }
        }
    }

    void set_Q_points(pcl::PointCloud<PointLidar>::Ptr &cloud) {
        if (!this->Q_points.empty())
            this->Q_points.clear();
        pcl::octree::OctreePointCloud<PointLidar> octree(3.0f); // 设置八叉树的分辨率
        octree.setInputCloud(cloud);
        octree.addPointsFromInputCloud();
        for (auto it = octree.leaf_begin(); it != octree.leaf_end(); it++) {
            int idx = it.getLeafContainer().getPointIndex();
            auto Q = cloud->points[idx];
            if (random() % 11 != 0)  // random 选取 Q 点 TODO
                continue;
            this->Q_points.push_back(Q);
        }
    }

    void
    icp(pcl::PointCloud<PointLidar>::Ptr &cloud_from, pcl::PointCloud<PointLidar>::Ptr &cloud_to) {
        pcl::IterativeClosestPoint<PointLidar, PointLidar> icp; // gicp / icp TODO
        icp.setInputSource(cloud_from);
        icp.setInputTarget(cloud_to);
        pcl::PointCloud<PointLidar>::Ptr Final(new pcl::PointCloud<PointLidar>);
        icp.align(*Final);
        cloud_from = Final;
    }

    // 计算p点和三轴的夹角
    static void
    compute_angle(PointLidar &p) {
        auto op = p.getVector3fMap().normalized(); // 单位向量
        Eigen::Vector3f ox(1, 0, 0), oy(0, 1, 0), oz(0, 0, 1); // 单位向量
        p.pox = acos(op.dot(ox)) * 180 / M_PI;
        p.poy = acos(op.dot(oy)) * 180 / M_PI;
        p.poz = acos(op.dot(oz)) * 180 / M_PI;
        // std::cout << "(" << pox << "," << poy << "," << poz << ") ";
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
        int id = 0;
        for (auto p : _cloud->points) {
            float dist = p.getVector3fMap().norm();  // 点到中心的距离
//            if ((dist > 30))
//                continue;
            PointLidar tmp{};
            tmp.x = p.x, tmp.y = p.y, tmp.z = p.z;
            tmp.e = 0.0, tmp.o = 0.0, tmp.u = 1.0;  // e o u 初始化
            tmp.cons = 0.0, tmp.conf = 0.0, tmp.unc = 1.0; // cons conf unc 初始化
            tmp.seq = header.seq;   // 用时间来标, TODO
            tmp.dist = dist;
            compute_angle(tmp);
            tmp.id = id++;
            cloud->push_back(tmp);
        }

        cloud_queue.push_back(cloud);
        if (cloud_queue.size() < 20) {
            return;
        }

        auto cloud_k = *(cloud_queue.end() - 1); // 当前帧，也就是最后一帧
        auto cloud_i = *(cloud_queue.begin());
        this->set_Q_points(cloud_k);
        this->dst(cloud_k);
        this->icp(cloud_i, cloud_k);
        this->dst(cloud_i);
        this->consistency_assessment(cloud_k, cloud_i);
//        for (auto it = cloud_queue.begin(); it != cloud_queue.end() - 1; it++) {
//            this->icp(*it, cloud_k);
//            this->dst(*it);
//            this->consistency_assessment(cloud_k, *it);
//        }

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_show(new pcl::PointCloud<pcl::PointXYZRGB>);
        for (auto p : cloud_k->points) {
            // std::cout << "(" << p.conf << "," << p.cons << "," << p.unc << ")";
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



