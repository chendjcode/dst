//
// Created by cdj on 2020/4/1.
//
/*
 * DST 部分不按照论文来，因为我感觉论文是错的
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
#include <pcl/filters/voxel_grid.h>

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
    float dist;
    float pox;
    float poy;
    float poz;
    int stamp;
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
                                           (float, dist, dist)
                                           (float, pox, pox)
                                           (float, poy, poy)
                                           (float, poz, poz)
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
    std::vector<PointLidar> Q_points;
    float ground_remove_octree_resolution;  // 用于去除地面的八叉树的分辨率
    float lambda_theta; // 雷达的垂直分辨率
    int stamp; // 用于标记点云
    int frame_count;
public:
    Node() {
        // Create a ROS subscriber for the input point cloud
        // sub = nh.subscribe("/kitti/velo/pointcloud", 1, &Node::cloud_cb, this);  // kitti
        sub = nh.subscribe("/velodyne_points", 1, &Node::cloud_cb, this);
        // Create a ROS publisher for the output point cloud
        pub = nh.advertise<pcl::PointCloud<pcl::PointXYZRGB>>("/tmp", 1);
        cloud_global = boost::make_shared<pcl::PointCloud<PointLidar>>();
        frame_count = 3; // 连续的帧数
        stamp = 0; // 用于标记当前帧的 id
        ground_remove_octree_resolution = 0.6; // 64:0.4，16线的要更大一些
        lambda_theta = 2.0; // 雷达的垂直分辨率 64-0.4, 16-2.0
    };

    // IROS-16，地面去除，据论文说是 novel 的
    void
    ground_remove(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_tmp(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::octree::OctreePointCloud<pcl::PointXYZ> octree(this->ground_remove_octree_resolution);
        octree.setInputCloud(cloud);
        octree.addPointsFromInputCloud();
        float h_hat = 0.0;
        float s = 0.09;
        for (auto it = octree.leaf_begin(); it != octree.leaf_end(); ++it) {
            std::vector<int> indices;
            it.getLeafContainer().getPointIndices(indices); // 树叶不可能为空
            float H = cloud->points[it.getLeafContainer().getPointIndex()].z;  // 初始化最高和最低值
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

    std::tuple<float, float, float>
    m(PointLidar P, PointLidar Q) {
        auto OQ = Q.getVector3fMap();
        auto OP = P.getVector3fMap();
        float _OP_ = abs(OP.dot(OQ.normalized()));  // OP 在 OQ 上的投影，有可能是负值，但是我取了绝对值
        float _OQ_ = OQ.norm(); // OQ 的范数，也就是模长
        float e, o, u;
        if (_OP_ < _OQ_) {
            auto theta = acos(OP.normalized().dot(OQ.normalized())) * 180 / M_PI;
            e = exp(-0.5 * pow(theta / this->lambda_theta, 2)); // lambda_theta 是雷达的垂直分辨率: 64:0.4, 16:2
            o = 0.0;
        } else {
            e = 0.0;
            float r = _OP_ - _OQ_;
            o = exp(-0.5f * pow(r, 2));
        }
        u = 1 - e - o;
//        if ( e > u)
//            std::cout << "test: " << e << " " << o << " " << u << "\n";
        return {e, o, u};
    }

    void
    fusion(PointLidar &P, PointLidar Q) {
        float e1 = P.e, o1 = P.o, u1 = P.u;
        auto eou = this->m(P, Q);
        float e2 = std::get<0>(eou), o2 = std::get<1>(eou), u2 = std::get<2>(eou);
        float K = o1 * e2 + e1 * o2;
        if (u1 == 1 && !isnan(e1) && !isnan(o1) && !isnan(u1)) {  // 针对第一次计算，直接赋eou的计算值，但是计算出来的 eou 要是正常的数值
            P.e = e2, P.o = o2, P.u = u2;
            return;
        }
        if (u2 == 1 || isnan(u2) || isnan(e2) || isnan(o2))  // 这就意味着新算出来的 dst 为 unknown 或者异常的，所以不用融合进去了
            return;
        if (K == 1)  //TODO 异常，意味着两位证人的判断结果是相反的
        {
            std::cout << "exception: K == 1" << std::endl;
            // P.e = 0, P.o = 0, P.u = 1;
            return;
        }
        float e = (e1 * e2 + e1 * u2 + u1 * e2) / (1 - K);
        float o = (o1 * o2 + o1 * u2 + u1 * o2) / (1 - K);
        float u = (u1 * u2) / (1 - K);
        if (isnan(e) || isnan(o) || isnan(u)) {
            std::cout << "exception: DST fusion eou isnan" << std::endl;
            return;
        }
        P.e = e, P.o = o, P.u = u;
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

    // 使用 k 帧的 P 点，在 i 帧中搜索附近的 Q 点
    void
    set_Q_Points(pcl::PointCloud<PointLidar>::Ptr &cloud) {
        this->Q_points.clear();
//        pcl::octree::OctreePointCloud<PointLidar> octree(2.0f); // 设置八叉树的分辨率为 0.4 m
//        octree.setInputCloud(cloud);  //
//        octree.addPointsFromInputCloud();
//        for (auto it = octree.leaf_begin(); it != octree.leaf_end(); it++) {
//            int idx = it.getLeafContainer().getPointIndex();  // 取体素中的一个点作为代表
//            auto Q = cloud->points[idx];  // 取体素中的一个点作为代表
//            if (Q.pox > 89 && Q.dist > 2)
//                this->Q_points.push_back(Q);
//        }
        for (int y = -15; y < 15; y+=3) {
            PointLidar tmp{};
            tmp.x = 0, tmp.y = y, tmp.z = 0;
            this->Q_points.push_back(tmp);
        }
        std::cout << "Q count = " << this->Q_points.size() << "\n";
    }

    // k 代表核心帧，i 代表参考帧
    // Dempster-Shafer Theory (DST)
    void
    DST(pcl::PointCloud<PointLidar>::Ptr &cloud) {
        // 用于减少计算量的 octree
        pcl::octree::OctreePointCloud<PointLidar> octree(0.1f); // 这里的八叉树主要是为了减少计算量，所以分辨率越小越好
        octree.setInputCloud(cloud);  //
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
                auto &p = cloud->points[idx_];  // idx instead of indices[idx]
                p.e = P.e, p.o = P.o, p.u = P.u;
            }
        }
        robust(cloud); // IROS-16
    }

    void consistency_relations(PointLidar &P, PointLidar P_) {
        float E1 = P.e, O1 = P.o, U1 = P.u;
        float E2 = P_.e, O2 = P_.o, U2 = P_.u;
        P.conf = E1 * O2 + O1 * E2;
        P.cons = E1 * E2 + O1 * O2 + U1 * U2;
        P.unc = U1 * (E2 + O2) + U2 * (E1 + O1);
        // std::cout << P.conf << " " << P.unc << " " << P.cons << "\n\n";
    }

    // 合并两个点云返回一个新的点云
    pcl::PointCloud<PointLidar>::Ptr
    mergePointCloud(pcl::PointCloud<PointLidar>::Ptr &cloud_A, pcl::PointCloud<PointLidar>::Ptr &cloud_B) {
        pcl::PointCloud<PointLidar>::Ptr cloud_both(new pcl::PointCloud<PointLidar>);
        for (auto p : cloud_A->points)
            cloud_both->push_back(p);
        for (auto p : cloud_B->points)
            cloud_both->push_back(p);
        return cloud_both;
    }

    // 求一个八叉树叶子中 eou 平均值
    PointLidar
    avg_eou(pcl::PointCloud<PointLidar>::Ptr &cloud, std::vector<int> indices) {
        auto p = cloud->points[indices[0]];
        for (auto i = 1; i < indices.size(); i++) {
            auto tmp = cloud->points[indices[i]];
            p.x = (p.x + tmp.x) / 2;
            p.y = (p.y + tmp.y) / 2;
            p.z = (p.z + tmp.z) / 2;
        }
        return p;
    }

    // 两帧之间的一致性评估
    pcl::PointCloud<PointLidar>::Ptr // k 是要计算的帧，i 是参考帧
    consistency_assessment(pcl::PointCloud<PointLidar>::Ptr &cloud_k, pcl::PointCloud<PointLidar>::Ptr &cloud_i) {
        auto cloud_both = mergePointCloud(cloud_k, cloud_i);
        pcl::octree::OctreePointCloud<PointLidar> octree(0.5f);
        octree.setInputCloud(cloud_both);
        octree.addPointsFromInputCloud();

        auto frame_k = cloud_k->points[0].stamp;
        auto frame_i = cloud_i->points[0].stamp;
        for (auto it = octree.leaf_begin(); it != octree.leaf_end(); it++) {
            std::vector<int> indices, indices_k, indices_i;
            it.getLeafContainer().getPointIndices(indices);
            for (auto ind : indices) {  // 将这个叶子中的两帧分离开
                auto p = cloud_both->points[ind];
                if (p.stamp == frame_k)
                    indices_k.push_back(ind);
                else
                    indices_i.push_back(ind);
            }
            if(!indices_k.empty() && indices_i.empty()) {
                for (auto ind : indices_k) {
                    auto &p = cloud_both->points[ind];
                    p.conf = 1, p.cons = p.unc = 0;
                }
            } else if (indices_k.empty() && !indices_i.empty()) {
                for (auto ind : indices_i) {
                    auto &p = cloud_both->points[ind];
                    p.conf = 1, p.cons = p.unc = 0;
                }
            } else {  // 一个叶子中存在两帧的点
                auto p_k_avg = avg_eou(cloud_both, indices_k);
                auto p_i_avg = avg_eou(cloud_both, indices_i);
                consistency_relations(p_k_avg, p_i_avg);
                for (auto i : indices) {
                    auto &p = cloud_both->points[i];
                    p.cons = p_k_avg.cons, p.conf = p_k_avg.conf, p.unc = p_k_avg.unc;
                }
            }
        }
        return cloud_both;
    }

    // 点云配准
    void
    registration(pcl::PointCloud<PointLidar>::Ptr &cloud_source) {
        return;
        pcl::GeneralizedIterativeClosestPoint<PointLidar, PointLidar> icp; // gicp / icp TODO
        icp.setInputSource(cloud_source);
        icp.setInputTarget(this->cloud_global);
        pcl::PointCloud<PointLidar>::Ptr Final(new pcl::PointCloud<PointLidar>);
        icp.align(*Final);
        cloud_source = Final;
    }

    // 计算p点和三轴的夹角
    static void
    compute_angle(PointLidar &p) {
        auto op = p.getVector3fMap().normalized(); // 单位向量
        Eigen::Vector3f ox(1, 0, 0), oy(0, 1, 0), oz(0, 0, 1); // 单位向量
        p.pox = acos(op.dot(ox)) * 180 / M_PI;
        p.poy = acos(op.dot(oy)) * 180 / M_PI;
        p.poz = acos(op.dot(oz)) * 180 / M_PI;
    }

    // 将 PointXYZ 点云转换成 PointLidar
    pcl::PointCloud<PointLidar>::Ptr
    toPointLidar(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud) {
        pcl::PointCloud<PointLidar>::Ptr cloud_lidar(new pcl::PointCloud<PointLidar>);
        for (auto p : cloud->points) {
            PointLidar tmp{};
            tmp.x = p.x, tmp.y = p.y, tmp.z = p.z;
            tmp.e = 0.0, tmp.o = 0.0, tmp.u = 1.0;  // e o u 初始化
            tmp.cons = 0.0, tmp.conf = 0.0, tmp.unc = 1.0; // cons conf unc 初始化
            tmp.stamp = this->stamp;
            cloud_lidar->push_back(tmp);
        }
        return cloud_lidar;
    }

    // 八叉树降采样
    void
    octreeDownsampling(pcl::PointCloud<PointLidar>::Ptr &cloud, float resolution = 0.01) {
        pcl::PointCloud<PointLidar>::Ptr cloud_filtered(new pcl::PointCloud<PointLidar>);
        pcl::octree::OctreePointCloud<PointLidar> octree(resolution); // 设置八叉树的分辨率
        octree.setInputCloud(cloud);
        octree.addPointsFromInputCloud();
        for (auto it = octree.leaf_begin(); it != octree.leaf_end(); it++) {
            auto idx = it.getLeafContainer().getPointIndex();  // 取体素中的一个点作为代表
            cloud_filtered->push_back(cloud->points[idx]);
        }
        cloud = cloud_filtered;
    }

    // 将配准后的点云合并到全局点云中
    void
    mergeCloud(pcl::PointCloud<PointLidar>::Ptr &cloud) {
        for (auto p : cloud->points)
            this->cloud_global->push_back(p);
        // 每次合并完全局点云后，使用八叉树降采样
        octreeDownsampling(this->cloud_global);
    }

    // 配准之后进行一些值初始化和过滤操作
    static void
    filterAndInitAfterRegistration(pcl::PointCloud<PointLidar>::Ptr &cloud) {
        pcl::PointCloud<PointLidar>::Ptr cloud_filtered(new pcl::PointCloud<PointLidar>);
        for (auto p : cloud->points) {
            p.dist = p.getVector3fMap().norm();
            if (p.dist > 30) continue;
            compute_angle(p);
            cloud_filtered->push_back(p);
        }
        cloud = cloud_filtered;
    }

    // call back
    void cloud_cb(const sensor_msgs::PointCloud2ConstPtr &input) {
        tim.tic();
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz(new pcl::PointCloud<pcl::PointXYZ>);
        // convert
        pcl::fromROSMsg(*input, *cloud_xyz);
        // keep header
        auto header = cloud_xyz->header;
        this->stamp = header.stamp % 100000;
        // ground remove
        ground_remove(cloud_xyz);
        auto cloud_lidar = toPointLidar(cloud_xyz);
        this->octreeDownsampling(cloud_lidar);
        // change to PointLidar type
        cloud_queue.push_back(cloud_lidar);
        if (cloud_queue.size() < this->frame_count) {
            return;
        }
        this->cloud_global->clear();
        auto cloud_k = this->cloud_queue[this->cloud_queue.size() / 2]; // 要计算 eou 的帧，核心帧
        this->filterAndInitAfterRegistration(cloud_k); // 算一些半径啊，夹角啊之类的
        this->set_Q_Points(cloud_k);  // Q 点通过核心帧来选取
        this->DST(cloud_k);
        this->mergeCloud(cloud_k); // 合并到全局点云中
        for (auto cloud_i : this->cloud_queue) {  // 参考帧
            if (cloud_i != cloud_k) {
                this->registration(cloud_i);
                this->filterAndInitAfterRegistration(cloud_i);
                this->DST(cloud_i);
                this->mergeCloud(cloud_i);
            }
        }

        auto cloud_both = this->consistency_assessment(cloud_k, *(cloud_queue.begin()));

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_show(new pcl::PointCloud<pcl::PointXYZRGB>);
        for (auto p : cloud_both->points) {
            if (p.stamp != cloud_k->points[0].stamp) continue;
            // std::cout << p.conf << " " << p.unc << " " << p.cons << "\n";
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

