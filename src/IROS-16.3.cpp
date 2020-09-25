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
        sub = nh.subscribe("/kitti/velo/pointcloud", 1, &Node::cloud_cb, this);  // kitti
        // sub = nh.subscribe("/velodyne_points", 1, &Node::cloud_cb, this);
        lambda_theta = 0.4; // 雷达的垂直分辨率 64-0.4, 16-2.0
        // Create a ROS publisher for the output point cloud
        pub = nh.advertise<pcl::PointCloud<pcl::PointXYZRGB>>("/tmp", 1);
        cloud_global = boost::make_shared<pcl::PointCloud<PointLidar>>();
        frame_count = 5; // 连续的帧数
        Q_count = 3;
        stamp = 0; // 用于标记当前帧的 id
        ground_remove_octree_resolution = 0.4;
    };

    // IROS-16，地面去除，据论文说是 novel 的
    void
    ground_remove(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_tmp(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::octree::OctreePointCloud<pcl::PointXYZ> octree(this->ground_remove_octree_resolution);
        octree.setInputCloud(cloud);
        octree.addPointsFromInputCloud();
        double h_hat = 0.0;
        double s = 0.09;
        for (auto it = octree.leaf_begin(); it != octree.leaf_end(); ++it) {
            std::vector<int> indices;
            it.getLeafContainer().getPointIndices(indices); // 树叶不可能为空
            auto p = cloud->points[it.getLeafContainer().getPointIndex()]; // 叶子里的一个点（源代码里好像就是 vector 的最后一个值。。。）
            auto r = p.getVector3fMap().norm(); // 点的半径
//            if (r > 30.0 || r < 3.0)
//                continue;
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
            } else {
                h_hat = H;
                for (auto idx : indices) {
                    cloud_tmp->push_back(cloud->points[idx]);   // 非地面的
                }
            }
        }
        cloud = cloud_tmp;
    }

    std::tuple<double, double, double>
    m(PointLidar P, PointLidar Q) {
        Eigen::Vector3f OQ = Q.getVector3fMap(); // 向量
        Eigen::Vector3f OP = P.getVector3fMap();
        auto theta = pcl::getAngle3D(OP, OQ, true);  // 两个向量的夹角
        double _OP_ = abs(OP.dot(OQ.normalized()));  // OP 在 OQ 上的投影，有可能是负值，但是我取了绝对值
        double _OQ_ = OQ.norm(); // OQ 的范数，也就是模长
        double e, o, u;
        if (theta > 0.3) {  // 如果角 POQ 过大，说明 P 点是空的
            e = 1;
            o = 0;
        } else if (abs(_OP_ - _OQ_) < 0.1) {
            e = 0;
            o = 1;
        } else if (_OP_ < _OQ_) {
            e = 1;
            o = 0.0;
            auto f_theta = exp(-0.5 * pow(theta / this->lambda_theta, 2)); // lambda_theta 是雷达的垂直分辨率: 64:0.4, 16:2
            e = e * f_theta;
        } else {
            e = 0.0;
            o = 1.0;
            double r = _OP_ - _OQ_;
            o = o * exp(-0.5f * pow(r, 2));
        }
        u = 1 - e - o;
        // std::cout << "test: " << e << " " << o << " " << u << "\n";
        return {e, o, u};
    }

    void
    fusion(PointLidar &P, PointLidar Q) {
        double e1 = P.e, o1 = P.o, u1 = P.u;
        // TODO 这个地方贼适合用来 angle 3d
        auto eou = this->m(P, Q);
        double e2 = std::get<0>(eou), o2 = std::get<1>(eou), u2 = std::get<2>(eou);
        double K = o1 * e2 + e1 * o2;
        if (K == 1) {  // TODO 异常，意味着两位证人的判断结果是相反的
            std::cout << "exception: K == 1" << std::endl;
            P.e = 0, P.o = 0, P.u = 1;  // 因为默认是 o，所以相反的判断便是 e，其实就是移动点
            return;
        }
        double e = (e1 * e2 + e1 * u2 + u1 * e2) / (1 - K);
        double o = (o1 * o2 + o1 * u2 + u1 * o2) / (1 - K);
        double u = (u1 * u2) / (1 - K);
        if (isnan(e) || isnan(o) || isnan(u)) {
            std::cout << "exception: DST fusion eou isnan" << std::endl;
            std::cout << e << " " << o << " " << u << "\n";
            P.e = 0, P.o = 0, P.u = 1;
            return;
        }
        P.e = e, P.o = o, P.u = u;
    }

    void
    robust(pcl::PointCloud<PointLidar>::Ptr &cloud) {
        if (cloud->empty())
            return;
        PointLidar B = cloud->points[0]; // 最远的点
        for (auto &p : cloud->points)
            if (p.dist > B.dist)
                B = p;
        double r_sup = 0.8, r_inf = 0.6;
        for (auto &P : cloud->points) {
            double l = r_sup - (r_sup - r_inf) * (P.dist / B.dist);
            P.e = (P.e > P.o) && (P.e > P.u) ? l : 0;
            P.o = (P.o > P.e) && (P.o > P.u) ? l : 0;
            P.u = 1 - P.e - P.o;
        }
    }

    // 变成适合进行 kdtree 搜索的形式，以和三轴的夹角作为 kd
    void
    cloud_change_to(pcl::PointCloud<PointLidar>::Ptr &cloud) {
        for (auto &p : cloud->points) {
            this->point_change_to(p);
        }
    }

    void
    point_change_to(PointLidar &p) {
        float x, y, z;
        x = p.x, y = p.y, z = p.z;
        p.x = p.pox, p.y = p.poy, p.z = p.poz;
        p.pox = x, p.poy = y, p.poz = z;
    }

    std::vector<int>
    angle_kdtree_search(pcl::KdTreeFLANN<PointLidar> &kdtree, PointLidar P) {
        point_change_to(P);
        // K nearest neighbor search
        int K = this->Q_count;
        std::vector<int> pointIdxNKNSearch(K);
        std::vector<float> pointNKNSquaredDistance(K);
        kdtree.nearestKSearch(P, K, pointIdxNKNSearch, pointNKNSquaredDistance);
        return pointIdxNKNSearch;
    }

    // k 代表核心帧，i 代表参考帧
    // Dempster-Shafer Theory (DST)
    pcl::PointCloud<PointLidar>::Ptr
    DST(pcl::PointCloud<PointLidar>::Ptr &_cloud_k, pcl::PointCloud<PointLidar>::Ptr _cloud_i) {
        // 对两帧做一个拷贝，不改变初始点云的值
        pcl::PointCloud<PointLidar>::Ptr cloud_k(new pcl::PointCloud<PointLidar>);
        pcl::copyPointCloud(*_cloud_k, *cloud_k);
        pcl::PointCloud<PointLidar>::Ptr cloud_i(new pcl::PointCloud<PointLidar>);
        pcl::copyPointCloud(*_cloud_i, *cloud_i);
        // 这棵八叉树主要是为了减少计算量，所以分辨率越小越好
        pcl::octree::OctreePointCloud<PointLidar> octree_k(0.01f);
        octree_k.setInputCloud(cloud_k);
        octree_k.addPointsFromInputCloud();
        // 这棵 kdtree 以和三轴的夹角作为 kd
        pcl::KdTreeFLANN<PointLidar> angle_kdtree;
        this->cloud_change_to(cloud_i);
        angle_kdtree.setInputCloud(cloud_i);

        for (auto it = octree_k.leaf_begin(); it != octree_k.leaf_end(); it++) {
            std::vector<int> indices;
            it.getLeafContainer().getPointIndices(indices);
            int idx = it.getLeafContainer().getPointIndex();  // 取体素中的一个点作为代表
            auto P = cloud_k->points[idx];  // 取体素中的一个点作为代表
            auto Qs = this->angle_kdtree_search(angle_kdtree, P);
            for (auto Qi : Qs) { // 仅计算体素中的一个点的dst
                auto Q = _cloud_i->points[Qi];
                fusion(P, Q);
            }
            for (auto idx_ : indices) {  // 体素中其它的点全都按照第一个点的dst来进行赋值
                auto &p = cloud_k->points[idx_];  // idx instead of indices[idx]
                p.e = P.e, p.o = P.o, p.u = P.u;
            }
        }
        this->robust(cloud_k); // IROS-16
        return cloud_k;
    }

    void consistency_relations(PointLidar &P, PointLidar P_) {
        double E1 = P.e, O1 = P.o, U1 = P.u;
        double E2 = P_.e, O2 = P_.o, U2 = P_.u;
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

    // 两帧之间的一致性评估
    pcl::PointCloud<PointLidar>::Ptr // i, j 就是同一帧相对与不同参考帧的 DST 计算结果
    consistency_assessment(pcl::PointCloud<PointLidar>::Ptr _cloud_i, pcl::PointCloud<PointLidar>::Ptr cloud_j) {
        pcl::PointCloud<PointLidar>::Ptr cloud_i(new pcl::PointCloud<PointLidar>);
        pcl::copyPointCloud(*_cloud_i, *cloud_i); // 这三个点云反正都是一样大的
        for (auto i = 0; i < cloud_i->size(); i++) {
            auto &pi = cloud_i->points[i];
            auto pj = cloud_j->points[i];
            this->consistency_relations(pi, pj);
//            if (pi.conf != 0)
//            std::cout << pi.cons << " " << pi.conf << " " << pi.unc << "\n";
        }
        return cloud_i;
    }

    // 点云配准
    void
    registration(pcl::PointCloud<PointLidar>::Ptr &cloud_source, pcl::PointCloud<PointLidar>::Ptr &cloud_target) {
        // return;
        pcl::IterativeClosestPoint<PointLidar, PointLidar> icp; // gicp / icp TODO
        icp.setInputSource(cloud_source);
        icp.setInputTarget(cloud_target);
        pcl::PointCloud<PointLidar>::Ptr Final(new pcl::PointCloud<PointLidar>);
        std::cout << icp.getMaximumIterations() << "\n";
        icp.align(*Final);
        cloud_source = Final;
    }

    // 计算p点和三轴的夹角
    static void
    compute_angle(PointLidar &p) {
        auto op = p.getVector3fMap().normalized(); // 单位向量
        Eigen::Vector3f ox(1, 0, 0), oy(0, 1, 0), oz(0, 0, 1); // 单位向量
        p.pox = pcl::getAngle3D(p.getVector3fMap(), ox, true);
        p.poy = pcl::getAngle3D(p.getVector3fMap(), oy, true);
        p.poz = pcl::getAngle3D(p.getVector3fMap(), oz, true);
    }

    // 将 PointXYZ 点云转换成 PointLidar
    pcl::PointCloud<PointLidar>::Ptr
    toPointLidar(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud) {
        pcl::PointCloud<PointLidar>::Ptr cloud_lidar(new pcl::PointCloud<PointLidar>);
        for (auto p : cloud->points) {
            PointLidar tmp{};
            tmp.x = p.x, tmp.y = p.y, tmp.z = p.z;
            tmp.e = 0.0, tmp.o = 1.0, tmp.u = 0.0;  // e o u 初始化，初始化为 occupancy
            tmp.cons = 1.0, tmp.conf = 0.0, tmp.unc = 0.0; // cons conf unc 初始化，这个其实无所谓，反正都要覆盖掉
            tmp.stamp = this->stamp;
            cloud_lidar->push_back(tmp);
        }
        return cloud_lidar;
    }

    // 八叉树降采样
    void
    octreeDownsampling(pcl::PointCloud<PointLidar>::Ptr &cloud, double resolution = 0.01) {
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
    void
    filterAndInitAfterRegistration(pcl::PointCloud<PointLidar>::Ptr &cloud) {
        pcl::PointCloud<PointLidar>::Ptr cloud_filtered(new pcl::PointCloud<PointLidar>);
        for (auto p : cloud->points) {
            p.dist = p.getVector3fMap().norm();
            compute_angle(p);  // 这个很重要，而且要在 ICP 之后计算
            cloud_filtered->push_back(p);
        }
        cloud = cloud_filtered;
    }

    void
    ca_fusion(PointLidar &P, PointLidar P_) {
        double conf1 = P.conf, cons1 = P.cons, unc1 = P.unc;
        double conf2 = P_.conf, cons2 = P_.cons, unc2 = P_.unc;
        double K = cons1 * conf2 + conf1 * cons2;
        if (K == 1) {  // TODO 异常，意味着两位证人的判断结果是相反的
            std::cout << "exception: ca fusion K == 1" << std::endl;
            P.unc = 0.5, P.cons = 0.5, P.conf = 0;
            return;
        }
        double conf = (conf1 * conf2 + conf1 * unc2 + unc1 * conf2) / (1 - K);
        double cons = (cons1 * cons2 + cons1 * unc2 + unc1 * cons2) / (1 - K);
        double unc = (unc1 * unc2) / (1 - K);
        if (isnan(conf) || isnan(cons) || isnan(unc)) {
            std::cout << "exception: DST fusion eou isnan" << std::endl;
            std::cout << conf << " " << cons << " " << unc << "\n";
            P.conf = 0, P.cons = 0, P.unc = 1;
            return;
        }
        P.conf = conf, P.cons = cons, P.unc = unc;
    }

    // 对多次的一致性评估做一个求和或者求平均
    pcl::PointCloud<PointLidar>::Ptr
    avg_consistency_assessment(std::vector<pcl::PointCloud<PointLidar>::Ptr> &cloud_cas) {
        pcl::PointCloud<PointLidar>::Ptr cloud_result(new pcl::PointCloud<PointLidar>);
        pcl::copyPointCloud(*cloud_cas[0], *cloud_result);
        for (auto cloud_ca : cloud_cas) {
            if (cloud_ca == cloud_result)
                continue;
            for (auto i = 0; i < cloud_result->size(); i++) {
                auto &p = cloud_result->points[i];
                auto p_ = cloud_ca->points[i];
                // this->ca_fusion(p, p_);
                p.cons += p_.cons; p.cons /= 2;
                p.conf += p_.conf; p.conf /= 2;
                p.unc += p_.unc; p.unc /= 2;
            }
        }
        return cloud_result;
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
        auto cloud_lidar = toPointLidar(cloud_xyz);
        // 用八叉树降采样，叶子边长 1 cm
        // this->octreeDownsampling(cloud_lidar);
        // change to PointLidar type
        cloud_queue.push_back(cloud_lidar);
        if (cloud_queue.size() < this->frame_count) {
            return;
        }

        std::vector<pcl::PointCloud<PointLidar>::Ptr> cloud_dsts;
        auto &cloud_k = *(this->cloud_queue.end() - 1); // 要计算 eou 的帧，核心帧，这里选择最新的一帧
        auto &cloud_i = *(this->cloud_queue.begin());
        this->registration(cloud_i, cloud_k);
        this->filterAndInitAfterRegistration(cloud_k);
        this->filterAndInitAfterRegistration(cloud_i);
        auto dst_i = this->DST(cloud_k, cloud_i);
        auto ca = this->consistency_assessment(cloud_k, dst_i);
//        this->filterAndInitAfterRegistration(cloud_k); // 计算夹角啊之类的
//        for (auto cloud_i : this->cloud_queue) { // 全部计算一遍DST，包括核心帧参考核心帧
//            if (cloud_i == cloud_k)
//                continue;
//            // this->registration(cloud_i, cloud_k);
//            this->filterAndInitAfterRegistration(cloud_i);
//            auto dst_i = this->DST(cloud_k, cloud_i);
//            cloud_dsts.push_back(dst_i);
//        }
//
//        std::vector<pcl::PointCloud<PointLidar>::Ptr> cloud_cas;
//        auto dst_k = cloud_k;  // 都当做 occupancy
//        for (auto dst_i : cloud_dsts) {
//            auto cloud_ca = this->consistency_assessment(dst_k, dst_i);
//            cloud_cas.push_back(cloud_ca);
//        }
//        auto cloud_result = this->avg_consistency_assessment(cloud_cas);

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_show(new pcl::PointCloud<pcl::PointXYZRGB>);
        for (auto p : ca->points) {
            // std::cout << p.conf + p.unc + p.cons << std::endl;
            // std::cout << p.conf << " " << p.unc << " " << p.cons << "\n";
            pcl::PointXYZRGB tmp;
            tmp.x = p.x, tmp.y = p.y, tmp.z = p.z;
            tmp.r = 0, tmp.g = 255, tmp.b = 0;
            if ((p.conf > p.cons) || (p.unc > p.cons)) {
                tmp.r = 255, tmp.g = 0, tmp.b = 0;
                // tmp.r = 255 * p.conf, tmp.g = 255 * p.cons, tmp.b = 255 * p.unc;
            }
            cloud_show->push_back(tmp);
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
