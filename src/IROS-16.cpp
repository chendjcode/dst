//
// Created by cdj on 2020/3/31.
//
/*
 * 按照 IROS-16 的论文来，设置一个全局点云
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
)

class Node {
private:
    ros::NodeHandle nh;
    ros::Subscriber sub;
    ros::Publisher pub;
    pcl::console::TicToc tim;  // 计时器
    std::vector<pcl::PointCloud<PointLidar>::Ptr> cloud_queue, cloud_eous; // 保存了连续的点云; eou 结果临时保留
    pcl::PointCloud<PointLidar>::Ptr cloud_global;  // 全局点云
public:
    Node() {
        // Create a ROS subscriber for the input point cloud
        // sub = nh.subscribe("/kitti/velo/pointcloud", 1, &Node::cloud_cb, this);  // kitti
        sub = nh.subscribe("/velodyne_points", 1, &Node::cloud_cb, this);
        // Create a ROS publisher for the output point cloud
        pub = nh.advertise<pcl::PointCloud<pcl::PointXYZRGB>>("/tmp", 1);
        cloud_global = boost::make_shared<pcl::PointCloud<PointLidar>>();
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
        auto theta = acos(OP.normalized().dot(OQ.normalized())) * 180 / M_PI;
        auto f_theta = exp(-0.5 * pow(theta / 0.4, 2));
        return f_theta;
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

    // 构造一棵用于角度搜索的 ktree
    pcl::KdTreeFLANN<PointLidar>::Ptr
    plantAngleKdTree(pcl::PointCloud<PointLidar>::Ptr &cloud) {
        pcl::PointCloud<PointLidar>::Ptr cloud_i_copy(new pcl::PointCloud<PointLidar>);
        pcl::copyPointCloud(*cloud, *cloud_i_copy);
        cloud_change_to(cloud_i_copy);
        pcl::KdTreeFLANN<PointLidar>::Ptr kdtree(new pcl::KdTreeFLANN<PointLidar>);
        kdtree->setInputCloud(cloud_i_copy);
        return kdtree;
    }

    // 使用 k 帧的 P 点，在 i 帧中搜索附近的 Q 点
    std::vector<int>
    get_Q_points_indices(pcl::KdTreeFLANN<PointLidar>::Ptr &kdtree, PointLidar P) {
        std::vector<int> Q_indices;
        // Neighbors within radius search
        std::vector<int> pointIdxRadiusSearch;
        std::vector<float> pointRadiusSquaredDistance;
        float radius = 2.0;
        kdtree->radiusSearch (P, radius, pointIdxRadiusSearch, pointRadiusSquaredDistance);
        for (auto i = 0; i < pointRadiusSquaredDistance.size(); i++) {
            if (pointRadiusSquaredDistance[i] > (radius / 2))
                Q_indices.push_back(pointIdxRadiusSearch[i]);
        }
        // std::cout << Q_indices.size() << std::endl;
        return Q_indices;
    }

    // k 代表核心帧，i 代表参考帧
    // Dempster-Shafer Theory (DST)
    pcl::PointCloud<PointLidar>::Ptr
    DST(pcl::PointCloud<PointLidar>::Ptr &cloud_k, pcl::PointCloud<PointLidar>::Ptr &cloud_i) {
        // 用于减少计算量的 octree
        pcl::PointCloud<PointLidar>::Ptr cloud_k_copy(new pcl::PointCloud<PointLidar>);
        pcl::copyPointCloud(*cloud_k, *cloud_k_copy);
        pcl::octree::OctreePointCloud<PointLidar> octree(0.3f); // 设置八叉树的分辨率为 0.4 m
        octree.setInputCloud(cloud_k_copy);  //
        octree.addPointsFromInputCloud();
        // 用于加速搜索的 kdtree
        pcl::KdTreeFLANN<PointLidar>::Ptr kdtree(new pcl::KdTreeFLANN<PointLidar>);
        kdtree->setInputCloud(cloud_i);

        for (auto it = octree.leaf_begin(); it != octree.leaf_end(); it++) {
            std::vector<int> indices;
            it.getLeafContainer().getPointIndices(indices);
            int idx = it.getLeafContainer().getPointIndex();  // 取体素中的一个点作为代表
            auto P = cloud_k_copy->points[idx];  // 取体素中的一个点作为代表
            auto Q_indices = get_Q_points_indices(kdtree, P);
            for (auto idx_ : Q_indices) { // 仅计算体素中的一个点的dst
                auto Q = cloud_i->points[idx_];
                fusion(P, Q);
            }
            // std::cout << P.e << " " << P.o << " " << P.u << "\n";
            // std::cout << P.cons << " " << P.conf << " " << P.unc << "\n";
            for (auto idx_ : indices) {  // 体素中其它的点全都按照第一个点的dst来进行赋值
                auto &p = cloud_k_copy->points[idx_];  // idx instead of indices[idx]t
                p.e = P.e, p.o = P.o, p.u = P.u;
            }
        }
        robust(cloud_k_copy); // IROS-16
        return cloud_k_copy;
    }

    static void consistency_relations(PointLidar &P, PointLidar P_) {
        float E1 = P.e, O1 = P.o, U1 = P.u;
        float E2 = P_.e, O2 = P_.o, U2 = P_.u;
        P.conf = E1 * O2 + O1 * E2;
        P.cons = E1 * E2 + O1 * O2 + U1 * U2;
        P.unc = U1 * (E2 + O2) + U2 * (E1 + O1);
        // std::cout << P.conf << " " << P.unc << " " << P.cons << "\n";
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

    void // k 是要计算的帧，ref 是参考帧
    consistency_assessment(pcl::PointCloud<PointLidar>::Ptr &cloud_k, pcl::PointCloud<PointLidar>::Ptr &cloud_i) {
        for (auto i = 0; i < cloud_k->size(); i++) {
            auto &P = cloud_k->points[i];
            auto P_ = cloud_i->points[i];
            this->consistency_relations(P, P_);
        }
    }

    // 点云配准
    void
    registration(pcl::PointCloud<PointLidar>::Ptr &cloud_source) {
        pcl::IterativeClosestPoint<PointLidar, PointLidar> icp; // gicp / icp TODO
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
    static pcl::PointCloud<PointLidar>::Ptr
    toPointLidar(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud) {
        pcl::PointCloud<PointLidar>::Ptr cloud_lidar(new pcl::PointCloud<PointLidar>);
        for (auto p : cloud->points) {
            float dist = p.getVector3fMap().norm();  // 点到中心的距离
            if (dist > 30) continue; // 只保留半径30米以内的点
            PointLidar tmp{};
            tmp.x = p.x, tmp.y = p.y, tmp.z = p.z;
            tmp.e = 0.0, tmp.o = 0.0, tmp.u = 1.0;  // e o u 初始化
            tmp.cons = 0.0, tmp.conf = 0.0, tmp.unc = 1.0; // cons conf unc 初始化
            tmp.dist = dist;
            // compute_angle(tmp);
            cloud_lidar->push_back(tmp);
        }
        return cloud_lidar;
    }

    // 将配准后的点云合并到全局点云中
    void
    mergeCloud(pcl::PointCloud<PointLidar>::Ptr &cloud) {
        for (auto p : cloud->points)
            if ((p.cons > p.unc) || (p.cons > p.conf) || this->cloud_eous.empty())
                this->cloud_global->push_back(p);
        // 每次合并完全局点云后，使用八叉树降采样
        pcl::PointCloud<PointLidar>::Ptr cloud_filtered(new pcl::PointCloud<PointLidar>);
        pcl::octree::OctreePointCloud<PointLidar> octree(0.01f); // 设置八叉树的分辨率
        octree.setInputCloud(this->cloud_global);
        octree.addPointsFromInputCloud();
        for (auto it = octree.leaf_begin(); it != octree.leaf_end(); it++) {
            auto idx = it.getLeafContainer().getPointIndex();  // 取体素中的一个点作为代表
            cloud_filtered->push_back(this->cloud_global->points[idx]);
        }
        this->cloud_global = cloud_filtered;
    }

    // 配准之后进行一些值初始化和过滤操作
    static void
    filterAfterRegistration(pcl::PointCloud<PointLidar>::Ptr &cloud) {
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
        ground_remove(cloud_xyz);
        auto cloud_lidar = toPointLidar(cloud_xyz);
        // ground remove
        // change to PointLidar type
        cloud_queue.push_back(cloud_lidar);
        if (cloud_queue.size() < 3) {
            return;
        }
        this->cloud_global->clear();
        this->cloud_eous.clear();
        auto cloud_k = this->cloud_queue[this->cloud_queue.size() / 2]; // 要计算 eou 的帧，核心帧
        this->filterAfterRegistration(cloud_k);
        this->mergeCloud(cloud_k);
        for (auto cloud_i : this->cloud_queue) {  // 参考帧
            if (cloud_i != cloud_k) {
                this->registration(cloud_i);
                this->filterAfterRegistration(cloud_i);
                auto cloud_i_eou = this->DST(cloud_k, cloud_i);
                this->cloud_eous.push_back(cloud_i_eou);
                this->mergeCloud(cloud_i_eou);
                std::cout << ".";
            }
        }

        auto ca_k = this->cloud_eous.begin();
        for (auto it = ca_k + 1; it != this->cloud_eous.end(); it++) {
            this->consistency_assessment(*ca_k, *it);
        }

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_show(new pcl::PointCloud<pcl::PointXYZRGB>);
        for (auto p : (*ca_k)->points) {
            // std::cout << "(" << p.conf << "," << p.cons << "," << p.unc << ") \n";
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
