//
// Created by cdj on 2020/3/24.
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

typedef pcl::octree::OctreeBase<pcl::octree::OctreeContainerPointIndices>::LeafNodeIterator LeafNodeIterator;

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
    std::vector<pcl::PointCloud<PointLidar>::Ptr> cloud_queue, cloud_queue_copy;
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

    void
    dst(pcl::PointCloud<PointLidar>::Ptr &cloud) {
        pcl::octree::OctreePointCloudSearch<PointLidar> octree(1.0f); // 设置八叉树的分辨率为 0.4 m
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
        robust(cloud);
    }

    static void consistency_relations(PointLidar *P, PointLidar *P_) {
        float E1 = P->e, O1 = P->o, U1 = P->u;
        float E2 = P_->e, O2 = P_->o, U2 = P_->u;
        P->conf = E1 * O2 + O1 * E2;
        P->cons = E1 * E2 + O1 * O2 + U1 * U2;
        P->unc = U1 * (E2 + O2) + U2 * (O1 + O1);
    }

    static void avg_eou(PointLidar &P, PointLidar Q) {
        // std::cout << P.e << " " << P.o << " " << P.u << "\n";
        P.e = (P.e + Q.e) / 2;
        P.o = (P.o + Q.o) / 2;
        P.u = (P.u + Q.u) / 2;
    }

    static void
    consistency_assessment(pcl::PointCloud<PointLidar>::Ptr &cloud_now, pcl::PointCloud<PointLidar>::Ptr &cloud_pre) {
        pcl::PointCloud<PointLidar>::Ptr cloud(new pcl::PointCloud<PointLidar>);
        for (auto p : cloud_now->points)
            cloud->points.push_back(p);
        for (auto p : cloud_pre->points)
            cloud->points.push_back(p);
        int pre_seq = cloud_pre->points[0].seq;
        int now_seq = cloud_now->points[0].seq;
        cloud_now->clear();
        cloud_pre->clear();
        pcl::octree::OctreePointCloudSearch<PointLidar> octree(1.0f); // 设置八叉树的分辨率为 0.4 m
        octree.setInputCloud(cloud);
        octree.addPointsFromInputCloud();

        for (auto it = octree.leaf_begin(); it != octree.leaf_end(); it++) {
            std::vector<int> indices;
            it.getLeafContainer().getPointIndices(indices);
            std::vector<PointLidar> diff;
            PointLidar *pre, *now;
            for (auto idx : indices) {
                auto p = cloud->points[idx];
                if (diff.empty()) {
                    diff.push_back(p);
                    continue;
                }
                if (diff.size() == 1) {
                    if (p.seq != diff[0].seq) {
                        diff.push_back(p);
                        continue;
                    } else {
                        avg_eou(diff[0], p);
                        continue;
                    }
                }
                if (diff.size() == 2) {
                    if (p.seq == diff[0].seq) {
                        avg_eou(diff[0], p);
                        continue;
                    } else {
                        avg_eou(diff[1], p);
                        continue;
                    }
                }
            }
            // std::cout << diff.size() << "\n";
            if (diff.size() < 2) {
                // 只有一种，这说明这个叶子里的点都是 conf
                for (auto idx : indices) {  // 体素中其它的点全都按照这一个点来进行赋值
                    auto &p = cloud->points[idx];  // idx instead of indices[idx]
                    p.cons = 0;
                    p.conf = 1;
                    p.unc = 0;
                    if (p.seq == pre_seq)
                        cloud_pre->push_back(p);
                    else
                        cloud_now->push_back(p);
                }
                continue;
            }
            if (diff[0].seq > diff[1].seq) {
                now = &diff[0];
                pre = &diff[1];
            } else {
                now = &diff[1];
                pre = &diff[0];
            }
            consistency_relations(now, pre);
            for (auto idx : indices) {  // 体素中其它的点全都按照这一个点来进行赋值
                auto &p = cloud->points[idx];  // idx instead of indices[idx]
                p.cons += now->cons;
                p.cons /= 2;
                p.conf += now->conf;
                p.conf /= 2;
                p.unc += now->unc;
                p.unc /= 2;
                if (p.seq == pre_seq)
                    cloud_pre->push_back(p);
                else
                    cloud_now->push_back(p);
                // std::cout << p.cons << " " << p.conf << " " << p.unc << "\n";
            }
        }

    }

    void set_Q_points(pcl::PointCloud<PointLidar>::Ptr &cloud) {
        if (!this->Q_points.empty())
            this->Q_points.clear();
        pcl::octree::OctreePointCloudSearch<PointLidar> octree(3.0f); // 设置八叉树的分辨率
        octree.setInputCloud(cloud);
        octree.addPointsFromInputCloud();
        for (auto it = octree.leaf_begin(); it != octree.leaf_end(); it++) {
            int idx = it.getLeafContainer().getPointIndex();
            auto Q = cloud->points[idx];
            auto r = Q.getVector3fMap().norm();
            if (r > 1.0f)
                this->Q_points.push_back(Q);
        }
    }

    static Eigen::Matrix<float, 4, 4>
    icp(pcl::PointCloud<PointLidar>::Ptr &cloud_pre, pcl::PointCloud<PointLidar>::Ptr &cloud_now) {
        pcl::IterativeClosestPoint<PointLidar, PointLidar> icp;
        icp.setInputSource(cloud_pre);
        icp.setInputTarget(cloud_now);
        pcl::PointCloud<PointLidar> Final;
        icp.align(Final);
        return icp.getFinalTransformation();
    }

    void
    transform(pcl::PointCloud<PointLidar>::Ptr &cloud_pre, pcl::PointCloud<PointLidar>::Ptr &cloud_now) {
        auto T = icp(cloud_pre, cloud_now);
        for (auto &p : cloud_pre->points) {
            auto p_new = T * p.getVector4fMap();
            p.x = p_new.x();
            p.y = p_new.y();
            p.z = p_new.z();
        }
    }

    void clouds_copy() {
        this->cloud_queue_copy.clear();
        for (auto p : this->cloud_queue) {
            pcl::PointCloud<PointLidar>::Ptr tmp(new pcl::PointCloud<PointLidar>);
            pcl::copyPointCloud(*p, *tmp);
            this->cloud_queue_copy.push_back(tmp);
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

        this->clouds_copy();
        auto cloud_now = cloud_queue_copy[cloud_queue_copy.size() / 2];
        set_Q_points(cloud_now);

//        std::vector<std::thread> transform_threads;
//        for (auto cloud_tmp : this->cloud_queue_copy)
//            if (cloud_tmp != cloud_now)
//                transform_threads.emplace_back(&Node::transform, this, std::ref(cloud_tmp), std::ref(cloud_now));
//        for (auto &t : transform_threads)
//            t.join();
//        std::vector<std::thread> dst_threads;
//        for (auto cloud_dst : this->cloud_queue_copy)
//            dst_threads.emplace_back(&Node::dst, this, std::ref(cloud_dst));
//        for (auto &t : dst_threads)
//            t.join();

        for (auto cloud_tmp : this->cloud_queue_copy)
            if (cloud_tmp != cloud_now)
                this->transform(cloud_tmp, cloud_now);
        for (auto cloud_tmp : this->cloud_queue_copy)
            this->dst(cloud_tmp);

        // 一致性评估
        for (auto cloud_ca : this->cloud_queue_copy)
            consistency_assessment(cloud_now, cloud_ca);

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_show(new pcl::PointCloud<pcl::PointXYZRGB>);
        for (auto p : cloud_now->points) {
            // std::cout << p.cons << " " << p.conf << " " << p.unc << "\n";
//            if (p.seq != header.seq)
//                continue;
            if ((p.conf > p.cons) || (p.conf > p.unc)) {
                pcl::PointXYZRGB tmp;
                tmp.x = p.x, tmp.y = p.y, tmp.z = p.z;
                tmp.r = 0, tmp.g = 255, tmp.b = 0;
                cloud_show->push_back(tmp);
            }
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

