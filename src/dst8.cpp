//
// Created by cdj on 2020/3/22.
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
)

class Node {
private:
    ros::NodeHandle nh;
    ros::Subscriber sub;
    ros::Publisher pub;
    pcl::console::TicToc tim;
    std::vector<pcl::PointCloud<PointLidar>::Ptr> cloud_queue;
    std::vector<PointLidar> Q_points;
public:
    Node() {
        // Create a ROS subscriber for the input point cloud
        // sub = nh.subscribe("/kitti/velo/pointcloud", 1, &Node::cloud_cb, this);  // kitti
        sub = nh.subscribe("/velodyne_points", 1, &Node::cloud_cb, this);
        // Create a ROS publisher for the output point cloud
        pub = nh.advertise<pcl::PointCloud<pcl::PointXYZRGB>>("/tmp", 1);
    };

    static void ground_remove(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud) {
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_tmp(new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl::octree::OctreePointCloud<pcl::PointXYZRGB> octree(0.4f); // 设置八叉树的分辨率为 0.4 m
        octree.setInputCloud(cloud);
        octree.addPointsFromInputCloud();
        float h_hat = 0.0;
        float s = 0.09;
        for (auto it = octree.breadth_begin(); it != octree.breadth_end(); ++it) {    // 其实八叉树的树叉从某种意义上来说是没有东西的
            if (it.isBranchNode())  // 如果是树枝，就跳过
                continue;
            std::vector<int> indices;
            it.getLeafContainer().getPointIndices(indices); // 树叶不可能为空
            if (indices.size() < 3)  // 降噪
                continue;
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

    static float angle(PointLidar &P, PointLidar &Q) {
        auto OP = P.getVector3fMap();
        auto OQ = Q.getVector3fMap();
        float theta = acos(OP.normalized().dot(OQ.normalized())) * 180 / M_PI;
        return theta;
    }

    void dst_leaf(pcl::PointCloud<PointLidar>::Ptr &cloud, const std::vector<int> &indices, int idx) {
        auto P = cloud->points[idx];  // 取体素中的一个点作为代表
        for (auto Q : this->Q_points) { // 仅计算体素中的一个点的dst
            // if (angle(P, Q) > 45)  // 为了去除移动物体背景的阴影
            fusion(P, Q);
        }
        for (auto idx_ : indices) {  // 体素中其它的点全都按照第一个点的dst来进行赋值
            auto &p = cloud->points[idx_];  // idx instead of indices[idx]t
            p.e = P.e, p.o = P.o, p.u = P.u;
            // std::cout << p.e << " " << p.o << " " << p.u << "\n";
        }
    }

    void
    dst(pcl::PointCloud<PointLidar>::Ptr &cloud) {
        pcl::octree::OctreePointCloudSearch<PointLidar> octree(0.2f); // 设置八叉树的分辨率为 0.4 m
        octree.setInputCloud(cloud);
        octree.addPointsFromInputCloud();

        std::vector<std::thread> threads;
        for (auto it = octree.leaf_begin(); it != octree.leaf_end(); it++) {
            std::vector<int> indices;
            it.getLeafContainer().getPointIndices(indices);
            int idx = it.getLeafContainer().getPointIndex();  // 取体素中的一个点作为代表
            threads.emplace_back(&Node::dst_leaf, this, std::ref(cloud), indices, idx);
        }
        for (auto &t : threads)
            t.join();
    }

    static void consistency_relations(PointLidar *P, PointLidar *P_) {
        float E1 = P->e, O1 = P->o, U1 = P->u;
        float E2 = P_->e, O2 = P_->o, U2 = P_->u;
        P->conf = E1 * O2 + O1 * E2;
        P->cons = E1 * E2 + O1 * O2 + U1 * U2;
        P->unc = U1 * (E2 + O2) + U2 * (E1 + O1);
    }

    static void avg_eou(PointLidar &P, PointLidar Q) {
        // std::cout << P.e << " " << P.o << " " << P.u << "\n";
        P.e = (P.e + Q.e) / 2;
        P.o = (P.o + Q.o) / 2;
        P.u = (P.u + Q.u) / 2;
    }

    static pcl::PointCloud<PointLidar>::Ptr
    consistency_assessment(pcl::PointCloud<PointLidar>::Ptr &cloud_now, pcl::PointCloud<PointLidar>::Ptr &cloud_pre) {
        pcl::PointCloud<PointLidar>::Ptr cloud(new pcl::PointCloud<PointLidar>);
        for (auto p : cloud_now->points)
            cloud->points.push_back(p);
        for (auto p : cloud_pre->points)
            cloud->points.push_back(p);
        pcl::octree::OctreePointCloudSearch<PointLidar> octree(0.2f); // 设置八叉树的分辨率为 0.4 m
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
            if (diff.size() < 2)
                continue;
            if (diff[0].seq > diff[1].seq) {
                now = &diff[0];
                pre = &diff[1];
            } else {
                now = &diff[1];
                pre = &diff[0];
            }
            consistency_relations(now, pre);
//            std::cout << now->e << " " << now->o << " " << now->u << "\n";
//             std::cout << now->cons << " " << now->conf << " " << now->unc << "\n\n";
            for (auto idx : indices) {  // 体素中其它的点全都按照这一个点来进行赋值
                auto &p = cloud->points[idx];  // idx instead of indices[idx]
                p.cons = now->cons, p.conf = now->conf, p.unc = now->unc;
            }
        }
        return cloud;
    }

    void set_Q_points() {
        if (!this->Q_points.empty())
            this->Q_points.clear();
        auto cloud = this->cloud_queue[0];
        pcl::octree::OctreePointCloudSearch<PointLidar> octree(2.0f); // 设置八叉树的分辨率
        octree.setInputCloud(cloud);
        octree.addPointsFromInputCloud();
        for (auto it = octree.leaf_begin(); it != octree.leaf_end(); it++) {
            int idx = it.getLeafContainer().getPointIndex();
            auto Q = cloud->points[idx];
            if (Q.getVector3fMap().norm() > 1.0f)
                this->Q_points.push_back(Q);
            if (this->Q_points.size() >= 10)
                return;
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
        // ground_remove(_cloud);
        // change to PointLidar type
        pcl::PointCloud<PointLidar>::Ptr cloud(new pcl::PointCloud<PointLidar>);
        for (auto p : _cloud->points) {
//            if ((p.x < 0) || (p.getVector3fMap().norm() > 30))
            if (p.getVector3fMap().norm() > 30)
                continue;
            PointLidar tmp{};
            tmp.x = p.x, tmp.y = p.y, tmp.z = p.z;
            tmp.e = 0.0, tmp.o = 0.0, tmp.u = 1.0;
            tmp.cons = 0.0, tmp.conf = 0.0, tmp.unc = 1.0;
            tmp.seq = header.seq;   // 用时间来标
            cloud->push_back(tmp);
        }

        cloud_queue.push_back(cloud);
        if (cloud_queue.size() < 2) {
            return;
        }

        auto cloud_pre = cloud_queue[0];
        auto cloud_now = cloud_queue[1];
        set_Q_points();
//        std::thread dst_1(&Node::dst, this, std::ref(cloud_pre));
//        std::thread dst_2(&Node::dst, this, std::ref(cloud_now));
//        dst_1.join();
//        dst_2.join();
        dst(cloud_pre);
        dst(cloud_now);

        pcl::PointCloud<PointLidar>::Ptr cloud_ = consistency_assessment(cloud_now, cloud_pre);

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_show(new pcl::PointCloud<pcl::PointXYZRGB>);
        for (auto p : cloud_->points) {
//            std::cout << p.cons << " " << p.conf << " " << p.unc << "\n";
            if (p.seq != header.seq)
                continue;
//            if (abs(p.y) > 5.0 || p.x > 12.0)
//                continue;
            if ((p.conf > p.cons) || (p.unc > p.cons)) {
                pcl::PointXYZRGB tmp;
                tmp.x = p.x, tmp.y = p.y, tmp.z = p.z;
                tmp.r = 0, tmp.g = 255, tmp.b = 0;
                cloud_show->push_back(tmp);
            } else {
                pcl::PointXYZRGB tmp;
                tmp.x = p.x, tmp.y = p.y, tmp.z = p.z;
                tmp.r = 0, tmp.g = 0, tmp.b = 0;
                cloud_show->push_back(tmp);
            }
//            std::cout << p.e << " " << p.o << " " << p.u << "\n";
//            std::cout << p.conf << " " << p.cons << " " << p.unc << "\n\n";
//            pcl::PointXYZRGB tmp;
//            tmp.x = p.x, tmp.y = p.y, tmp.z = p.z;
//            tmp.r = 255*p.e, tmp.g = 255*p.o, tmp.b = 255*p.u;
//            // std::cout << p.e << " " << p.o << " " << p.u << "\n";
//            cloud_show->push_back(tmp);
        }
        ground_remove(cloud_show);

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
