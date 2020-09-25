//
// Created by cdj on 2020/3/11.
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
    double e;    // empty
    double o;    // occupied
    double u;    // unknown
    double conf;
    double cons;
    double unc;
    // PCL_MAKE_ALIGNED_OPERATOR_NEW     // make sure our new allocators are aligned
} EIGEN_ALIGN16;                    // enforce SSE padding for correct memory alignment

POINT_CLOUD_REGISTER_POINT_STRUCT (PointLidar, (float, x, x)(float, y, y)(float, z, z)
        (double, e, e)
        (double, o, o)
        (double, u, u)
        (double, conf, conf)
        (double, cons, cons)
        (double, unc, unc)
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

    std::tuple<float, float, float> m(PointLidar P, PointLidar Q) {
        auto OQ = Q.getVector3fMap();
        auto OP = P.getVector3fMap();
        float OP_scalar_projection = OP.dot(OQ.normalized());
        float OQ_norm = OQ.norm();
        double e(0.0), o(0.0), u(1.0);
        if (OP_scalar_projection < 0) {
            e = 0.0;
            o = 0.0;
            u = 1.0;
        } else if (OP_scalar_projection < OQ_norm) {
            e = 1.0;
            o = 0.0;
            u = 0.0;
        } else {
            e = 0.0;
            double r = OP_scalar_projection - OQ_norm;  // 因为在体素内选点，所以这个值比较小
            o = exp(-r*r / 2);
            u = 1 - e - o;
            // isnum(u);
        }
        // std::cout << e << " " << o << " " << u << std::endl;
        return {e, o, u};
    }

    void fusion(PointLidar &P, PointLidar Q) {
        double e1 = P.e, o1 = P.o, u1 = P.u;
        auto eou = m(P, Q);
        double e2 = std::get<0>(eou), o2 = std::get<1>(eou), u2 = std::get<2>(eou);
        double K = o1*e2 + e1*o2;
        // 我假设这种 K = 0 的情况是第一次融合，那就不融合了，直接将这次算出来的结果赋给它
        if (K == 0) {
            P.e = e2;
            P.o = o2;
            P.u = u2;
            return;
        }
        P.e = (e1 * e2 + e1 * u2 + u1 * e2) / K;
        P.o = (o1 * o2 + o1 * u2 + u1 * o2) / K;
        P.u = (u1 * u2) / K;
//        std::cout << e1 << " " << o1 << " " << u1 << std::endl;
//        std::cout << e2 << " " << o2 << " " << u2 << std::endl;
//        std::cout << K << std::endl;
//        std::cout << P.e << " " << P.o << " " << P.u << std::endl;
//        std::cout << std::endl;
    }

    pcl::PointCloud<PointLidar>::Ptr
    DST_octree(const pcl::PointCloud<PointLidar>::Ptr &cloud_ref, const pcl::PointCloud<PointLidar>::Ptr &cloud) {
        pcl::octree::OctreePointCloudSearch<PointLidar> octree(1.0f); // 设置八叉树的分辨率为 0.4 m
        octree.setInputCloud(cloud_ref);
        octree.addPointsFromInputCloud();

        pcl::PointCloud<PointLidar>::Ptr cloud_compute(new pcl::PointCloud<PointLidar>);
        for (auto p : cloud->points)
            cloud_compute->push_back(p);

//        for (auto &P : cloud_compute->points) {
//            // Neighbors within voxel search
//            std::vector<int> pointIdxVec;
//            if (octree.voxelSearch(P, pointIdxVec)) {
//                int K = 10;
//                for (int i = 0; i < K; i++) {
//                    int idr = rand() % pointIdxVec.size();  // 随机抽取体素内的临近点
//                    int idx = pointIdxVec[idr];
//                    auto Q = cloud_ref->points[idx];
//                    fusion(P, Q);
//                }
//            }
//        }
        for (auto &P : cloud_compute->points) {
            // Neighbors within radius search
            std::vector<int> pointIdxRadiusSearch;
            std::vector<float> pointRadiusSquaredDistance;
            float radius = 2.0f;
            if (octree.radiusSearch (P, radius, pointIdxRadiusSearch, pointRadiusSquaredDistance) > 0) {
                int count = 0;
                while(count < 10) {
                    int idr = rand() % pointIdxRadiusSearch.size();  // 随机抽取体素内的临近点
                    int idx = pointIdxRadiusSearch[idr];
                    auto Q = cloud_ref->points[idx];
                    fusion(P, Q);
                    count++;
                }
            }
        }
        return cloud_compute;
    }

    pcl::PointCloud<PointLidar>::Ptr
    DST_kdtree(const pcl::PointCloud<PointLidar>::Ptr &cloud_ref, const pcl::PointCloud<PointLidar>::Ptr &cloud) {

        pcl::PointCloud<PointLidar>::Ptr cloud_compute(new pcl::PointCloud<PointLidar>);
        // Create the filtering object
        pcl::VoxelGrid<PointLidar> sor;
        sor.setInputCloud (cloud_ref);
        sor.setLeafSize (0.4f, 0.4f, 0.4f);
        sor.filter (*cloud_compute);

        pcl::KdTreeFLANN<PointLidar> kdtree;
        kdtree.setInputCloud (cloud_compute);

        for (auto &P : cloud->points) {
            // K nearest neighbor search
            int K = 10;
            std::vector<int> pointIdxNKNSearch(K);
            std::vector<float> pointNKNSquaredDistance(K);
            if ( kdtree.nearestKSearch (P, K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0 ) {
                for (auto idx : pointIdxNKNSearch) {
                    auto Q = cloud_ref->points[idx];
                    fusion(P, Q);
                }
            }
        }
        return cloud_compute;
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
            tmp.e = tmp.o = 0.0, tmp.u = 1.0;
            tmp.cons = tmp.conf = tmp.unc = 0.0;
            cloud->push_back(tmp);
        }

        cloud_queue.push_back(cloud);
        if (cloud_queue.size() < 3) {
            return;
        }

        auto cloud0 = cloud_queue[0];
        auto cloud1 = cloud_queue[1];
        auto cloud2 = cloud_queue[2];

        auto dst_20 = DST_kdtree(cloud2, cloud0);
        auto dst_21 = DST_kdtree(cloud2, cloud1);
        for (auto i = 0; i < cloud2->size(); i++) { // now
            auto &P = cloud2->points[i];
            auto P1 = dst_20->points[i];
            auto P2 = dst_21->points[i];
            double e1 = P1.e, o1 = P1.o, u1 = P1.u;
            double e2 = P2.e, o2 = P2.o, u2 = P2.u;
//            std::cout << e1 << " " << o1 << " " << u1 << std::endl;
//            std::cout << e2 << " " << o2 << " " << u2 << std::endl;
//            std::cout << std::endl;
            P.conf = e1*o2 + o1*e2;
            P.cons = e1*e2 + o1*o2 + u1*u2;
            P.unc = u1*(e2+o2) + u2*(e1+o1);
//            std::cout << P.conf << " " << P.cons << " " << P.unc << std::endl;
        }

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_show(new pcl::PointCloud<pcl::PointXYZRGB>);
        for (auto p : cloud2->points) {
            // std::cout << p.conf << " " << p.cons << " " << p.unc << std::endl;
            if (p.conf > p.cons) {
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




