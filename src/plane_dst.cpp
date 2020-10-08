//
// Created by cdj on 2020/3/22.
//
#define PCL_NO_PRECOMPILE

#include <ros/ros.h>
#include <visualization_msgs/MarkerArray.h>
#include <visualization_msgs/Marker.h>
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
//#include <pcl/kdtree/kdtree_flann.h>
//#include <pcl/filters/voxel_grid.h>

#include <tuple>
#include <thread>
//#include <mutex>
#include <atomic>

#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>

//typedef pcl::octree::OctreeBase<pcl::octree::OctreeContainerPointIndices>::LeafNodeIterator LeafNodeIterator;
using namespace std;

struct PointLidar {
    PCL_ADD_POINT4D;                  // preferred way of adding a XYZ+padding
    float intensity;  // intensity
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
                                   (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)
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
    ros::Publisher markers_pub;
    pcl::console::TicToc tim;
    std::vector<pcl::PointCloud<PointLidar>::Ptr> cloud_queue;
    std::vector<PointLidar> Q_points;
public:
    Node() {
        // Create a ROS subscriber for the input point cloud
        // sub = nh.subscribe("/kitti/velo/pointcloud", 1, &Node::cloud_cb, this);  // kitti
        sub = nh.subscribe("/merge_points", 1, &Node::cloud_cb, this);
        // Create a ROS publisher for the output point cloud
        pub = nh.advertise<pcl::PointCloud<pcl::PointXYZRGB>>("/motion_points", 1);
        markers_pub = nh.advertise<visualization_msgs::MarkerArray>("/motion_markers", 1);
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
        /*
         * 一致性评估
         * 代码最早写于年初，大概四月份
         * 九月份时，在商飞开始第二次使用
         */
        pcl::PointCloud<PointLidar>::Ptr cloud(new pcl::PointCloud<PointLidar>);
        for (auto p : cloud_now->points)
            cloud->points.push_back(p);
        for (auto p : cloud_pre->points)
            cloud->points.push_back(p);
        pcl::octree::OctreePointCloudSearch<PointLidar> octree(0.2f); // 设置八叉树的分辨率为 0.4 m
        octree.setInputCloud(cloud);
        octree.addPointsFromInputCloud();
        // 遍历
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
        auto cloud = this->cloud_queue[0];  // 利用雷达点云队列里的第一帧
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
        pcl::PointCloud<pcl::PointXYZI>::Ptr _cloud(new pcl::PointCloud<pcl::PointXYZI>);
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
            tmp.intensity = p.intensity;
            tmp.e = 0.0, tmp.o = 0.0, tmp.u = 1.0;
            tmp.cons = 0.0, tmp.conf = 0.0, tmp.unc = 1.0;
            tmp.seq = header.seq;   // 用时间来标
            cloud->push_back(tmp);
        }

        cloud_queue.push_back(cloud);
        set_Q_points();  // 我感觉随便什么时候选点都可以，但是我这里写的是会更新的
        if (cloud_queue.size() < 2) {
            dst(cloud);
            return;
        }

        auto cloud_pre = cloud_queue[0];
        auto cloud_now = cloud_queue[1];
        dst(cloud_now);  // 进行 DST 运算，只算最新一帧就可以了，入队后，下一次就变成前一帧了

        pcl::PointCloud<PointLidar>::Ptr cloud_ = consistency_assessment(cloud_now, cloud_pre);
        // 分割线，上面的是移动点检测算法，cloud_ 点云中为最新帧的所有点

        // 先将移动点和静态点分开存放
        pcl::PointCloud<PointLidar>::Ptr cloud_motion(new pcl::PointCloud<PointLidar>);
        pcl::PointCloud<PointLidar>::Ptr cloud_static(new pcl::PointCloud<PointLidar>);
        for (auto p : cloud_->points) {
//            std::cout << p.cons << " " << p.conf << " " << p.unc << "\n";
            if (p.seq != header.seq)
                continue;
//            if (abs(p.y) > 5.0 || p.x > 12.0)
//                continue;
            PointLidar tmp{};
            tmp.x = p.x, tmp.y = p.y, tmp.z = p.z;
            tmp.intensity = p.intensity;
            if ((p.conf > p.cons) || (p.unc > p.cons))
                cloud_motion->push_back(tmp);
            else
                cloud_static->push_back(tmp);
        }
        cout << "移动点数量：" << cloud_motion->size() << "，静止点数量：" << cloud_static->size() << endl;

        // TODO 做一个平面去除

        // 聚类
        // Creating the KdTree object for the search method of the extraction
        pcl::search::KdTree<PointLidar>::Ptr tree(new pcl::search::KdTree<PointLidar>);
        tree->setInputCloud(cloud_motion);

        std::vector<pcl::PointIndices> cluster_indices;
        pcl::EuclideanClusterExtraction<PointLidar> ec;
        ec.setClusterTolerance(0.2); // 聚类 tolerance 距离，单位 m
        ec.setMinClusterSize(50); // 聚类点最小数量
        ec.setMaxClusterSize(25000);
        ec.setSearchMethod(tree);
        ec.setInputCloud(cloud_motion);
        ec.extract(cluster_indices);

        // 根据聚类的结果发布 markers
        visualization_msgs::MarkerArray ma;  // markers 数组
        int count = 0;
        visualization_msgs::Marker clear_marker;
        clear_marker.header.frame_id = "/merge";
        clear_marker.header.stamp = ros::Time::now();
        clear_marker.id = count++;
        clear_marker.action = visualization_msgs::Marker::DELETEALL;
        ma.markers.push_back(clear_marker);
        for (const auto &cluster : cluster_indices) {  // 每次遍历一个聚类
            pcl::PointCloud<PointLidar>::Ptr cloud_cluster(new pcl::PointCloud<PointLidar>);
            visualization_msgs::Marker marker;
            float xmax = -100, xmin = 100, ymax = -100, ymin = 100, zmax = -100, zmin = 100;
            float intensity_sum = 0;
            for (auto idx : cluster.indices) {
                auto p = cloud_motion->points[idx];
                xmax = max(xmax, p.x);
                xmin = min(xmin, p.x);
                ymax = max(ymax, p.y);
                ymin = min(ymin, p.y);
                zmax = max(zmax, p.z);
                zmin = min(zmin, p.z);
                intensity_sum += p.intensity;
                cloud_cluster->push_back(cloud_motion->points[idx]);
            }
            float intensity_mean = intensity_sum / cloud_cluster->size();  // 计算该聚类簇的激光强度均值
            if (zmax - zmin > 2.0 || ymax - ymin > 1.0 || xmax - xmin > 1.0)
                continue;

            marker.header.frame_id = "/merge";
            marker.header.stamp = ros::Time::now();
            marker.id = count;
            marker.action = visualization_msgs::Marker::ADD;
            marker.type = visualization_msgs::Marker::CUBE;
            marker.pose.position.x = (xmax + xmin) / 2;
            marker.pose.position.y = (ymax + ymin) / 2;
            marker.pose.position.z = (zmax + zmin) / 2;
            marker.scale.x = xmax - xmin;
            marker.scale.y = ymax - ymin;
            marker.scale.z = zmax - zmin;
            marker.color.r = 1.0f;
            marker.color.g = 0.0f;
            marker.color.b = 0.0f;
            marker.color.a = 1.0;
            ma.markers.push_back(marker);

            cloud_cluster->width = cloud_cluster->size();
            cloud_cluster->height = 1;
            cloud_cluster->is_dense = true;
            cout << count << "号类点的数量：" << cloud_cluster->size() << "，" << "激光强度均值：" << intensity_mean << endl;
            count++;
        }
        markers_pub.publish(ma);

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
                tmp.r = 255, tmp.g = 255, tmp.b = 255;
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
        // ground_remove(cloud_show);

        cloud_show->header = header;
        pub.publish(cloud_show);

        // 这个点云队列里，我存了2个点云，如果每次删除第一帧，则比较的是前后两帧
        // cloud_queue.erase(cloud_queue.begin());
        // 如果删除的最后一帧，则相当于把第一帧当做静止的背景来使用
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
