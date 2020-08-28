#ifndef OCTREE_TYPES_H
#define OCTREE_TYPES_H

#include <cstdint>
#include <vector>
#include <iostream>

struct OctreeParams {
        OctreeParams() : bucket_size(32), min_extent(0.f) {}
        uint32_t bucket_size;
        double min_extent;
};

struct Octant {
        Octant() 
        : is_leaf(true), x(0),y(0),z(0), extent(0),start(0),end(0),size(0)
        {
                for (size_t i = 0; i < 8; i++) child[i] = -1;
        }

        bool is_leaf;

        // bounding box of the octant needed for overlap and contains tests...
        double x, y, z;  // center
        double extent;   // half of side-length

        uint32_t start, end;  // start and end in succ_
        uint32_t size;        // number of points

        // Octant* child[8];
        int32_t child[8];
};

struct Point {
        double x, y, z;
        double dx, dy, dz;
};

class Octree {
public:
        Octree() : root_(-1), data_(nullptr) {}
        ~Octree() {}
        Octree(Octree&) = delete;
        Octree& operator=(Octree &rhs) = delete;

        void initialize(const Point *data, uint32_t size, const OctreeParams &params=OctreeParams());
        void radius_nn(const Point &query, double radius, std::vector<int64_t> &result) const;

private:
        int32_t create_octant(double x, double y, double z, double extent, 
                uint32_t start, uint32_t end, uint32_t size);
        bool overlaps(const Point &query, double radius, const Octant &o) const;
        bool contains(const Point &query, double radius, const Octant &o) const;
        void radius_nn(const Octant &o, const Point &query, double radius, std::vector<int64_t> &) const;

        struct OctreeParams params_;
        // struct Octant *root_;
        std::vector<struct Octant> octants_;
        int32_t root_;
        const struct Point *data_;
        uint32_t ndata_;
        std::vector<uint32_t> successor_;
};

#endif /* OCTREE_TYPES_H */
