#include <cstdint>
#include <vector>
#include <cmath>
#include <iostream>
#include <memory>
#include <thread>
#include "octree_types.h"

using namespace std;

#define MAX(a, b) (a) > (b) ? (a) : (b)
#define NORM(x, y, z) sqrt((x)*(x)+(y)*(y)+(z)*(z))

void Octree::initialize(const Point *pts, uint32_t size, const OctreeParams &params)
{
        const uint32_t N = size;

        params_ = params;
        data_ = pts;
        successor_ = vector<uint32_t>(N);
        ndata_ = N;

        // determine axis-aligned bounding box.
        double min[3] = {pts[0].x, pts[0].y, pts[0].z};
        double max[3] = {pts[0].x, pts[0].y, pts[0].z};
        for (uint32_t i = 0; i < N; i++) {
                const Point &p = pts[i];
                if (p.x < min[0]) min[0] = p.x;
                if (p.y < min[1]) min[1] = p.y;
                if (p.z < min[2]) min[2] = p.z;
                if (p.x > max[0]) max[0] = p.x;
                if (p.y > max[1]) max[1] = p.y;
                if (p.z > max[2]) max[2] = p.z;
                successor_[i] = i + 1;
        }

        double ctr[3] = {min[0], min[1], min[2]};
        double max_extent = 0.5*(max[0]-min[0]);
        for (uint32_t i = 0; i < 3; i++) {
                double extent = 0.5*(max[i] - min[i]);
                if (extent > max_extent) max_extent = extent;
                ctr[i] += extent;
        }

        root_ = create_octant(ctr[0], ctr[1], ctr[2], max_extent, 0, N-1, N);
}

int32_t Octree::create_octant(double x, double y, double z, double extent,
        uint32_t start_idx, uint32_t end_idx, uint32_t size)
{
        Octant octant;
        octant.is_leaf = true;
        octant.x = x;
        octant.y = y;
        octant.z = z;
        octant.extent = extent;
        octant.start = start_idx;
        octant.end = end_idx;
        octant.size = size;

        if (size <= params_.bucket_size || extent <= 2*params_.min_extent) {
                octants_.push_back(octant);
                return octants_.size() - 1;
        }
        
        // subdivide subset of points and re-link points according to Morton codes
        vector<uint32_t> child_starts(8, 0);
        vector<uint32_t> child_ends(8, 0);
        vector<uint32_t> child_sizes(8, 0);
        uint32_t idx = start_idx;

        octant.is_leaf = false;
        // re-link disjoint child subsets
        for (uint32_t i = 0; i < size; i++) {
                const Point &p = data_[idx];
                uint32_t mortoncode = 0;

                // determine Morton code for each point
                if (p.x > x) mortoncode |= 1;
                if (p.y > y) mortoncode |= 2;
                if (p.z > z) mortoncode |= 4;

                // set child starts and update successors
                if (child_sizes[mortoncode] == 0)
                        child_starts[mortoncode] = idx;
                else
                        successor_[child_ends[mortoncode]] = idx;
                child_sizes[mortoncode]++;
                child_ends[mortoncode] = idx;
                idx = successor_[idx];
        }
        // now, we can create the child nodes
        double cex = 0.5 * extent;
        bool firsttime = true;
        uint32_t lst_child_idx = 0;
        for (uint32_t i = 0; i < 8; i++) {
                if (child_sizes[i] == 0) continue;
                double cx = x + ((i&1) > 0 ? 0.5 : -0.5) * extent;
                double cy = y + ((i&2) > 0 ? 0.5 : -0.5) * extent;
                double cz = z + ((i&4) > 0 ? 0.5 : -0.5) * extent;
                octant.child[i] = create_octant(cx, cy, cz, cex, 
                        child_starts[i], child_ends[i], child_sizes[i]);
                        
                if (firsttime)
                        octant.start = octants_[octant.child[i]].start;
                else
                        // we have to ensure that also the child ends link to the next child start.
                        successor_[octants_[octant.child[lst_child_idx]].end] = 
                                octants_[octant.child[i]].start;
                
                lst_child_idx = i;
                octant.end = octants_[octant.child[i]].end;
                firsttime = false;
        }
        octants_.push_back(octant);
        return octants_.size() - 1;
}

bool Octree::contains(const Point &query, double radius, const Octant &o) const
{
        // we exploit the symmetry to reduce the test to test
        // whether the farthest corner is inside the search ball.
        double x = abs(query.x - o.x) + o.extent;
        double y = abs(query.y - o.y) + o.extent;
        double z = abs(query.z - o.z) + o.extent;
        return (NORM(x, y, z) < radius);
}

bool Octree::overlaps(const Point &query, double radius, const Octant &o) const
{
        // we exploit the symmetry to reduce the test to testing if its inside 
        // the Minkowski sum around the positive quadrant.
        double x = abs(query.x - o.x);
        double y = abs(query.y - o.y);
        double z = abs(query.z - o.z);
        double maxdist = radius + o.extent;

        // completely outside
        if (x > maxdist || y > maxdist || z > maxdist) return false;

        // inside the surface region of the octant
        if ((x<o.extent)+(y<o.extent)+(z<o.extent) > 1) return true;

        // check the corner region && edge region
        x = MAX(x - o.extent, 0.);
        y = MAX(y - o.extent, 0.);
        z = MAX(z - o.extent, 0.);
        return (NORM(x, y, z) < radius);
}

void Octree::radius_nn(const Octant &octant, const Point &query, double radius, 
        std::vector<int64_t> &result) const
{
        // if search ball S(q,r) contains octant, simply add point indexes.
        if (contains(query, radius, octant)) {
                uint32_t idx = octant.start;
                for (uint32_t i = 0; i < octant.size; i++){
                        result.push_back(idx);
                        idx = successor_[idx];
                }
                return;
        }

        if (octant.is_leaf) {
                uint32_t idx = octant.start;
                for (uint32_t i = 0; i < octant.size; i++) {
                        const Point &p = data_[idx];
                        double dist = NORM(p.x-query.x, p.y-query.y, p.z-query.z);
                        if (dist < radius) result.push_back(idx);
                        idx = successor_[idx];
                }
                return;
        }

        // check whether child nodes are in range
        for (uint32_t c = 0; c < 8; c++) {
                if (octant.child[c] == -1) continue;
                if (!overlaps(query, radius, octants_[octant.child[c]])) continue;
                radius_nn(octants_[octant.child[c]], query, radius, result);
        }
}

void Octree::radius_nn(const Point &query, double radius, std::vector<int64_t> &result) const
{
        if (root_ == -1) return;
        radius_nn(octants_[root_], query, radius, result);
}

static struct Octree tree;
static shared_ptr<int64_t[]> csr_indices; // CSR representation
static shared_ptr<int64_t[]> csr_indptr;
static shared_ptr<double[]> csr_data;

static void _nn_radius(const double *query, size_t start, size_t end, 
        double radius, vector<int64_t> &ind, vector<int64_t> &nind)
{
        const Point *q = reinterpret_cast<const Point*>(query);
        size_t nnz = 0;

        ind.clear();
        nind.clear();
        for (size_t i = start; i < end; i++) {
                tree.radius_nn(q[i], radius, ind);
                nind.push_back(ind.size() - nnz);
                nnz = ind.size();
        }
}

extern "C" {

void nn_fit(const double *pts, size_t size, uint32_t bucket_size=32)
{
        OctreeParams prm;
        prm.bucket_size = bucket_size;
        tree.initialize(reinterpret_cast<const Point*>(pts), size, prm);
}

int64_t nn_radius(const double *query, size_t sz, double radius,
        int64_t **indices, int64_t **indptr, double **data)
{
        const size_t NTHREAD = 40;
        int64_t nnz = 0;
        size_t tail = 1;
        vector<thread> ts(NTHREAD);
        vector<vector<int64_t> > ind(NTHREAD), n_ind(NTHREAD);

        for (size_t i = 0; i < NTHREAD; i++) {
                size_t start = (sz / NTHREAD) * i;
                size_t end = (i==NTHREAD-1) ? sz : (sz / NTHREAD) * (i + 1);
                ts[i] = thread(_nn_radius, query, start, end, radius, 
                        ref(ind[i]), ref(n_ind[i]));
        }
        for (size_t i = 0; i < NTHREAD; i++)
                ts[i].join();

        // construct CSR while maintain memory usage
        for (size_t i = 0; i < NTHREAD; i++) 
                nnz += ind[i].size();
        
        csr_indices.reset(new int64_t[nnz]);
        csr_indptr.reset(new int64_t[sz+1]);
        csr_data.reset(new double[nnz]);
        csr_indptr[0] = 0;
        for (size_t i = 0; i < NTHREAD; i++) {
                copy(ind[i].begin(), ind[i].end(), csr_indices.get()+csr_indptr[tail-1]);
                for (size_t j = 0; j < n_ind[i].size(); j++) {
                        csr_indptr[tail] = csr_indptr[tail-1] + n_ind[i][j];
                        tail++;
                }
                vector<int64_t>().swap(ind[i]);
        }

        *indices = csr_indices.get();
        *indptr = csr_indptr.get();
        *data = csr_data.get();
        return nnz;
}

/* 
 * This function is not related to octree, but is put here for convenience
 */
void m_dot_v(const int64_t *__restrict ia, const int64_t *__restrict ja, 
        const double *__restrict a, size_t n, const double *__restrict x, 
        double *__restrict y, size_t nrhs)
{
#pragma omp parallel for num_threads(40)
        for (size_t i = 0; i < n; i++) {
                int64_t start = ia[i], end = ia[i+1];
                for (size_t j = 0; j < nrhs; j++) {
                        volatile double tmp = 0.;

                        for (size_t k = start; k < end; k++) {
                                tmp += a[k] * x[ja[k]*nrhs+j];
                        }
                        y[i*nrhs+j] = tmp;
                }
        }
}

}