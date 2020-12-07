#ifndef RAYH
#define RAYH
#include "vec3.h"

class ray
{
    public:
        __device__ ray() {}
        __device__ ray(const vec3& a, const vec3& b) { A = a; B = b; }
        __device__ vec3 origin() const       { return A; }
        __device__ vec3 direction() const    { return B; }
        __device__ vec3 point_at_parameter(float t) const { return A + t*B; }

        vec3 A;
        vec3 B;
};

#endif



// #ifndef RAY_H
// #define RAY_H

// #include "vec3.h"

// class ray {
//     public:
//         __device__ ray() {}
//         __device__ ray(const point3& origin, const vec3& direction)
//             : orig(origin), dir(direction)
//         {}

//         __device__ point3 origin() const  { return orig; }
//         __device__ vec3 direction() const { return dir; }

//         __device__ point3 at(float t) const {
//             return orig + t*dir;
//         }

//     public:
//         point3 orig;
//         vec3 dir;
// };

// #endif
