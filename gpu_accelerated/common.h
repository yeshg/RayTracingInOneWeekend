#ifndef COMMON_H
#define COMMON_H

#include <cmath>
#include <limits>

#include <curand_kernel.h>

#ifndef RANDVEC3
#define RANDVEC3 vec3(curand_uniform(local_rand_state),curand_uniform(local_rand_state),curand_uniform(local_rand_state))
#endif

// Usings

using std::sqrt;

// Constants

const float infinity = std::numeric_limits<float>::infinity();
const float pi = 3.1415926535897932385;

// Utility Functions

__host__ __device__ inline float degrees_to_radians(float degrees) {
    return degrees * pi / 180.0;
}

__device__ inline float random_double(curandState *local_rand_state) {
    return curand_uniform(local_rand_state);
}

__device__ inline float random_double(float min, float max, curandState *local_rand_state) {
    float myrandf = curand_uniform(local_rand_state);
    myrandf *= (max - min+0.999999);
    myrandf += min;
    return myrandf;
}

__host__ __device__ inline float clamp(float x, float min, float max) {
    if (x < min) return min;
    if (x > max) return max;
    return x;
}

// Common Headers

#include "ray.h"
#include "vec3.h"

#endif
