#ifndef VEC3H
#define VEC3H

#include <math.h>
#include <stdlib.h>
#include <iostream>

class vec3  {


public:
    __host__ __device__ vec3() {}
    __host__ __device__ vec3(float e0, float e1, float e2) { e[0] = e0; e[1] = e1; e[2] = e2; }
    __host__ __device__ inline float x() const { return e[0]; }
    __host__ __device__ inline float y() const { return e[1]; }
    __host__ __device__ inline float z() const { return e[2]; }
    __host__ __device__ inline float r() const { return e[0]; }
    __host__ __device__ inline float g() const { return e[1]; }
    __host__ __device__ inline float b() const { return e[2]; }

    __host__ __device__ inline const vec3& operator+() const { return *this; }
    __host__ __device__ inline vec3 operator-() const { return vec3(-e[0], -e[1], -e[2]); }
    __host__ __device__ inline float operator[](int i) const { return e[i]; }
    __host__ __device__ inline float& operator[](int i) { return e[i]; };

    __host__ __device__ inline vec3& operator+=(const vec3 &v2);
    __host__ __device__ inline vec3& operator-=(const vec3 &v2);
    __host__ __device__ inline vec3& operator*=(const vec3 &v2);
    __host__ __device__ inline vec3& operator/=(const vec3 &v2);
    __host__ __device__ inline vec3& operator*=(const float t);
    __host__ __device__ inline vec3& operator/=(const float t);

    __host__ __device__ inline float length() const { return sqrt(e[0]*e[0] + e[1]*e[1] + e[2]*e[2]); }
    __host__ __device__ inline float squared_length() const { return e[0]*e[0] + e[1]*e[1] + e[2]*e[2]; }
    __host__ __device__ inline void make_unit_vector();


    float e[3];
};



inline std::istream& operator>>(std::istream &is, vec3 &t) {
    is >> t.e[0] >> t.e[1] >> t.e[2];
    return is;
}

inline std::ostream& operator<<(std::ostream &os, const vec3 &t) {
    os << t.e[0] << " " << t.e[1] << " " << t.e[2];
    return os;
}

__host__ __device__ inline void vec3::make_unit_vector() {
    float k = 1.0 / sqrt(e[0]*e[0] + e[1]*e[1] + e[2]*e[2]);
    e[0] *= k; e[1] *= k; e[2] *= k;
}

__host__ __device__ inline vec3 operator+(const vec3 &v1, const vec3 &v2) {
    return vec3(v1.e[0] + v2.e[0], v1.e[1] + v2.e[1], v1.e[2] + v2.e[2]);
}

__host__ __device__ inline vec3 operator-(const vec3 &v1, const vec3 &v2) {
    return vec3(v1.e[0] - v2.e[0], v1.e[1] - v2.e[1], v1.e[2] - v2.e[2]);
}

__host__ __device__ inline vec3 operator*(const vec3 &v1, const vec3 &v2) {
    return vec3(v1.e[0] * v2.e[0], v1.e[1] * v2.e[1], v1.e[2] * v2.e[2]);
}

__host__ __device__ inline vec3 operator/(const vec3 &v1, const vec3 &v2) {
    return vec3(v1.e[0] / v2.e[0], v1.e[1] / v2.e[1], v1.e[2] / v2.e[2]);
}

__host__ __device__ inline vec3 operator*(float t, const vec3 &v) {
    return vec3(t*v.e[0], t*v.e[1], t*v.e[2]);
}

__host__ __device__ inline vec3 operator/(vec3 v, float t) {
    return vec3(v.e[0]/t, v.e[1]/t, v.e[2]/t);
}

__host__ __device__ inline vec3 operator*(const vec3 &v, float t) {
    return vec3(t*v.e[0], t*v.e[1], t*v.e[2]);
}

__host__ __device__ inline float dot(const vec3 &v1, const vec3 &v2) {
    return v1.e[0] *v2.e[0] + v1.e[1] *v2.e[1]  + v1.e[2] *v2.e[2];
}

__host__ __device__ inline vec3 cross(const vec3 &v1, const vec3 &v2) {
    return vec3( (v1.e[1]*v2.e[2] - v1.e[2]*v2.e[1]),
                (-(v1.e[0]*v2.e[2] - v1.e[2]*v2.e[0])),
                (v1.e[0]*v2.e[1] - v1.e[1]*v2.e[0]));
}


__host__ __device__ inline vec3& vec3::operator+=(const vec3 &v){
    e[0]  += v.e[0];
    e[1]  += v.e[1];
    e[2]  += v.e[2];
    return *this;
}

__host__ __device__ inline vec3& vec3::operator*=(const vec3 &v){
    e[0]  *= v.e[0];
    e[1]  *= v.e[1];
    e[2]  *= v.e[2];
    return *this;
}

__host__ __device__ inline vec3& vec3::operator/=(const vec3 &v){
    e[0]  /= v.e[0];
    e[1]  /= v.e[1];
    e[2]  /= v.e[2];
    return *this;
}

__host__ __device__ inline vec3& vec3::operator-=(const vec3& v) {
    e[0]  -= v.e[0];
    e[1]  -= v.e[1];
    e[2]  -= v.e[2];
    return *this;
}

__host__ __device__ inline vec3& vec3::operator*=(const float t) {
    e[0]  *= t;
    e[1]  *= t;
    e[2]  *= t;
    return *this;
}

__host__ __device__ inline vec3& vec3::operator/=(const float t) {
    float k = 1.0/t;

    e[0]  *= k;
    e[1]  *= k;
    e[2]  *= k;
    return *this;
}

__host__ __device__ inline vec3 unit_vector(vec3 v) {
    return v / v.length();
}

#endif


// #ifndef VEC3_H
// #define VEC3_H

// #include <cmath>
// #include <iostream>

// #include "common.h"

// using std::sqrt;

// #ifndef RANDVEC3
// #define RANDVEC3 vec3(curand_uniform(local_rand_state),curand_uniform(local_rand_state),curand_uniform(local_rand_state))
// #endif

// class vec3 {
//     public:
//         __host__ __device__ vec3() : e{0,0,0} {}
//         __host__ __device__ vec3(float e0, float e1, float e2) : e{e0, e1, e2} {}

//         __host__ __device__ float x() const { return e[0]; }
//         __host__ __device__ float y() const { return e[1]; }
//         __host__ __device__ float z() const { return e[2]; }

//         __host__ __device__ vec3 operator-() const { return vec3(-e[0], -e[1], -e[2]); }
//         __host__ __device__ float operator[](int i) const { return e[i]; }
//         __host__ __device__ float& operator[](int i) { return e[i]; }

//         __host__ __device__ vec3& operator+=(const vec3 &v) {
//             e[0] += v.e[0];
//             e[1] += v.e[1];
//             e[2] += v.e[2];
//             return *this;
//         }

//         __host__ __device__ vec3& operator*=(const float t) {
//             e[0] *= t;
//             e[1] *= t;
//             e[2] *= t;
//             return *this;
//         }

//         __host__ __device__ vec3& operator/=(const float t) {
//             return *this *= 1/t;
//         }

//         __host__ __device__ float length() const {
//             return sqrt(length_squared());
//         }

//         __host__ __device__ float length_squared() const {
//             return e[0]*e[0] + e[1]*e[1] + e[2]*e[2];
//         }

//         __device__ inline static vec3 random(curandState *local_rand_state) {
//             return vec3(random_double(local_rand_state), random_double(local_rand_state), random_double(local_rand_state));
//         }

//         __device__ inline static vec3 random(float min, float max, curandState *local_rand_state) {
//             return vec3(random_double(min, max, local_rand_state), random_double(min, max, local_rand_state), random_double(min, max, local_rand_state));
//         }

//         __host__ __device__ bool near_zero() const {
//             // Return true if the vector is close to zero in all dimensions.
//             const auto s = 1e-8;
//             return (fabs(e[0]) < s) && (fabs(e[1]) < s) && (fabs(e[2]) < s);
//         }

//     public:
//         float e[3];
// };

// // Type aliases for vec3
// using point3 = vec3;   // 3D point
// using color = vec3;    // RGB color

// // vec3 Utility Functions

// inline std::ostream& operator<<(std::ostream &out, const vec3 &v) {
//     return out << v.e[0] << ' ' << v.e[1] << ' ' << v.e[2];
// }

// __host__ __device__ inline vec3 operator+(const vec3 &u, const vec3 &v) {
//     return vec3(u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]);
// }

// __host__ __device__ inline vec3 operator-(const vec3 &u, const vec3 &v) {
//     return vec3(u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]);
// }

// __host__ __device__ inline vec3 operator*(const vec3 &u, const vec3 &v) {
//     return vec3(u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2]);
// }

// __host__ __device__ inline vec3 operator*(float t, const vec3 &v) {
//     return vec3(t*v.e[0], t*v.e[1], t*v.e[2]);
// }

// __host__ __device__ inline vec3 operator*(const vec3 &v, float t) {
//     return t * v;
// }

// __host__ __device__ inline vec3 operator/(vec3 v, float t) {
//     return (1/t) * v;
// }

// __host__ __device__ inline float dot(const vec3 &u, const vec3 &v) {
//     return u.e[0] * v.e[0]
//          + u.e[1] * v.e[1]
//          + u.e[2] * v.e[2];
// }

// __host__ __device__ inline vec3 cross(const vec3 &u, const vec3 &v) {
//     return vec3(u.e[1] * v.e[2] - u.e[2] * v.e[1],
//                 u.e[2] * v.e[0] - u.e[0] * v.e[2],
//                 u.e[0] * v.e[1] - u.e[1] * v.e[0]);
// }

// __host__ __device__ inline vec3 unit_vector(vec3 v) {
//     return v / v.length();
// }

// // Incorrect implementation of lambertian light scattering
// __device__ vec3 random_in_unit_sphere(curandState *local_rand_state) {
//     vec3 p;
//     do {
//         p = 2.0f*RANDVEC3 - vec3(1,1,1);
//     } while (p.length_squared() >= 1.0f);
//     return p;
// }

// // For True lambertian light scattering
// __device__ vec3 random_unit_vector(curandState *local_rand_state) {
//     return unit_vector(random_in_unit_sphere(local_rand_state));
// }

// // For early hemispherical scattering
// __device__ vec3 random_in_hemisphere(const vec3& normal, curandState *local_rand_state) {
//     vec3 in_unit_sphere = random_in_unit_sphere(local_rand_state);
//     if (dot(in_unit_sphere, normal) > 0.0) // In the same hemisphere as the normal
//         return in_unit_sphere;
//     else
//         return -in_unit_sphere;
// }

// // used for generating rays with defocus blur
// __device__ vec3 random_in_unit_disk(curandState *local_rand_state) {
//     while (true) {
//         auto p = vec3(random_double(-1,1, local_rand_state), random_double(-1,1, local_rand_state), 0);
//         if (p.length_squared() >= 1) continue;
//         return p;
//     }
// }

// __device__ vec3 reflect(const vec3& v, const vec3& n) {
//     return v - 2.0f*dot(v,n)*n;
// }

// __device__ bool refract(const vec3& v, const vec3& n, float ni_over_nt, vec3& refracted) {
//     vec3 uv = unit_vector(v);
//     float dt = dot(uv, n);
//     float discriminant = 1.0f - ni_over_nt*ni_over_nt*(1-dt*dt);
//     if (discriminant > 0) {
//         refracted = ni_over_nt*(uv - n*dt) - n*sqrt(discriminant);
//         return true;
//     }
//     else
//         return false;
// }

// #endif
