#ifndef CAMERAH
#define CAMERAH

#include <curand_kernel.h>
#include "ray.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

__device__ vec3 random_in_unit_disk(curandState *local_rand_state) {
    vec3 p;
    do {
        p = 2.0f*vec3(curand_uniform(local_rand_state),curand_uniform(local_rand_state),0) - vec3(1,1,0);
    } while (dot(p,p) >= 1.0f);
    return p;
}

class camera {
public:
    __device__ camera(vec3 lookfrom, vec3 lookat, vec3 vup, float vfov, float aspect, float aperture, float focus_dist) { // vfov is top to bottom in degrees
        lens_radius = aperture / 2.0f;
        float theta = vfov*((float)M_PI)/180.0f;
        float half_height = tan(theta/2.0f);
        float half_width = aspect * half_height;
        origin = lookfrom;
        w = unit_vector(lookfrom - lookat);
        u = unit_vector(cross(vup, w));
        v = cross(w, u);
        lower_left_corner = origin  - half_width*focus_dist*u -half_height*focus_dist*v - focus_dist*w;
        horizontal = 2.0f*half_width*focus_dist*u;
        vertical = 2.0f*half_height*focus_dist*v;
    }
    __device__ ray get_ray(float s, float t, curandState *local_rand_state) {
        vec3 rd = lens_radius*random_in_unit_disk(local_rand_state);
        vec3 offset = u * rd.x() + v * rd.y();
        return ray(origin + offset, lower_left_corner + s*horizontal + t*vertical - origin - offset);
    }

    vec3 origin;
    vec3 lower_left_corner;
    vec3 horizontal;
    vec3 vertical;
    vec3 u, v, w;
    float lens_radius;
};

#endif



// #ifndef CAMERA_H
// #define CAMERA_H

// #include "common.h"
// #include <curand_kernel.h>

// class camera {
//     public:
//         __device__ camera(
//             point3 lookfrom,
//             point3 lookat,
//             vec3   vup,
//             float vfov, // vertical field-of-view in degrees
//             float aspect_ratio,
//             float aperture,
//             float focus_dist
//         ) {
//             auto theta = degrees_to_radians(vfov);
//             auto h = tan(theta/2);
//             auto viewport_height = 2.0 * h;
//             auto viewport_width = aspect_ratio * viewport_height;

//             w = unit_vector(lookfrom - lookat);
//             u = unit_vector(cross(vup, w));
//             v = cross(w, u);

//             origin = lookfrom;
//             horizontal = focus_dist * viewport_width * u;
//             vertical = focus_dist * viewport_height * v;
//             lower_left_corner = origin - horizontal/2 - vertical/2 - focus_dist*w;

//             lens_radius = aperture / 2;
//         }

//         __device__ ray get_ray(float s, float t, curandState *local_rand_state) const {
//             vec3 rd = lens_radius * random_in_unit_disk(local_rand_state);
//             vec3 offset = u * rd.x() + v * rd.y();

//             return ray(
//                 origin + offset,
//                 lower_left_corner + s*horizontal + t*vertical - origin - offset
//             );
//         }

//     private:
//         point3 origin;
//         point3 lower_left_corner;
//         vec3 horizontal;
//         vec3 vertical;
//         vec3 u, v, w;
//         float lens_radius;
// };
// #endif
