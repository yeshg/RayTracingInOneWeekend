#ifndef SPHEREH
#define SPHEREH

#include "hittable.h"

class sphere: public hittable  {
    public:
        __device__ sphere() {}
        __device__ sphere(vec3 cen, float r, material *m) : center(cen), radius(r), mat_ptr(m)  {};
        __device__ virtual bool hit(const ray& r, float tmin, float tmax, hit_record& rec) const;
        vec3 center;
        float radius;
        material *mat_ptr;
};

__device__ bool sphere::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
    vec3 oc = r.origin() - center;
    float a = dot(r.direction(), r.direction());
    float b = dot(oc, r.direction());
    float c = dot(oc, oc) - radius*radius;
    float discriminant = b*b - a*c;
    if (discriminant > 0) {
        float temp = (-b - sqrt(discriminant))/a;
        if (temp < t_max && temp > t_min) {
            rec.t = temp;
            rec.p = r.point_at_parameter(rec.t);
            rec.normal = (rec.p - center) / radius;
            rec.mat_ptr = mat_ptr;
            return true;
        }
        temp = (-b + sqrt(discriminant)) / a;
        if (temp < t_max && temp > t_min) {
            rec.t = temp;
            rec.p = r.point_at_parameter(rec.t);
            rec.normal = (rec.p - center) / radius;
            rec.mat_ptr = mat_ptr;
            return true;
        }
    }
    return false;
}


#endif







// #ifndef SPHERE_H
// #define SPHERE_H

// #include "hittable.h"
// #include "vec3.h"

// class sphere : public hittable {
//     public:
//         __device__ sphere() {}
//         __device__ sphere(point3 cen, float r, material* m)
//             : center(cen), radius(r), mat_ptr(m) {};

//         __device__ virtual bool hit(
//             const ray& r, float t_min, float t_max, hit_record& rec) const override;

//     public:
//         point3 center;
//         float radius;
//         material* mat_ptr;
// };

// __device__ bool sphere::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
//     vec3 oc = r.origin() - center;
//     auto a = r.direction().length_squared();
//     auto half_b = dot(oc, r.direction());
//     auto c = oc.length_squared() - radius*radius;

//     auto discriminant = half_b*half_b - a*c;
//     if (discriminant < 0) return false;
//     auto sqrtd = sqrt(discriminant);

//     // Find the nearest root that lies in the acceptable range.
//     auto root = (-half_b - sqrtd) / a;
//     if (root < t_min || t_max < root) {
//         root = (-half_b + sqrtd) / a;
//         if (root < t_min || t_max < root)
//             return false;
//     }

//     rec.t = root;
//     rec.p = r.at(rec.t);
//     vec3 outward_normal = (rec.p - center) / radius;
//     rec.set_face_normal(r, outward_normal);
//     rec.mat_ptr = mat_ptr;

//     return true;
// }

// #endif
