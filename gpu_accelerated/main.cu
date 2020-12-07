#include <iostream>
#include <time.h>
#include <float.h>
#include <curand_kernel.h>
#include "vec3.h"
#include "ray.h"
#include "sphere.h"
#include "hittable_list.h"
#include "camera.h"
#include "material.h"

// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

// Matching the C++ code would recurse enough into color() calls that
// it was blowing up the stack, so we have to turn this into a
// limited-depth loop instead.  Later code in the book limits to a max
// depth of 50, so we adapt this a few chapters early on the GPU.
__device__ vec3 color(const ray& r, hittable **world, curandState *local_rand_state) {
    ray cur_ray = r;
    vec3 cur_attenuation = vec3(1.0,1.0,1.0);
    for(int i = 0; i < 50; i++) {
        hit_record rec;
        if ((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec)) {
            ray scattered;
            vec3 attenuation;
            if(rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, local_rand_state)) {
                cur_attenuation *= attenuation;
                cur_ray = scattered;
            }
            else {
                return vec3(0.0,0.0,0.0);
            }
        }
        else {
            vec3 unit_direction = unit_vector(cur_ray.direction());
            float t = 0.5f*(unit_direction.y() + 1.0f);
            vec3 c = (1.0f-t)*vec3(1.0, 1.0, 1.0) + t*vec3(0.5, 0.7, 1.0);
            return cur_attenuation * c;
        }
    }
    return vec3(0.0,0.0,0.0); // exceeded recursion
}

__global__ void rand_init(curandState *rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curand_init(1984, 0, 0, rand_state);
    }
}

__global__ void render_init(int max_x, int max_y, curandState *rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j*max_x + i;
    // Original: Each thread gets same seed, a different sequence number, no offset
    // curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
    // BUGFIX, see Issue#2: Each thread gets different seed, same sequence for
    // performance improvement of about 2x!
    curand_init(1984+pixel_index, 0, 0, &rand_state[pixel_index]);
}

__global__ void render(vec3 *fb, int max_x, int max_y, int ns, camera **cam, hittable **world, curandState *rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j*max_x + i;
    curandState local_rand_state = rand_state[pixel_index];
    vec3 col(0,0,0);
    for(int s=0; s < ns; s++) {
        float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
        float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);
        ray r = (*cam)->get_ray(u, v, &local_rand_state);
        col += color(r, world, &local_rand_state);
    }
    rand_state[pixel_index] = local_rand_state;
    col /= float(ns);
    col[0] = sqrt(col[0]);
    col[1] = sqrt(col[1]);
    col[2] = sqrt(col[2]);
    fb[pixel_index] = col;
}

#define RND (curand_uniform(&local_rand_state))

__global__ void create_world(hittable **d_list, hittable **d_world, camera **d_camera, int nx, int ny, vec3 lookfrom, vec3 lookat, curandState *rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curandState local_rand_state = *rand_state;
        d_list[0] = new sphere(vec3(0,-1000.0,-1), 1000,
                               new lambertian(vec3(0.5, 0.5, 0.5)));
        int i = 1;
        for(int a = -11; a < 11; a++) {
            for(int b = -11; b < 11; b++) {
                float choose_mat = RND;
                vec3 center(a+RND,0.2,b+RND);
                if(choose_mat < 0.8f) {
                    d_list[i++] = new sphere(center, 0.2,
                                             new lambertian(vec3(RND*RND, RND*RND, RND*RND)));
                }
                else if(choose_mat < 0.95f) {
                    d_list[i++] = new sphere(center, 0.2,
                                             new metal(vec3(0.5f*(1.0f+RND), 0.5f*(1.0f+RND), 0.5f*(1.0f+RND)), 0.5f*RND));
                }
                else {
                    d_list[i++] = new sphere(center, 0.2, new dielectric(1.5));
                }
            }
        }
        d_list[i++] = new sphere(vec3(0, 1,0),  1.0, new dielectric(1.5));
        d_list[i++] = new sphere(vec3(-4, 1, 0), 1.0, new lambertian(vec3(0.4, 0.2, 0.1)));
        d_list[i++] = new sphere(vec3(4, 1, 0),  1.0, new metal(vec3(0.7, 0.6, 0.5), 0.0));
        *rand_state = local_rand_state;
        *d_world  = new hittable_list(d_list, 22*22+1+3);

        float dist_to_focus = 10.0; (lookfrom-lookat).length();
        float aperture = 0.1;
        *d_camera   = new camera(lookfrom,
                                 lookat,
                                 vec3(0,1,0),
                                 30.0,
                                 float(nx)/float(ny),
                                 aperture,
                                 dist_to_focus);
    }
}

__global__ void free_world(hittable **d_list, hittable **d_world, camera **d_camera) {
    for(int i=0; i < 22*22+1+3; i++) {
        delete ((sphere *)d_list[i])->mat_ptr;
        delete d_list[i];
    }
    delete *d_world;
    delete *d_camera;
}

int main(int argc,char *argv[] ) {

    float lx = 13;
    float ly = 2;
    float lz = 3;
    if( argc == 4 ) {
        lx = atof(argv[1]);
        ly = atof(argv[2]);
        lz = atof(argv[3]);
    }
    vec3 lookfrom(lx, ly, lz);

    int nx = 1280;
    int ny = 720;
    int ns = 10;
    int tx = 8;
    int ty = 8;

    std::cerr << "Rendering a " << nx << "x" << ny << " image with " << ns << " samples per pixel ";
    std::cerr << "in " << tx << "x" << ty << " blocks.\n";

    int num_pixels = nx*ny;
    size_t fb_size = num_pixels*sizeof(vec3);

    // allocate FB
    vec3 *fb;
    checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

    // allocate random state
    curandState *d_rand_state;
    checkCudaErrors(cudaMalloc((void **)&d_rand_state, num_pixels*sizeof(curandState)));
    curandState *d_rand_state2;
    checkCudaErrors(cudaMalloc((void **)&d_rand_state2, 1*sizeof(curandState)));

    // we need that 2nd random state to be initialized for the world creation
    rand_init<<<1,1>>>(d_rand_state2);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // make our world of hittables & the camera
    hittable **d_list;
    int num_hittables = 22*22+1+3;
    checkCudaErrors(cudaMalloc((void **)&d_list, num_hittables*sizeof(hittable *)));
    hittable **d_world;
    checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(hittable *)));
    // vec3 lookfrom(2,13,3);
    vec3 lookat(0,0,0);
    camera **d_camera;
    checkCudaErrors(cudaMalloc((void **)&d_camera, sizeof(camera *)));
    create_world<<<1,1>>>(d_list, d_world, d_camera, nx, ny, lookfrom, lookat, d_rand_state2);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    clock_t start, stop;
    start = clock();
    // Render our buffer
    dim3 blocks(nx/tx+1,ny/ty+1);
    dim3 threads(tx,ty);
    render_init<<<blocks, threads>>>(nx, ny, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    render<<<blocks, threads>>>(fb, nx, ny,  ns, d_camera, d_world, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    stop = clock();
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "took " << timer_seconds << " seconds.\n";

    // Output FB as Image
    std::cout << "P3\n" << nx << " " << ny << "\n255\n";
    for (int j = ny-1; j >= 0; j--) {
        for (int i = 0; i < nx; i++) {
            size_t pixel_index = j*nx + i;
            int ir = int(255.99*fb[pixel_index].r());
            int ig = int(255.99*fb[pixel_index].g());
            int ib = int(255.99*fb[pixel_index].b());
            std::cout << ir << " " << ig << " " << ib << "\n";
        }
    }

    // clean up
    checkCudaErrors(cudaDeviceSynchronize());
    free_world<<<1,1>>>(d_list,d_world,d_camera);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_camera));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(d_list));
    checkCudaErrors(cudaFree(d_rand_state));
    checkCudaErrors(cudaFree(d_rand_state2));
    checkCudaErrors(cudaFree(fb));

    cudaDeviceReset();
}


// #include "common.h"

// #include "color.h"

// #include "hittable_list.h"

// #include "sphere.h"

// #include "camera.h"

// #include "material.h"

// #include <iostream>

// #include <curand_kernel.h>

// #define RND (curand_uniform(&local_rand_state))

// #define MAX_DEPTH 5

// // limited version of checkCudaErrors from helper_cuda.h in CUDA examples
// #define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)

// void check_cuda(cudaError_t result, char
//     const *
//     const func,
//         const char *
//             const file, int
//     const line) {
//     if (result) {
//         std::cerr << "CUDA error = " << static_cast < unsigned int > (result) << " at " <<
//             file << ":" << line << " '" << func << "' \n";
//         // Make sure we call CUDA Device Reset before exiting
//         cudaDeviceReset();
//         exit(99);
//     }
// }

// __device__ color ray_color(const ray & r, hittable ** world, int depth, curandState *local_rand_state) {
//     hit_record rec;

//     // If we've exceeded the ray bounce limit, no more light is gathered.
//     if (depth <= 0) {
//         return color(0, 0, 0);
//     }

//     if ((*world)->hit(r, 0, infinity, rec)) {
//         ray scattered;
//         color attenuation;
//         if (rec.mat_ptr -> scatter(r, rec, attenuation, scattered, local_rand_state)) {
//             return attenuation * ray_color(scattered, world, depth - 1, local_rand_state);
//         }
//         return color(0, 0, 0);
//     }

//     vec3 unit_direction = unit_vector(r.direction());
//     float t = 0.5f * (unit_direction.y() + 1.0f);
//     return (1.0f - t) * color(1.0, 1.0, 1.0) + t * color(0.5, 0.7, 1.0);
// }

// __global__ void rand_init(curandState * rand_state) {
//     if (threadIdx.x == 0 && blockIdx.x == 0) {
//         curand_init(1984, 0, 0, rand_state);
//     }
// }

// __global__ void render_init(int max_x, int max_y, curandState * rand_state) {
//     int i = threadIdx.x + blockIdx.x * blockDim.x;
//     int j = threadIdx.y + blockIdx.y * blockDim.y;
//     if ((i >= max_x) || (j >= max_y)) return;
//     int pixel_index = j * max_x + i;
//     curand_init(1984 + pixel_index, 0, 0, & rand_state[pixel_index]);
// }

// __global__ void render(vec3 * fb, int max_x, int max_y, int ns, camera ** cam, hittable ** world, curandState * rand_state) {
//     int i = threadIdx.x + blockIdx.x * blockDim.x;
//     int j = threadIdx.y + blockIdx.y * blockDim.y;
//     if ((i >= max_x) || (j >= max_y)) return;
//     int pixel_index = j * max_x + i;
//     curandState local_rand_state = rand_state[pixel_index];
//     vec3 col(0, 0, 0);
//     for (int s = 0; s < ns; s++) {
//         float u = float(i + curand_uniform( & local_rand_state)) / float(max_x);
//         float v = float(j + curand_uniform( & local_rand_state)) / float(max_y);
//         ray r = ( * cam) -> get_ray(u, v, & local_rand_state);
//         col += ray_color(r, world, MAX_DEPTH, & local_rand_state);
//     }
//     rand_state[pixel_index] = local_rand_state;
//     col /= float(ns);
//     col[0] = sqrt(col[0]);
//     col[1] = sqrt(col[1]);
//     col[2] = sqrt(col[2]);
//     fb[pixel_index] = col;
// }

// __global__ void create_world(hittable ** d_list, hittable ** d_world, curandState * rand_state) {
//     if (threadIdx.x == 0 && blockIdx.x == 0) {
//         curandState local_rand_state = * rand_state;
//         d_list[0] = new sphere(vec3(0, -1000.0, -1), 1000,
//             new lambertian(vec3(0.5, 0.5, 0.5)));
//         int i = 1;
//         for (int a = -11; a < 11; a++) {
//             for (int b = -11; b < 11; b++) {
//                 float choose_mat = RND;
//                 vec3 center(a + RND, 0.2, b + RND);
//                 if (choose_mat < 0.8f) {
//                     d_list[i++] = new sphere(center, 0.2, new lambertian(vec3(RND * RND, RND * RND, RND * RND)));
//                 } else if (choose_mat < 0.95f) {
//                     d_list[i++] = new sphere(center, 0.2, new metal(vec3(0.5f * (1.0f + RND), 0.5f * (1.0f + RND), 0.5f * (1.0f + RND)), 0.5f * RND));
//                 } else {
//                     d_list[i++] = new sphere(center, 0.2, new dielectric(1.5));
//                 }
//             }
//         }
//         d_list[i++] = new sphere(vec3(0, 1, 0), 1.0, new dielectric(1.5));
//         d_list[i++] = new sphere(vec3(-4, 1, 0), 1.0, new lambertian(vec3(0.4, 0.2, 0.1)));
//         d_list[i++] = new sphere(vec3(4, 1, 0), 1.0, new metal(vec3(0.7, 0.6, 0.5), 0.0));
//         * rand_state = local_rand_state;
//         * d_world = new hittable_list(d_list, 22 * 22 + 1 + 3);
//     }
// }

// __global__ void create_camera(camera **d_camera, int image_width, int image_height) {
//     if (threadIdx.x == 0 && blockIdx.x == 0) {
//         vec3 lookfrom(13,2,3);
//         vec3 lookat(0,0,0);
//         float dist_to_focus = 10.0; (lookfrom-lookat).length();
//         float aperture = 0.1;
//         *d_camera   = new camera(lookfrom,
//                                  lookat,
//                                  vec3(0,1,0),
//                                  30.0,
//                                  float(image_width)/float(image_height),
//                                  aperture,
//                                  dist_to_focus);
//     }
// }

// __global__ void free_world(hittable ** d_list, hittable ** d_world) {
//     for (int i = 0; i < 22 * 22 + 1 + 3; i++) {
//         delete((sphere * ) d_list[i]) -> mat_ptr;
//         delete d_list[i];
//     }
//     delete * d_world;
// }

// int main() {

//     // Image

//     const auto aspect_ratio = 3.0 / 2.0;
//     const int image_width = 1200;
//     const int image_height = static_cast < int > (image_width / aspect_ratio);
//     const int samples_per_pixel = 500;
//     // gpu blocks
//     const int tx = 8;
//     const int ty = 8;

//     clock_t start, stop;

//     // allocate framebuffer
//     int num_pixels = image_width * image_height;
//     size_t fb_size = num_pixels * sizeof(vec3);
//     vec3 * fb;
//     checkCudaErrors(cudaMallocManaged((void ** ) & fb, fb_size));

//     // allocate random state
//     curandState * d_rand_state;
//     checkCudaErrors(cudaMalloc((void ** ) & d_rand_state, num_pixels * sizeof(curandState)));
//     curandState * d_rand_state2;
//     checkCudaErrors(cudaMalloc((void ** ) & d_rand_state2, 1 * sizeof(curandState)));

//     // World

//     // we need that 2nd random state to be initialized for the world creation
//     rand_init <<< 1, 1 >>> (d_rand_state2);
//     checkCudaErrors(cudaGetLastError());
//     checkCudaErrors(cudaDeviceSynchronize());

//     // make our world of hittables
//     hittable ** d_list;
//     int num_hittables = 22 * 22 + 1 + 3;
//     checkCudaErrors(cudaMalloc((void ** ) & d_list, num_hittables * sizeof(hittable * )));
//     hittable ** d_world;

//     create_world<<< 1, 1 >>>(d_list, d_world, d_rand_state2);
//     checkCudaErrors(cudaGetLastError());
//     checkCudaErrors(cudaDeviceSynchronize());

//     // Camera

//     camera ** d_camera;
//     checkCudaErrors(cudaMalloc((void ** ) & d_camera, sizeof(camera * )));

//     point3 lookfrom(13, 2, 3);
//     point3 lookat(0, 0, 0);
//     vec3 vup(0, 1, 0);
//     auto dist_to_focus = 10.0;
//     auto aperture = 0.1;

//     create_camera <<< 1, 1 >>> (d_camera, image_width, image_height);

//     checkCudaErrors(cudaGetLastError());
//     checkCudaErrors(cudaDeviceSynchronize());

//     // Render

//     start = clock();

//     dim3 blocks(image_width / tx + 1, image_height / ty + 1);
//     dim3 threads(tx, ty);

//     render_init << < blocks, threads >>> (image_width, image_height, d_rand_state);
//     checkCudaErrors(cudaGetLastError());
//     checkCudaErrors(cudaDeviceSynchronize());

//     render << < blocks, threads >>> (fb, image_width, image_height, samples_per_pixel, d_camera, d_world, d_rand_state);
//     checkCudaErrors(cudaGetLastError());
//     checkCudaErrors(cudaDeviceSynchronize());

//     stop = clock();
//     double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
//     std::cerr << "took " << timer_seconds << " seconds.\n";

//     std::cout << "P3\n" << image_width << ' ' << image_height << "\n255\n";
//     for (int j = image_height - 1; j >= 0; j--) {
//         std::cerr << "\rScanlines remaining: " << j << ' ' << std::flush;
//         for (int i = 0; i < image_width; i++) {
//             size_t pixel_index = j * image_width + i;
//             write_color(std::cout, fb[pixel_index]);
//         }
//     }

//     // clean up
//     checkCudaErrors(cudaDeviceSynchronize());
//     free_world << < 1, 1 >>> (d_list, d_world);
//     delete * d_camera;
//     checkCudaErrors(cudaGetLastError());
//     checkCudaErrors(cudaFree(d_camera));
//     checkCudaErrors(cudaFree(d_world));
//     checkCudaErrors(cudaFree(d_list));
//     checkCudaErrors(cudaFree(d_rand_state));
//     checkCudaErrors(cudaFree(d_rand_state2));
//     checkCudaErrors(cudaFree(fb));

//     cudaDeviceReset();

//     std::cerr << "\nDone.\n";
// }