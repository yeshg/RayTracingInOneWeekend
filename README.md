# Ray Tracing in One Weekend + GPU acceleration

![Animated Scene](animated.gif)

Results from following the famous book by Peter Shirley 'Ray Tracing in One Weekend': https://raytracing.github.io/books/RayTracingInOneWeekend.html#overview

Ray Tracing is slow, so I tried following another tutorial to use CUDA to make it faster: https://github.com/rogerallen/raytracinginoneweekendincuda

My solution for this second part did not work, so the original author's official code was used, mine is commented out below it.

`pure_cpp/` - contains my solution after following Shirley's book. A lot of the code the same.

`gpu_accelerated/` - contains parallelised version of the ray tracer with cuda. working code is directly copied from https://github.com/rogerallen/raytracinginoneweekendincuda

`generate_anim.py` calls the gpu_accelerated ray tracer for several different camera lookfrom points and puts the output into an images folder.
`create_gif.py` converts the folder of images into a gif like the one above.
