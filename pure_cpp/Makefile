CXX = g++

rt: main.cpp color.h ray.h sphere.h vec3.h hittable.h hittable_list.h camera.h material.h
	$(CXX) -O3 -o rt main.cpp

out.ppm: rt
	rm -f out.ppm
	time ./rt > out.ppm

out.jpg: out.ppm
	rm -f out.jpg
	ppmtojpeg out.ppm > out.jpg

clean:
	rm -f rt rt.o out.ppm out.jpg