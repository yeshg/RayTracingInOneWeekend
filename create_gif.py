import imageio
with imageio.get_writer('animated.gif', mode='I') as writer:
    for i in range(50):
        filename = f"gpu_accelerated/images/out_{i}.jpg"
        image = imageio.imread(filename)
        writer.append_data(image)
