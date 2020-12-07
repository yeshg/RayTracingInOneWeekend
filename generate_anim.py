import numpy as np
import os
import subprocess  # just to call an arbitrary command e.g. 'ls'

lookfrom_poses = []

# create spiral centered at 0,0,0 going up in z and slightly out in x y
R = 4
a = 2 / np.pi
t = 0.5+2*np.pi*np.sin(np.linspace(0, 2*np.pi, 100))

lookfrom_poses = np.vstack([
    R*np.cos(t)*t/(np.pi),
    a*t + 2.0,
    R*np.sin(t)*t/(np.pi)
])

print(lookfrom_poses.shape)

# import matplotlib.pyplot as plt
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot(lookfrom_poses[0], lookfrom_poses[1], lookfrom_poses[2])
# plt.show()
# exit()

class cd:
    """Context manager for changing the current working directory"""
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)

# enter the directory like this:
with cd("gpu_accelerated"):
    # we are in ~/Library
    # subprocess.call("ls -l", shell=True)
    subprocess.call("rm -rf ppms/*", shell=True)
    subprocess.call("rm -rf images/*", shell=True)
    subprocess.call("make clean", shell=True)
    subprocess.call("make", shell=True)
    for i in range(lookfrom_poses.shape[1]):
        pos = lookfrom_poses[:, i]
        subprocess.call(f"./cudart {pos[0]} {pos[1]} {pos[2]} > ppms/out_{i}.ppm", shell=True)
        subprocess.call(f"ppmtojpeg ppms/out_{i}.ppm > images/out_{i}.jpg", shell=True)
