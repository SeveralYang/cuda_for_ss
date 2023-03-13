from test_ss_cpu import main as cpu
from test_ss_gpu import main as gpu

if __name__ == '__main__':
    path = "1.png"
    cpu(path)
    gpu(path)

"""
核心代码
    R0 = numpy.zeros((300, 4))
    R0[:, 0] = img.shape[1]
    R0[:, 1] = img.shape[0]
    device_img = cuda.to_device(img)
    device_R = cuda.to_device(R0)
    get_R[(100, 100, 1), (32, 32, 1)](device_img, device_R)
    R0 = device_R.copy_to_host()
    tt1 = time()

@numba.cuda.jit
def get_R(img, res):
    i, j = cuda.grid(2)
    # i == cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    # j == cuda.threadIdx.y + cuda.blockDim.y * cuda.blockIdx.y:
    if i < img.shape[0] and j < img.shape[1]:
        r, g, b, l = img[i][j]
        l = int(l)
        res[l][0] = min(res[l][0], j)
        res[l][1] = min(res[l][1], i)
        res[l][2] = max(res[l][2], j)
        res[l][3] = max(res[l][3], i)
        
============源代码================
    for y, i in enumerate(img):
        for x, (r, g, b, l) in enumerate(i):
            # initialize a new region
            if l not in R:
                R[l] = {
                    "min_x": 0xffff, "min_y": 0xffff,
                    "max_x": 0, "max_y": 0, "labels": [l]}
            # bounding box
            if R[l]["min_x"] > x:
                R[l]["min_x"] = x
            if R[l]["min_y"] > y:
                R[l]["min_y"] = y
            if R[l]["max_x"] < x:
                R[l]["max_x"] = x
            if R[l]["max_y"] < y:
                R[l]["max_y"] = y
"""