import numpy as np
import statistics

encoded_image = np.random.randint(0, 100, size=(9, 6))

encoded_image[0, 0] = encoded_image[0, 0] - 128
# print(encoded_image)
rows, cols = encoded_image.shape
print("image shape: " + str(encoded_image.shape))
# for each pixel in position n, m
# first row -> predictor left from current pixel
# first column -> predictor on top of current pixel
# last column -> predictor is the median value between x(n-1, m), x(n,m-1) and x(n-1, m-1)
# else -> predictor is the median value between x(n-1, m) x(n, m-1), x(n-1, m+1)
# once the predictor is built you can calculate the prediction error y = x-p (it's an image)

print(encoded_image)

n = 0
while n < rows:
    m = 0
    while m < cols:
        # first row
        if n == 0:
            encoded_image[n, m] = encoded_image[n, m - 1]
        # first column
        if m == 0:
            encoded_image[n, m] = encoded_image[n - 1, 0]
        # last column
        if m == (cols - 1):
            v1 = encoded_image[n - 1, m]
            v2 = encoded_image[n, m - 1]
            v3 = encoded_image[n - 1, m - 1]
            # print("calculating the median between: " + str(np.array((v1,v2,v3))))
            encoded_image[n, m] = statistics.median((v1, v2, v3))
            m += 1
            continue
        v1 = encoded_image[n - 1, m]
        v2 = encoded_image[n, m - 1]
        print("n = " + str(n) + "\tm  = " + str(m))
        v3 = encoded_image[n - 1, m + 1]
        encoded_image[n, m] = statistics.median((v1, v2, v3))
        m += 1

    n += 1
print(encoded_image)