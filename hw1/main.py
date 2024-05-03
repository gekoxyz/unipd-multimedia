import cv2
import numpy as np
import gzip
import os
import io
import copy
import sys
import math
import statistics

BITS_PER_PIXEL = 8


def compressed_image_size_bits(image):
    image_filename = "compressed_image.gz"
    # Create an in-memory binary stream
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb") as f:
        f.write(image)
    with open(image_filename, "wb") as f:
        f.write(buffer.getvalue())
    compressed_size = os.path.getsize(image_filename)

    os.remove(image_filename)

    return compressed_size * BITS_PER_PIXEL

def expgolomb_signed(n):
    if n > 0:
        return expgolomb_unsigned(2 * n - 1)
    else:
        return expgolomb_unsigned(-2 * n)

def expgolomb_unsigned(n):
    if n == 0:
        return 1
    else:
        return (2 * math.ceil(math.log2(n + 2))) - 1

def entropy(image):
    luminance = 0.2126 * image[:, :] + 0.7152 * image[:, :] + 0.0722 * image[:, :]
    # hist contains the counts for each bin in the histogram
    # bins contains the edges of the bins
    hist, bins = np.histogram(luminance.flatten(), 255)
    # Calculate the probability of each bin
    prob = hist / hist.sum()
    # Calculate the entropy
    return -np.sum(prob[prob != 0] * np.log2(prob[prob != 0]))


def simple_predictive_encoding(img):
    # trasformo l'immagine in un array
    linear_image = img.reshape(-1).astype(np.float64)
    rows, cols = img.shape
    prediction = np.zeros(rows * cols)
    prediction[0] = linear_image[0] - 128

    for i in range(1, num_pixels):
        prediction[i] = linear_image[i] - linear_image[i - 1]

    # decoded_image = np.zeros(num_pixels).astype(np.float64)
    # decoded_image[0] = prediction[0] + 128
    # buf = decoded_image[0]
    # for i in range(1, num_pixels):
    #     buf += prediction[i]
    #     decoded_image[i] = buf
    # decoded_image = decoded_image.reshape(rows, cols).astype(np.uint8)
    prediction = np.array(prediction.reshape(rows, cols))
    encoded_img = img - prediction

    encoded_img_entropy = entropy(encoded_img)
    print("The entropy of the encoded image is: " + str(encoded_img_entropy))
    print("The obtainable compression rate is: " + str(8 / encoded_img_entropy))

    encoded_prediction_linear = encoded_img.reshape(-1).astype(np.float64)
    bit_count = 0
    for symbol in encoded_prediction_linear:
        bit_count += expgolomb_signed(symbol)

    EG_bpp = bit_count / num_pixels
    print("bit per pixel con codifica exp_golomb: " + str(EG_bpp))


def advanced_predictive_encoding(img):
    rows, cols = img.shape
    print("image shape: " + str(img.shape))
    print(f"bit count: {rows * cols}")

    prediction = np.zeros((rows, cols))
    prediction[0, 0] = img[0, 0] - 128

    # for each pixel in position n, m
    # first row -> predictor left from current pixel
    # first column -> predictor on top of current pixel
    # last column -> predictor is the median value between x(n-1, m), x(n,m-1) and x(n-1, m-1)
    # else -> predictor is the median value between x(n-1, m) x(n, m-1), x(n-1, m+1)
    # once the predictor is built you can calculate the prediction error y = x-p (it's an image)

    n = 0
    while n < rows:
        m = 0
        while m < cols:
            # first row
            if n == 0:
                prediction[n, m] = img[n, m - 1]
            # first column
            elif m == 0:
                prediction[n, m] = img[n - 1, m]
            # last column
            elif m == (cols - 1):
                v1 = img[n - 1, m]
                v2 = img[n, m - 1]
                v3 = img[n - 1, m - 1]
                # print("calculating the median between: " + str(np.array((v1,v2,v3))))
                prediction[n, m] = statistics.median((v1, v2, v3))
            else:
                v1 = img[n - 1, m]
                v2 = img[n, m - 1]
                v3 = img[n - 1, m + 1]
                prediction[n, m] = statistics.median((v1, v2, v3))
            m += 1

        n += 1

    encoded_img = img - prediction
    encoded_img_entropy = entropy(encoded_img)
    print("The entropy of the encoded image is: " + str(encoded_img_entropy))
    print(
        "The obtainable compression rate is: "
        + str(BITS_PER_PIXEL / encoded_img_entropy)
    )

    encoded_prediction_linear = encoded_img.reshape(-1).astype(np.float64)
    bit_count = 0
    for symbol in encoded_prediction_linear:
        bit_count += expgolomb_signed(symbol)

    print(f"bit count: {bit_count}")
    EG_bpp = bit_count / num_pixels
    print("bit per pixel con codifica exp_golomb: " + str(EG_bpp))

    return


if __name__ == "__main__":
    images = ["einst.pgm", "house.pgm", "lake.pgm"]
    for img_path in images:
        print(f"===== {img_path} =====")
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        img_entropy = entropy(img)
        print("Entropy of the image: " + str(img_entropy))
        print(
            "The theoretical compression rate is: " + str(BITS_PER_PIXEL / img_entropy)
        )

        rows, cols = img.shape
        num_pixels = rows * cols

        print("The image has " + str(num_pixels) + " pixels")

        # compressione con gzip
        # compressed_img_size = compressed_image_size_bits(img)
        # print(
        #     "The bitrate of the image compressed with gzip is: "
        #     + str(compressed_img_size / num_pixels)
        #     + "bits per pixel"
        # )

        # compressione con codifica predittiva semplice
        simple_predictive_encoding(img)

        # compressione con codifica predittiva avanzata
        advanced_predictive_encoding(img)




# def dec2bin(dec_n, min_digits):
#     # [2:] is to convert to binary and remove the '0b' prefix
#     return bin(int(dec_n))[2:].zfill(min_digits)


# def expgolomb_signed(n):
#     if n > 0:
#         return expgolomb_unsigned(2 * n - 1)
#     else:
#         return expgolomb_unsigned(-2 * n)


# def expgolomb_unsigned(n):
#     if n == 0:
#         return 1
#     else:
#         trail_bits = dec2bin(n + 1, math.ceil(math.log2(n + 1)))
#         headbits = dec2bin(0, len(trail_bits) - 1)
#         return int(headbits + trail_bits)

