import cv2
import numpy as np
import gzip
import os
import io
import copy
import sys
import math
import statistics

BITS_IN_BYTE = 8


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

    return compressed_size * BITS_IN_BYTE


def expgolomb_signed(n):
    if n > 0:
        return expgolomb_unsigned(2 * n - 1)
    else:
        return expgolomb_unsigned(-2 * n)


def dec2bin(dec_n, min_digits):
    # [2:] is to convert to binary and remove the '0b' prefix
    return bin(int(dec_n))[2:].zfill(min_digits)


def expgolomb_unsigned(n):
    if n == 0:
        return 1
    else:
        trail_bits = dec2bin(n + 1, math.floor(math.log2(n + 1)))
        headbits = dec2bin(0, len(trail_bits) - 1)
        return headbits + trail_bits


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
    # codifica predittiva semplice
    # trasformo l'immagine in un array
    linear_image = img.reshape(-1).astype(np.float64)
    encoded_image = copy.deepcopy(linear_image)
    encoded_image[0] = linear_image[0] - 128

    for i in range(1, num_pixels):
        encoded_image[i] = linear_image[i] - linear_image[i - 1]

    # cv2.imshow("img", img_gray)
    # cv2.imshow("enc", encoded_image.reshape(rows, cols))
    # print("encoded image")
    # print(np.array(encoded_image.reshape(rows, cols)))

    decoded_image = np.zeros(num_pixels).astype(np.float64)
    decoded_image[0] = encoded_image[0] + 128
    buf = decoded_image[0]
    for i in range(1, num_pixels):
        buf += encoded_image[i]
        decoded_image[i] = buf

    decoded_image = decoded_image.reshape(rows, cols).astype(np.uint8)
    # cv2.imshow("dec", decoded_image)
    # cv2.waitKey(0)

    # Valutare il numero di bit necessari per codificare l’errore di predizione
    # y con la codifica Exp Golomb con segno
    # y e' l' array con l' immagine codificata al suo interno (linear_image)

    encoded_img_entropy = entropy(np.array(encoded_image.reshape(rows, cols)))
    print("The entropy of the encoded image is: " + str(encoded_img_entropy))
    print("The obtainable compression rate is: " + str(8 / encoded_img_entropy))

    bit_count = 0
    for symbol in encoded_image:
        # print("symbol " + str(symbol) + " type is " + str(type(symbol)))
        codeword = expgolomb_signed(symbol)
        bit_count += len(str(codeword))

    EG_bpp = bit_count / num_pixels
    print("bit per pixel con codifica exp_golomb: " + str(EG_bpp))


def advanced_predictive_encoding(img):
    # linear_image = img.reshape(-1).astype(np.float64)
    # TODO: UNCOMMENT UNDER HERE
    # encoded_image = copy.deepcopy(img)
    encoded_image = np.random.randint(0, 100, size=(9, 6))

    encoded_image[0, 0] = encoded_image[0, 0] - 128
    # print(encoded_image)
    rows, cols = img.shape
    print("image shape: " + str(img.shape))
    # for each pixel in position n, m
    # first row -> predictor left from current pixel
    # first column -> predictor on top of current pixel
    # last column -> predictor is the median value between x(n-1, m), x(n,m-1) and x(n-1, m-1)
    # else -> predictor is the median value between x(n-1, m) x(n, m-1), x(n-1, m+1)
    # once the predictor is built you can calculate the prediction error y = x-p (it's an image)

    print(encoded_image)

    for i in range(1, cols):
        encoded_image[0, i] = encoded_image[0, i - 1]
    for i in range(1, rows):
        encoded_image[i, 0] = encoded_image[i - 1, 0]
    n = 1
    while n < rows:
        m = 1
        while m < cols:
            # in last column TODO: if it doesn't work move by -1
            if m == (cols - 1):
                v1 = encoded_image[n - 1, m]
                v2 = encoded_image[n, m - 1]
                v3 = encoded_image[n - 1, m - 1]
                # print("calculating the median between: " + str(np.array((v1,v2,v3))))
                encoded_image[n, m] = statistics.median((v1, v2, v3))
                continue
            v1 = encoded_image[n - 1, m]
            v2 = encoded_image[n, m - 1]
            # print("n = " + str(n) + "\tm  = " + str(m))
            v3 = encoded_image[n - 1, m + 1]
            encoded_image[n, m] = statistics.median((v1, v2, v3))
            m += 1
        n += 1
    print(encoded_image)
    return


if __name__ == "__main__":
    img_path = "einst.pgm"
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    img_entropy = entropy(img)
    print("Entropy of the image: " + str(img_entropy))
    print("The theoretical compression rate is: " + str(8 / img_entropy))

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
    # simple_predictive_encoding(img)

    # compressione con codifica predittiva avanzata
    advanced_predictive_encoding(img)
