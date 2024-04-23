import cv2
import numpy as np
import gzip
import os
import io
import copy
import sys
import math

BITS_IN_BYTE = 8

"""
returns the image size in bits
"""
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
        return expgolomb_unsigned(2*n-1)
    else:
        return expgolomb_unsigned(-2*n)

def dec2bin(dec_n, min_digits):
    # [2:] is to convert to binary and remove the '0b' prefix
    return bin(int(dec_n))[2:].zfill(min_digits)


def expgolomb_unsigned(n):
    if n == 0:
        return 1
    else:
        trail_bits = dec2bin(n+1, math.floor(math.log2(n+1)))
        headbits = dec2bin(0, len(trail_bits)-1)
        return headbits + trail_bits

if __name__ == "__main__":
    img_path = "zanzara_lowres.jpg"
    img = cv2.imread(img_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rows, cols = img_gray.shape

    luminance = 0.2126 * img[:, :, 0] + 0.7152 * img[:, :, 1] + 0.0722 * img[:, :, 2]

    # hist contains the counts for each bin in the histogram
    # bins contains the edges of the bins
    hist, bins = np.histogram(luminance.flatten(), 255)

    prob = hist / np.sum(hist)
    entropy = -np.sum(prob * np.log2(prob))

    print("The entropy of the image is: " + str(entropy) + " bits")

    compressed_img_size = compressed_image_size_bits(img_gray)
    rows, cols = img_gray.shape
    num_pixels = rows * cols
    print(
        "The bitrate of the image is: "
        + str(compressed_img_size / num_pixels)
        + "bits per pixel"
    )

    print("The theoretical compression rate is: " + str(8 / entropy) + "\n\n")

    print("the image has " + str(num_pixels) + "pixels")
    # codifica predittiva semplice

    linear_image = img_gray.reshape(-1).astype(np.float64)
    encoded_image = copy.deepcopy(linear_image)
    encoded_image[0] = linear_image[0] - 128

    for i in range(1, num_pixels):
        encoded_image[i] = linear_image[i] - linear_image[i - 1]

    cv2.imshow("img", img_gray)
    cv2.imshow("enc", encoded_image.reshape(rows, cols))

    decoded_image = np.zeros(num_pixels).astype(np.float64)
    decoded_image[0] = encoded_image[0] + 128
    buf = decoded_image[0]
    for i in range(1, num_pixels):
        buf += encoded_image[i]
        decoded_image[i] = buf

    decoded_image = decoded_image.reshape(rows, cols).astype(np.uint8)
    cv2.imshow("dec", decoded_image)
    cv2.waitKey(0)

    # Valutare il numero di bit necessari per codificare l’errore di predizione 
    # y con la codifica Exp Golomb con segno
    # y e' l' array con l' immagine codificata al suo interno (linear_image)

    bit_count = 0
    for symbol in linear_image:
        # print("symbol " + str(symbol) + " type is " + str(type(symbol)))
        codeword = expgolomb_signed(symbol)
        bit_count += len(str(codeword))

    EG_bpp = bit_count/(rows*cols)
    print("bit per pixel con codifica exp_golomb: " + str(EG_bpp))

    
    # Valutare il numero di bit necessari per codificare l’errore di predizione y
    # con la codifica Exp Golomb con segno, dedurne il bitrate di codifica e 
    # confrontare tale valore con quello ottenuto ai punti 1 e 3
