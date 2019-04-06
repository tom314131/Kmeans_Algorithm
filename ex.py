from copy import copy
import numpy as np
from numpy.core._multiarray_umath import floor
import load
import init_centroids
import numpy.linalg
from scipy.misc import imread
import scipy.misc
import matplotlib.pyplot as plt
import imageio


def show_and_save_image(dictionary, centroids, k):
    pixels = load.X.copy()
    newdict = {}
    for key in range(0, len(centroids)):
        for value in dictionary[key]:
            newdict[str(value)] = key
    for i in range(0, len(pixels)):
        pixels[i] = centroids[newdict[str(pixels[i])]]
    nd_array_pixels = np.array([pixels])
    reshape_img = nd_array_pixels.reshape(load.img_size[0], load.img_size[1], load.img_size[2])
    image_rgb = reshape_img * 255
    image_rgb = image_rgb.astype(int)

    plt.imshow(image_rgb)
    plt.grid(False)
    # plt.show()
    scipy.misc.imsave(str(k) + "k_means_" + load.path, image_rgb)


def average(centroid):
    pix = centroid[0].copy()
    for i in range(1, len(centroid)):
        pix += centroid[i]
    x = pix / len(centroid)
    return x


def distance_vectors(vec1, vec2):
    return numpy.linalg.norm(vec1 - vec2)


def initial_dictionary(k):
    my_dict = {}
    for i in range(k):
        my_dict[i] = []
    return my_dict


def print_array(array, k):
    pixels_int = floor(array * 100) / 100
    for i in range(0, k):
        print("[", end='')
        print(", ".join(map(str, pixels_int[i])), end='')
        print("]", end="")
        if i != k - 1:
            print(", ", end="")
    print()


def k_means(k):
    centroids = init_centroids.init_centroids(0, k)
    print("k=" + str(k) + ":")
    for iter in range(0, 11):
        pixels = load.X.copy()
        my_dict = initial_dictionary(k)
        print("iter " + str(iter) + ": ", end='')
        print_array(centroids, k)
        pixels_size = len(pixels)
        for pixel_idx in range(0, pixels_size):
            cen_idx = 0
            min_dis = distance_vectors(pixels[pixel_idx], centroids[0])
            for cen in range(1, k):
                dis = distance_vectors(pixels[pixel_idx], centroids[cen])
                if dis < min_dis:
                    cen_idx = cen
                    min_dis = dis
            my_dict[cen_idx].append(pixels[pixel_idx])
        for i in range(0, k):
            centroids[i] = average(my_dict[i])
    show_and_save_image(my_dict, centroids, k)


if __name__ == '__main__':
    for i in [2, 4, 8, 16]:
        k_means(i)
