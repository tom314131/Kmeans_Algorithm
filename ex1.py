from copy import copy
import numpy as np
from numpy.core._multiarray_umath import floor
import load
import init_centroids
import numpy.linalg
from scipy.misc import imread
import scipy.misc
import matplotlib.pyplot as plt


def show_and_save_image(dictionary, centroids, k):
    """
    Saving the image after the last iteration in a result folder
    :param dictionary: a dictionary between centroids and the pixels
    :param centroids: the array of the centroids after the last iteration
    :param k: the number of the centroids
    :return: None
    """
    pixels = load.X.copy()
    new_dict = {}
    for key in range(0, len(centroids)):
        for value in dictionary[key]:
            new_dict[str(value)] = key
    for i in range(0, len(pixels)):
        pixels[i] = centroids[new_dict[str(pixels[i])]]
    nd_array_pixels = np.array([pixels])
    reshape_img = nd_array_pixels.reshape(load.img_size[0], load.img_size[1], load.img_size[2])
    image_rgb = reshape_img * 255
    image_rgb = image_rgb.astype(int)

    plt.imshow(image_rgb)
    plt.grid(False)
    # plt.show()

    # saving the results of the output
    scipy.misc.imsave("results\\"+str(k) + "k_means_" + load.path, image_rgb)


def average(centroid_pixels):
    """
    :param centroid_pixels: the array pixel that belong to the centroid
    :return: a new centroid that is the average of all the pixels in the array
    """
    pix = centroid_pixels[0].copy()
    cen_length = len(centroid_pixels)
    for idx in range(1, cen_length):
        pix += centroid_pixels[idx]
    return pix / len(centroid_pixels)


def distance_vectors(vec1, vec2):
    """
    :param vec1: the first vector
    :param vec2: the second vector
    :return: distance between to vectors
    """
    return numpy.linalg.norm(vec1 - vec2)


def initial_dictionary(k):
    """
    :param k: the number of the centroids
    :return: initialize dictionary
    """
    my_dict = {}
    for idx in range(k):
        my_dict[idx] = []
    return my_dict


def print_array(array, k):
    """
    Printing the array of pixels in the iteration

    :param array: the array of pixel in the iteration
    :param k: the number of the centroids
    :return: None
    """
    pixels_int = floor(array * 100) / 100
    for idx in range(0, k):
        print("[", end='')
        print(", ".join(map(str, pixels_int[idx])), end='')
        print("]", end="")
        if idx != k - 1:
            print(", ", end="")
    print()


def k_means(k):
    """
    Executing the kmeans algorithm
    :param k: the number of the centroids
    :return: None
    """
    centroids = init_centroids.init_centroids(0, k)
    print("k=" + str(k) + ":")
    for iteration in range(0, 11):
        pixels = load.X.copy()
        my_dict = initial_dictionary(k)
        print("iter " + str(iteration) + ": ", end='')
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
    show_and_save_image(my_dict, centroids,k)


if __name__ == '__main__':
    for i in [2, 4, 8, 16]:
        k_means(i)
