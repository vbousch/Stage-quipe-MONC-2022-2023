# This script aims at reading images of spheroids for 3 doses of propranolol, normalizing their grayscale,
# creating a mask by image segmentation and process this mask by printing meaningful values (mean, min, max, std...)
# and showing their histograms and borders

# we are using here pip 23.0 version

# importing useful packages
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import matplotlib.patches as patches
import matplotlib.image as mpimg
from scipy import misc
from scipy.stats import lognorm, gamma, exponweib
import cv2 as cv
import pandas as pd
import time
import os
import imageio
import itk
import sys

start = time.time()
print("ITK version :", itk.Version.GetITKVersion())
print("OpenCV version :", cv.__version__)
dim = 2               # dimension of the image
seg_threshold = 0.35  # threshold value for the image segmentation
Lx = 4.34
Ly = 3.25
spheroid_gif = []

# B1 case for the first and final time
# input_filename = 'images/VID811_B1_1_2021y11m15d_14h15m.PNG'
input_filename1 = 'images/VID821_G8_1_2021y11m23d_14h00m.PNG'
# input_filename = 'images/carre.PNG'
output_filename = 'images/output.PNG'
# for coronas
corona_scale = 0.2
N_coro = 30


# allows to rescale an extracted contour
def scale_contour(cnt, scale):
    M = cv.moments(cnt)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])

    cnt_norm = cnt - [cx, cy]
    cnt_scaled = cnt_norm * scale
    cnt_scaled = cnt_scaled + [cx, cy]
    cnt_scaled = cnt_scaled.astype(np.int32)

    return cnt_scaled


def DataExtraction(input_filename):

    print(input_filename)
    PixelType = itk.UC
    ImageType = itk.Image[PixelType, dim]
    image = itk.imread(input_filename, PixelType)  # reading the image in grayscale level
    size = image.GetLargestPossibleRegion().GetSize()  # accessing the image size
    dx = Lx/size[0]
    dy = Ly/size[1]
    origin = image.GetLargestPossibleRegion().GetIndex()  # accessing the origin of the image index
    # print("size : ", size)
    # print("origin : ", origin)


    '''cropping the image : the origin is the top left corner and size is the size of the cropped image'''
    origin[0] = 550
    origin[1] = 300
    size[0] = 480
    size[1] = 490

    # print("new size : ", size)
    # print("new origin : ", origin)

    extract_region = itk.ImageRegion[dim]()
    extract_region.SetIndex(origin)
    extract_region.SetSize(size)

    ExtractFilterType = itk.ExtractImageFilter[itk.Image[PixelType, dim], itk.Image[PixelType, dim]]
    extract_filter = ExtractFilterType.New()
    extract_filter.SetExtractionRegion(extract_region)
    extract_filter.SetInput(image)
    extract_filter.Update()
    cropped_image = extract_filter.GetOutput()
    # np_view1 = itk.array_view_from_image(cropped_image)
    # plt.imshow(np_view1)
    # plt.show()


    # creating the image output in the right format
    NewPixelType = itk.F
    NewImageType = itk.Image[NewPixelType, dim]
    output_image = NewImageType.New()
    output_image.SetRegions(cropped_image.GetLargestPossibleRegion())
    output_image.Allocate()

    # normalizing each pixel in the image
    min_val = 0
    max_val = 255
    for i in range(size[0]):
        for j in range(size[1]):
            index = [origin[0]+i, origin[1]+j]
            pixel = cropped_image.GetPixel(index)
            normalized_pixel = (pixel - min_val) / (max_val - min_val)
            output_image.SetPixel(index, normalized_pixel)

    image_array = itk.array_view_from_image(output_image)

    # '''image segmentation using arbitrary thresholds values'''
    # seg_time1 = time.time()
    #
    # for i in range(size[0]):
    #     for j in range(size[1]):
    #         index = [origin[0]+i, origin[1]+j]
    #         pixel = output_image.GetPixel(index)
    #         if pixel >= seg_threshold:
    #             output_image.SetPixel(index, 0.)
    #
    # seg_time2 = time.time()


    '''creating a binary image to facilitate contours detection'''
    binary_image = NewImageType.New()
    binary_image.SetRegions(cropped_image.GetLargestPossibleRegion())
    binary_image.Allocate()
    for i in range(size[0]):
        for j in range(size[1]):
            index = [origin[0]+i, origin[1]+j]
            pixel = output_image.GetPixel(index)
            if pixel < seg_threshold:
                binary_image.SetPixel(index, 1.)
            elif pixel > seg_threshold:
                binary_image.SetPixel(index, 0.)
    binary_array = itk.array_view_from_image(binary_image)
    plt.imsave(output_filename, binary_array, cmap='gray')

    '''using the OpenCV library to extract the largest contour of the spheroid'''
    img = cv.imread(output_filename, cv.IMREAD_GRAYSCALE)

    # applying binary threshold to the image
    _, threshold = cv.threshold(img, 128, 255, cv.THRESH_BINARY)

    # finding all contours in the image
    contours, _ = cv.findContours(threshold, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # finding the largest contour of the spheroid
    largest_contour = None
    largest_area = 0
    for c in contours:
        area = cv.contourArea(c)
        if area > largest_area:
            largest_area = area
            largest_contour = c

    # drawing the largest contour detected (only for visual purposes) # TODO
    img = cv.imread(output_filename)
    image_with_contour = cv.drawContours(img, [largest_contour], -1, (0, 255, 0), 2)
    # writing/displaying the image and the largest contour detected
    cv.imwrite('images/contour.png', image_with_contour)
    contour_image = np.zeros(binary_array.shape, dtype=np.uint8)
    cv.drawContours(contour_image, [largest_contour], -1, 255, 1)
    cv.imwrite('images/largest_contour.png', contour_image)

    '''write and fill the contour to create a mask'''
    mask = np.zeros(binary_array.shape, dtype=np.uint8)
    cv.drawContours(mask, [largest_contour], -1, 255, -1)
    # plt.imshow(mask)
    # plt.show()
    cv.imwrite('images/mask.png', mask)


    # multiplying the cropped image by the mask
    masked_image = np.multiply(image_array, mask//255)
    # plt.imshow(masked_image)
    # plt.show()
    spheroid_gif.append((masked_image*255).astype('uint8'))
    cv.imwrite('images/masked_image.png', masked_image*255)

    # isolate spheroid
    spheroid = ma.masked_array(image_array, 1 - mask//255)
    cv.imwrite('images/spheroid.png', spheroid * 255)
    # plt.imshow(spheroid)
    # plt.show()

    # isolate spheroid and holes are filled with 0.
    # mx = ma.masked_array(np.where(image_array > 0.5, 0., image_array), 1 - mask//255)
    # mx = ma.masked_array(np.where(image_array > 0.3, np.ma.masked, image_array), 1 - mask//255)

    # isolate spheroid without holes
    spheroid_no_holes = ma.masked_array(np.ma.masked_where(image_array > 0.5, image_array), 1-mask//255)
    cv.imwrite('images/spheroid_no_holes.png', spheroid_no_holes * 255)
    # plt.imshow(spheroid_no_holes)
    # plt.show()

    # compute several values of interests : mean, std, max and min (if necessary)
    mean = spheroid.mean()
    # std = np.ma.std(spheroid)
    # maximum = np.ma.max(spheroid)
    # minimum = np.ma.min(spheroid)

    read_image_contour = itk.imread('images/largest_contour.png', PixelType)  # reading the contour image in grayscale level
    Z = itk.array_view_from_image(read_image_contour)/255
    size = 1
    xmin, xmax = 0, Z.shape[1]
    ymin, ymax = 0, Z.shape[0]

    # size of pixel
    px_size_x = 4.34/1536
    px_size_y = 3.25/1152

    # computation of radius, volumes and m for the spheroid
    area = px_size_x * px_size_y * np.ma.count(spheroid)
    area_T = area*(1 - mean)
    radius = np.sqrt(area/np.pi)
    radius_T = np.sqrt(area_T/np.pi)
    volume = (4/3)*np.pi*np.power(radius, 3)
    volume_T = (4/3)*np.pi*np.power(radius_T, 3)
    m_value = 1 - volume_T/volume
    I_px_sphero = 1 - (1 - mean)

    # coronas treatment here
    # using contour rescaling to define coronas
    radius_coronas_dr = []
    for i in range(100):
        temp_radius = 0.005 * (i + 1)
        if temp_radius - 0.005 < radius:
            radius_coronas_dr.append(temp_radius)
    radius_coronas = [elt - 0.005 for elt in radius_coronas_dr]
    scaling_coeffs = [elt/radius for elt in radius_coronas]

    corona_contours = [scale_contour(largest_contour, scaling_coeffs[i]) for i in range(len(radius_coronas))]

    # build a mask for the corona/coronas
    corona_masks = [np.zeros_like(mask) for i in range(len(radius_coronas))]
    for i in range(len(corona_masks)):
        cv.fillPoly(corona_masks[i], [largest_contour, corona_contours[i]], (255, 255, 255))

    # isolate corona
    coronas = [ma.masked_array(image_array, 1 - corona_masks[i]//255) for i in range(len(radius_coronas))]
    for i in range(len(radius_coronas)):
        cv.imwrite('images/corona/coronas' + str(i) + '.png', coronas[i] * 255)

    mean_coronas = [coronas[i].mean() for i in range(len(radius_coronas))]

    # computation of radius, volumes and m for the spheroid's corona
    I_px_coronas = [1 - (1 - elt) for elt in mean_coronas]

    # print("radius", radius)
    # print("coronas thickness", radius - radius_coronas)
    # print("m value", m_value)
    # print("m value coronas", m_value_coronas)

    return radius, radius_coronas_dr, I_px_sphero, I_px_coronas
    # return volume, volume_T, m_value, m_value_corona
    # return mean, std, minimum, maximum, volume, volume_T, m_value


# TODO : éventuellement revoir comment fit les histogrammes
def Img_and_hist(input_filename):

    PixelType = itk.UC
    ImageType = itk.Image[PixelType, dim]
    image = itk.imread(input_filename, PixelType)  # reading the image in grayscale level
    size = image.GetLargestPossibleRegion().GetSize()  # accessing the image size
    dx = Lx/size[0]
    dy = Ly/size[1]
    origin = image.GetLargestPossibleRegion().GetIndex()  # accessing the origin of the image index

    '''cropping the image : the origin is the top left corner and size is the size of the cropped image'''
    origin[0] = 550
    origin[1] = 300
    size[0] = 480
    size[1] = 490

    extract_region = itk.ImageRegion[dim]()
    extract_region.SetIndex(origin)
    extract_region.SetSize(size)

    ExtractFilterType = itk.ExtractImageFilter[itk.Image[PixelType, dim], itk.Image[PixelType, dim]]
    extract_filter = ExtractFilterType.New()
    extract_filter.SetExtractionRegion(extract_region)
    extract_filter.SetInput(image)
    extract_filter.Update()
    cropped_image = extract_filter.GetOutput()

    # creating the image output in the right format
    NewPixelType = itk.F
    NewImageType = itk.Image[NewPixelType, dim]
    output_image = NewImageType.New()
    output_image.SetRegions(cropped_image.GetLargestPossibleRegion())
    output_image.Allocate()

    # normalizing each pixel in the image
    min_val = 0
    max_val = 255
    for i in range(size[0]):
        for j in range(size[1]):
            index = [origin[0]+i, origin[1]+j]
            pixel = cropped_image.GetPixel(index)
            normalized_pixel = (pixel - min_val) / (max_val - min_val)
            output_image.SetPixel(index, normalized_pixel)

    image_array = itk.array_view_from_image(output_image)
    # image_to_scatter = image_array
    # histogram, bins = np.histogram(image_to_scatter, bins='auto', range=(0.0001, 1.))

    '''creating a binary image to facilitate contours detection'''
    binary_image = NewImageType.New()
    binary_image.SetRegions(cropped_image.GetLargestPossibleRegion())
    binary_image.Allocate()
    for i in range(size[0]):
        for j in range(size[1]):
            index = [origin[0]+i, origin[1]+j]
            pixel = output_image.GetPixel(index)
            if pixel < seg_threshold:
                binary_image.SetPixel(index, 1.)
            elif pixel > seg_threshold:
                binary_image.SetPixel(index, 0.)
    binary_array = itk.array_view_from_image(binary_image)
    plt.imsave(output_filename, binary_array, cmap='gray')

    '''using the OpenCV library to extract the largest contour of the spheroid'''
    img = cv.imread(output_filename, cv.IMREAD_GRAYSCALE)

    # applying binary threshold to the image
    _, threshold = cv.threshold(img, 128, 255, cv.THRESH_BINARY)

    # finding all contours in the image
    contours, _ = cv.findContours(threshold, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # finding the largest contour of the spheroid
    largest_contour = None
    largest_area = 0
    for c in contours:
        area = cv.contourArea(c)
        if area > largest_area:
            largest_area = area
            largest_contour = c

    print("largest contour :", largest_contour)

    # drawing the largest contour detected
    img = cv.imread(output_filename)
    image_with_contour = cv.drawContours(img, [largest_contour], -1, (0, 255, 0), 2)

    # writing/displaying the image and the largest contour detected
    # cv.imwrite('images/contour.png', image_with_contour)
    contour_image = np.zeros(binary_array.shape, dtype=np.uint8)
    cv.drawContours(contour_image, [largest_contour], -1, 255, 1)
    # cv.imwrite('images/largest_contour.png', contour_image)

    # build a mask
    '''write and fill the contour to create a mask'''
    mask = np.zeros(binary_array.shape, dtype=np.uint8)
    cv.drawContours(mask, [largest_contour], -1, 255, -1)
    # plt.imshow(mask)
    # plt.show()
    # cv.imwrite('images/mask.png', mask)

    # multiplying the cropped image by the mask
    masked_image = np.multiply(image_array, mask//255)
    # plt.imshow(masked_image)
    # plt.show()
    # spheroid_gif.append((masked_image*255).astype('uint8'))
    # cv.imwrite('images/masked_image.png', masked_image*255)
    #
    # # computing min, max, std, mean values
    #
    # # isolate spheroid
    spheroid = ma.masked_array(image_array, 1 - mask//255)
    image_to_scatter = spheroid

    histogram, bins = np.histogram(spheroid.compressed(), bins='auto', range=(0.0001, 1.))  # spheroid.filled(0)

    index_fit = np.where((spheroid.compressed() < 0.4))
    array_fit = spheroid.compressed()[index_fit]
    # shape, loc, scale = lognorm.fit(array_fit, loc=-0.02, scale=0.106, method="MM")  # fitting with lognormal distrib
    shape, loc, scale = gamma.fit(array_fit, loc=-0.032, scale=0.07, method="MM")  # fitting with gamma distrib

    return image_to_scatter, spheroid.compressed(), bins, shape, loc, scale


# radius, radius_coronas, m_value, m_value_coronas = DataExtraction('images/VID821_G8_1_2021y11m23d_14h00m.PNG')
# radius, radius_coronas, m_value, m_value_coronas = DataExtraction('images/VID811_A4_1_2021y11m15d_11h15m.PNG')

''' gathering all files related to an experiment and writing results'''

directory = 'D:/stage-MONC/SO-Molas-121121 (diapo 15)/'
# list of all spheroids for control, 10 and 50 microM of propranolol
Exp_ID_control = ["A1_", "A2_", "A3_", "A4_", "A5_", "A6_", "A7_", "A8_", "A9_", "A10_", "A11_", "A12_"]
Exp_ID_10mM = ["B1_", "B2_", "B3_", "B4_", "B5_", "B6_", "B7_", "B8_", "B9_", "B10_", "B11_", "B12_"]
Exp_ID_50mM = ["C1_", "C2_", "C3_", "C4_", "C5_", "C6_", "C7_", "C8_", "C9_", "C10_", "C11_", "C12_"]

# list of all spheroids for control, 10 and 50 microM of propranolol after outliers elimination
Exp_ID_control_cleaned = ["A2_", "A3_", "A5_", "A6_", "A7_", "A8_", "A9_", "A10_"]
Exp_ID_10mM_cleaned = ["B2_", "B3_", "B4_", "B5_", "B6_", "B7_", "B8_", "B9_", "B11_"]
Exp_ID_50mM_cleaned = ["C2_", "C3_", "C4_", "C5_", "C6_", "C7_", "C8_", "C9_", "C11_"]


# write some information in textfile  using the DataExtraction function
def write_in_file(list_ID, output_name):
    with open(output_name + ".txt", "w") as f:
        # f.write("ID ExperimentNumber Time Mean std Min Max Area TumorArea RatioTAonA\n")
        f.write("ID ExperimentNumber Time Radius Intensity\n")
        for Exp_ID in list_ID:
            file_list = []
            exp_times = []
            for filename in os.listdir(directory):
                if filename.endswith(".png") and filename.startswith("VID811_" + Exp_ID):
                    file_list.append(filename)
                    exp_times.append(filename[20:29])
            # for i in np.arange(len(file_list)):
            #     if 20 <= i <= 50:
            for i in range(20, 51):
                    filename = file_list[i]
                    full_filename = directory + filename
                    # mean, std, minimum, maximum, volume, volume_T, ratio_volume = DataExtraction(full_filename)
                    radius, radius_coronas, I_px_sphero, I_px_coronas = DataExtraction(full_filename)
                    f.write(Exp_ID + ' ' + Exp_ID[1:-1] + '_' + ' ' + str(i) + ' ' + str(radius) + ' ' + str(I_px_sphero) + '\n')
                    # f.write(Exp_ID + ' ' + Exp_ID[1:-1] + '_' + ' ' + str(i) + ' ' + str(mean) + ' ' + str(std) + ' ' + str(minimum) + ' ' + str(
                    #     maximum) + ' ' + str(volume) + ' ' + str(volume_T) + ' ' + str(ratio_volume) + '\n')
    return


# write some information in textfile  using the DataExtraction function (monolix formalism)
# def write_in_file_monolix(list_ID, output_name):
#     with open(output_name + ".txt", "w") as f:
#         f.write("ID ExperimentNumber Time ObservationID Observation\n")
#         for Exp_ID in list_ID:
#             file_list = []
#             exp_times = []
#             for filename in os.listdir(directory):
#                 if filename.endswith(".png") and filename.startswith("VID811_" + Exp_ID):
#                     file_list.append(filename)
#                     exp_times.append(filename[20:29])
#             # for i in np.arange(len(file_list)):
#             for i in range(20, 51):
#                 filename = file_list[i]
#                 full_filename = directory + filename
#                 radius, radius_coronas, m_value, m_value_coronas = DataExtraction(full_filename)
#                 f.write(Exp_ID[:-1] + ' ' + Exp_ID[1:-1] + ' ' + str(i) + ' ' + '1' + ' ' + str(volume) + '\n')
#             # for i in np.arange(len(file_list)):
#             for i in range(20, 51):
#                 filename = file_list[i]
#                 full_filename = directory + filename
#                 radius, radius_coronas, m_value, m_value_coronas = DataExtraction(full_filename)
#                 f.write(Exp_ID[:-1] + ' ' + Exp_ID[1:-1] + ' ' + str(i) + ' ' + '2' + ' ' + str(volume_T) + '\n')
#             # for i in np.arange(len(file_list)):
#             for i in range(20, 51):
#                 filename = file_list[i]
#                 full_filename = directory + filename
#                 radius, radius_coronas, m_value, m_value_coronas = DataExtraction(full_filename)
#                 f.write(Exp_ID[:-1] + ' ' + Exp_ID[1:-1] + ' ' + str(i) + ' ' + '3' + ' ' + str(m_value) + '\n')
#     return

# write some information in textfile  using the DataExtraction function (monolix 2) m values for differents radius
# def write_in_file_monolix(list_ID, output_name, time):
#     with open(output_name + "_" + str(time) + ".txt", "w") as f:
#         f.write("ID ExpNb Radius ObservationID Observation\n")
#         for Exp_ID in list_ID:
#             file_list = []
#             exp_times = []
#             for filename in os.listdir(directory):
#                 if filename.endswith(".png") and filename.startswith("VID811_" + Exp_ID):
#                     file_list.append(filename)
#                     exp_times.append(filename[20:29])
#             filename = file_list[time]
#             full_filename = directory + filename
#             radius, radius_coronas, m_value, m_value_coronas = DataExtraction(full_filename)
#             f.write(Exp_ID[:-1] + ' ' + Exp_ID[1:-1] + ' ' + str(radius) + ' ' + '1' + ' ' + str(m_value) + '\n')
#             for i in range(N_coro):
#                 f.write(Exp_ID[:-1] + ' ' + Exp_ID[1:-1] + ' ' + str(radius_coronas[i]) + ' ' + '1' + ' ' + str(m_value_coronas[i]) + '\n')
#
#     return

# computation of the mean value of m for control at every time

# allows us to get the mean value for spheroids volume and m
def mean_V(list_ID):
    mean_V = np.zeros(31)
    mean_m = np.zeros(31)
    for Exp_ID in list_ID:
        file_list = []
        exp_times = []
        for filename in os.listdir(directory):
            if filename.endswith(".png") and filename.startswith("VID811_" + Exp_ID):
                file_list.append(filename)
                exp_times.append(filename[20:29])
        for i in range(20, 51):
            filename = file_list[i]
            full_filename = directory + filename
            volume, volume_T, m_value, m_value_corona = DataExtraction(full_filename)
            mean_V[i - 20] += volume
            mean_m[i - 20] += m_value
    mean_V = mean_V / len(list_ID)
    mean_m = mean_m / len(list_ID)
    return mean_V, mean_m


# write some information in textfile  using the DataExtraction function (monolix 3)
def write_in_file_monolix3(list_ID, output_name):
    with open(output_name + ".txt", "w") as f:
        f.write("ID ExperimentNumber Time ObservationID Observation\n")
        # for i in range(20, 51):
        #     f.write('mean_V' + ' ' + '0' + ' ' + str(i) + ' ' + '1' + ' ' + str(mean_array[i - 20]) + '\n')
        for Exp_ID in list_ID:
            file_list = []
            exp_times = []
            for filename in os.listdir(directory):
                if filename.endswith(".png") and filename.startswith("VID811_" + Exp_ID):
                    file_list.append(filename)
                    exp_times.append(filename[20:29])
            for i in range(20, 51):
                filename = file_list[i]
                full_filename = directory + filename
                volume, volume_T, m_value, m_value_corona = DataExtraction(full_filename)
                f.write(Exp_ID[:-1] + ' ' + Exp_ID[1:-1] + ' ' + str(i) + ' ' + '1' + ' ' + str(volume) + '\n')
            for i in range(20, 51):
                filename = file_list[i]
                full_filename = directory + filename
                volume, volume_T, m_value, m_value_corona = DataExtraction(full_filename)
                f.write(Exp_ID[:-1] + ' ' + Exp_ID[1:-1] + ' ' + str(i) + ' ' + '2' + ' '
                        + str(((1-m_value)/(1-mean_m_control[i - 20])) * volume) + '\n')
            for i in range(20, 51):
                filename = file_list[i]
                full_filename = directory + filename
                volume, volume_T, m_value, m_value_corona = DataExtraction(full_filename)
                f.write(Exp_ID[:-1] + ' ' + Exp_ID[1:-1] + ' ' + str(i) + ' ' + '3' + ' '
                        + str((1-m_value)/(1-mean_m_control[i - 20])) + '\n')
    return


def write_in_file_m(list_ID, output_name):
    with open(output_name + ".txt", "w") as f:
        f.write("ID ExpNb Time R m_value \n")
        for Exp_ID in list_ID:
            file_list = []
            exp_times = []
            for filename in os.listdir(directory):
                if filename.endswith(".png") and filename.startswith("VID811_" + Exp_ID):
                    file_list.append(filename)
                    exp_times.append(filename[20:29])
            for i in range(20, 51):
                filename = file_list[i]
                full_filename = directory + filename
                radius, radius_coronas, m_value, m_value_coronas = DataExtraction(full_filename)
                f.write(Exp_ID + ' ' + Exp_ID[1:-1] + '_' + ' ' + str(i) + ' ' + 'R0' + ' ' + str(m_value) + '\n')
                f.write(Exp_ID + ' ' + Exp_ID[1:-1] + '_' + ' ' + str(i) + ' ' + 'R1' + ' ' + str(m_value_coronas[0]) + '\n')
                f.write(Exp_ID + ' ' + Exp_ID[1:-1] + '_' + ' ' + str(i) + ' ' + 'R2' + ' ' + str(m_value_coronas[1]) + '\n')
                f.write(Exp_ID + ' ' + Exp_ID[1:-1] + '_' + ' ' + str(i) + ' ' + 'R3' + ' ' + str(m_value_coronas[2]) + '\n')
    return


# write the mean instensity of pixels and the height h in a textfile
def write_in_file_Iandh(list_ID, output_name, time):
    with open(output_name + "_" + str(time) + ".txt", "w") as f:
        f.write("ID ExpNb I ObservationID Observation\n")
        for Exp_ID in list_ID:
            file_list = []
            exp_times = []
            for filename in os.listdir(directory):
                if filename.endswith(".png") and filename.startswith("VID811_" + Exp_ID):
                    file_list.append(filename)
                    exp_times.append(filename[20:29])
            filename = file_list[time]
            full_filename = directory + filename
            radius, radius_coronas_dr, I_px_sphero, I_px_coronas = DataExtraction(full_filename)
            for j in range(len(radius_coronas_dr)):
                f.write(Exp_ID[:-1] + ' ' + Exp_ID[1:-1] + ' ' + str(I_px_coronas[j]) + ' ' + '1'
                        + ' ' + str(radius_coronas_dr[j]) + '\n')
            # f.write(Exp_ID[:-1] + ' ' + Exp_ID[1:-1] + ' ' + str(I_px_sphero) + ' ' + '1' + ' ' + str(radius) + '\n')
    return


# write_in_file(Exp_ID_control_cleaned, "radius/results_control_r")
# write_in_file(Exp_ID_10mM_cleaned, "radius/results_10microM_r")
# write_in_file(Exp_ID_50mM_cleaned, "radius/results_50microM_r")

# write_in_file(Exp_ID_control_cleaned, "results_control_cleaned")
# write_in_file(Exp_ID_10mM_cleaned, "results_10microM_cleaned")
# write_in_file(Exp_ID_50mM_cleaned, "results_50microM_cleaned")

# write_in_file_monolix(Exp_ID_control_cleaned, "data_control_vol")
# write_in_file_monolix(Exp_ID_10mM_cleaned, "data_10microM_vol")
# write_in_file_monolix(Exp_ID_50mM_cleaned, "data_50microM_vol")

# write_in_file_monolix3(Exp_ID_control_cleaned, "idée3/data_control_M_ok")
# write_in_file_monolix3(Exp_ID_10mM_cleaned, "idée3/data_10microM_M_ok")
# write_in_file_monolix3(Exp_ID_50mM_cleaned, "idée3/data_50microM_M_ok")

# for k in range(20, 51):
#     write_in_file_monolix(Exp_ID_control_cleaned, "data_cleaned/control/data_control", k)
#     write_in_file_monolix(Exp_ID_10mM_cleaned, "data_cleaned/10mM/data_10mM", k)
#     write_in_file_monolix(Exp_ID_50mM_cleaned, "data_cleaned/50mM/data_50mM", k)

for k in range(20, 51):
    # write_in_file_Iandh(Exp_ID_control_cleaned, "Find_h/control/data_control", k)
    write_in_file_Iandh(Exp_ID_10mM_cleaned, "Find_h/10mM/data_10mM", k)
    write_in_file_Iandh(Exp_ID_50mM_cleaned, "Find_h/50mM/data_50mM", k)

# write_in_file_m(Exp_ID_10mM_cleaned, "comparaisons m/results_10microM_m")
# write_in_file_m(Exp_ID_50mM_cleaned, "comparaisons m/results_50microM_m")

'''plotting and saving the results'''

colors = ['blue', 'orange', 'green', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'k']


def plot_images_and_hist(list_ID, output_name):
    for Exp_ID in list_ID:
        exp_times = []
        # hist_gif = []
        for filename in os.listdir(directory):
            if (filename.endswith("14h15m.png") or filename.endswith("02h15m.png")) and filename.startswith("VID811_" + Exp_ID):
                exp_times.append(filename[20:29])

    # fig_img, axs_img = plt.subplots(nrows=len(list_ID), ncols=len(exp_times))
    # fig_hist, axs_hist = plt.subplots(nrows=len(list_ID), ncols=len(exp_times))

    count = 0

    # for Exp_ID in list_ID:
    #     file_list = []
    #     gamma_shape = []
    #     gamma_scale = []
    #
    #     for filename in os.listdir(directory):
    #         if (filename.endswith("14h15m.png") or filename.endswith("02h15m.png")) and filename.startswith("VID811_" + Exp_ID):
    #             file_list.append(filename)
    #     for j in np.arange(len(file_list)):
    #         filename = file_list[j]
    #         print(filename)
    #         full_filename = directory + filename
    #
    #         img, data, bins, shape, loc, scale = Img_and_hist(full_filename)    # histogram, bins,
    #         #axs_hist[count, j].hist(bins[:-1], bins, weights=histogram)
    #         axs_hist[count, j].hist(data, bins='auto', range=(0.0001, 1.), density=True, alpha=0.5)
    #         x = np.linspace(0.0001, 1., 1000)
    #         # pdf = lognorm.pdf(x, shape, loc, scale)
    #         pdf = gamma.pdf(x, shape, loc, scale)
    #         # pdf = exponweib.pdf(x, a, shape, loc, scale)
    #         axs_hist[count, j].plot(x, pdf, 'r-', lw=2)
    #         gamma_shape.append(shape)
    #         gamma_scale.append(scale)
    #
    #         axs_hist[count, j].set_ylim(0., 12.)
    #         axs_img[count, j].imshow(img)
    #
    #         if count == len(list_ID)-1:
    #             hist_xlabel = axs_hist[count, j].set_xlabel(str(j*12)+'h')
    #             img_xlabel = axs_img[count, j].set_xlabel(str(j * 12)+'h')
    #             hist_xlabel.set_color("red")
    #             img_xlabel.set_color("red")
    #             hist_xlabel.set_weight("bold")
    #             img_xlabel.set_weight("bold")
    #         if j == 0:
    #             hist_ylabel = axs_hist[count, j].set_ylabel(Exp_ID[:-1])
    #             img_ylabel = axs_img[count, j].set_ylabel(Exp_ID[:-1])
    #             hist_ylabel.set_color("red")
    #             img_ylabel.set_color("red")
    #             hist_ylabel.set_weight("bold")
    #             img_ylabel.set_weight("bold")
    #
    #     with open("histotests/fitting_data" + Exp_ID + ".txt", "w") as fit:
    #         fit.write("mean and variance for experiment every 12h" + Exp_ID + "\n")
    #         for i in np.arange(len(gamma_scale)):
    #             fit.write(str(gamma_scale[i]*gamma_shape[i]) + ' ' + str(gamma_shape[i]*gamma_scale[i]*gamma_scale[i]) + "\n")
    #     fit.close()
    #
    #     count += 1
    #     print(count)

    fitting_plot = plt.figure
    time_plot = [j*12 for j in np.arange(len(exp_times))]
    list_colors = ['r', 'b', 'g', 'c', 'm', 'y', 'orange', 'pink', 'olive', 'cyan', 'purple', 'brown']
    print("time plot", time_plot)
    j = 0
    print(len(list_ID))
    for Exp_ID in list_ID:
        data_fit = np.loadtxt("histotests/fitting_data" + Exp_ID + ".txt", skiprows=1, usecols=(0, 1))

        plt.plot(time_plot[:], data_fit[:, 0], color=list_colors[j], label=Exp_ID)
        plt.plot(time_plot[:], data_fit[:, 1], '--', color=list_colors[j])
        plt.ylim(0.015, 0.2)

        j += 1

    plt.legend()
    plt.show()

    figure = plt.gcf()
    figure.set_size_inches(50, 50)

    fig_img.set_figheight(20)
    fig_img.set_figwidth(20)
    fig_img.tight_layout()
    fig_img.savefig('scattered_images_' + output_name + '.pdf', dpi=300)
    fig_hist.savefig('histograms_' + output_name + '.pdf')

    return


# plot_images_and_hist(Exp_ID_control, "results_control")
# plot_images_and_hist(Exp_ID_10mM, "results_10microM")
# plot_images_and_hist(Exp_ID_50mM, "results_50microM")


def plot_file(list_ID, output_name):
    # data = np.loadtxt(output_name + '.txt', usecols=(1, 2, 3, 4, 5, 6, 7, 8), skiprows=1)
    dtypes = [('col0', 'U10'), ('col1', 'U10'), ('col2', float), ('col3', float),
              ('col4', float), ('col5', float), ('col6', float), ('col7', float), ('col8', float)]
    data = np.loadtxt(output_name + '.txt', dtype=dtypes, skiprows=1, usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8))

    # fig, axs = plt.subplots(nrows=2, ncols=2)
    for Exp_ID in list_ID:
        Exp_mask = np.char.startswith(data['col0'].astype(str), Exp_ID)
        data_Exp = data[Exp_mask]
        print(data_Exp)
        plt.plot(data_Exp['col2'], data_Exp['col7'], label=Exp_ID)
        plt.legend(loc='best')
        # axs[0, 0].plot(data_Exp[:, 1], data_Exp[:, 6], label=Exp_ID)
        # axs[0, 0].legend(loc='best')

    plt.gca().set_prop_cycle('color', colors)
    plt.ylim(0.2, 1.)
    # plt.tight_layout()
    plt.savefig(output_name + '_test' + '.pdf')

    return


def plot_volumes():

    dtypes = [('col0', 'U10'), ('col1', 'U10'), ('col2', float), ('col3', float),
              ('col4', float), ('col5', float), ('col6', float), ('col7', float), ('col8', float)]
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

    data = np.loadtxt('results_control' + '.txt', dtype=dtypes, skiprows=1, usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8))
    for Exp_ID in Exp_ID_control:
        Exp_mask = np.char.startswith(data['col0'].astype(str), Exp_ID)
        data_Exp = data[Exp_mask]
        ax1.plot(data_Exp['col2'], data_Exp['col7'], label=Exp_ID[:-1])
        # ax1.legend(loc='best')
        ax1.set_ylim(0.2, 1)

    data = np.loadtxt("results_10microM" + '.txt', dtype=dtypes, skiprows=1, usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8))
    for Exp_ID in Exp_ID_10mM:
        Exp_mask = np.char.startswith(data['col0'].astype(str), Exp_ID)
        data_Exp = data[Exp_mask]
        ax2.plot(data_Exp['col2'], data_Exp['col7'], label=Exp_ID[:-1])
        # ax2.legend(loc='best')
        ax2.set_ylim(0.2, 1)

    data = np.loadtxt("results_50microM" + '.txt', dtype=dtypes, skiprows=1, usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8))
    for Exp_ID in Exp_ID_50mM:
        Exp_mask = np.char.startswith(data['col0'].astype(str), Exp_ID)
        data_Exp = data[Exp_mask]
        ax3.plot(data_Exp['col2'], data_Exp['col7'], label=Exp_ID[:-1])
        # ax3.legend(loc='best')
        ax3.set_ylim(0.2, 1)

    ax1.set(title='results_control')
    ax2.set(title='results_10microM')
    ax3.set(title='results_50microM')

    plt.gca().set_prop_cycle('color', colors)

    figure = plt.gcf()
    figure.set_size_inches(15, 6)

    plt.savefig('volumes.pdf')

    return


def plot_radius():

    dtypes = [('col0', 'U10'), ('col1', 'U10'), ('col2', float), ('col3', float),
              ('col4', float)]
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

    data = np.loadtxt('radius/results_control_r' + '.txt', dtype=dtypes, skiprows=1, usecols=(0, 1, 2, 3, 4))
    for Exp_ID in Exp_ID_control_cleaned:
        Exp_mask = np.char.startswith(data['col0'].astype(str), Exp_ID)
        data_Exp = data[Exp_mask]
        ax1.plot(data_Exp['col2'], data_Exp['col3'], label=Exp_ID[:-1])
        ax1.legend(loc='best')
        ax1.set_ylim(0.3, 0.55)

    data = np.loadtxt("radius/results_10microM_r" + '.txt', dtype=dtypes, skiprows=1, usecols=(0, 1, 2, 3, 4))
    for Exp_ID in Exp_ID_10mM_cleaned:
        Exp_mask = np.char.startswith(data['col0'].astype(str), Exp_ID)
        data_Exp = data[Exp_mask]
        ax2.plot(data_Exp['col2'], data_Exp['col3'], label=Exp_ID[:-1])
        ax2.legend(loc='best')
        ax2.set_ylim(0.3, 0.55)

    data = np.loadtxt("radius/results_50microM_r" + '.txt', dtype=dtypes, skiprows=1, usecols=(0, 1, 2, 3, 4))
    for Exp_ID in Exp_ID_50mM_cleaned:
        Exp_mask = np.char.startswith(data['col0'].astype(str), Exp_ID)
        data_Exp = data[Exp_mask]
        ax3.plot(data_Exp['col2'], data_Exp['col3'], label=Exp_ID[:-1])
        ax3.legend(loc='best')
        ax3.set_ylim(0.3, 0.55)

    ax1.set(title='results_control')
    ax2.set(title='results_10microM')
    ax3.set(title='results_50microM')

    ax1.set_xlabel('time (h)')
    ax1.set_ylabel('radius (mm)')
    ax2.set_xlabel('time (h)')
    ax2.set_ylabel('radius (mm)')
    ax3.set_xlabel('time (h)')
    ax3.set_ylabel('radius (mm)')


    plt.gca().set_prop_cycle('color', colors)

    figure = plt.gcf()
    figure.set_size_inches(15, 6)

    plt.savefig('radius/radius.pdf')

    return


plot_radius()


def plot_m(list_ID_1, list_ID_2, list_ID_3, output_name):

    dtypes = [('col0', 'U10'), ('col1', 'U10'), ('col2', float), ('col3', 'U10'), ('col4', float)]
    fig, axs = plt.subplots(nrows=4, ncols=4)

    data = np.loadtxt('comparaisons m/results_control_m.txt', dtype=dtypes, skiprows=1, usecols=(0, 1, 2, 3, 4))
    for Exp_ID in list_ID_1:
        Exp_mask = np.char.startswith(data['col0'].astype(str), Exp_ID)
        data_Exp = data[Exp_mask]
        Exp_mask = np.char.startswith(data_Exp['col3'].astype(str), "R0")
        data_Exp = data_Exp[Exp_mask]
        axs[0, 1].plot(data_Exp['col2'], data_Exp['col4'], 'o-', lw=1, label=Exp_ID)
    for Exp_ID in list_ID_1:
        Exp_mask = np.char.startswith(data['col0'].astype(str), Exp_ID)
        data_Exp = data[Exp_mask]
        Exp_mask = np.char.startswith(data_Exp['col3'].astype(str), "R1")
        data_Exp = data_Exp[Exp_mask]
        axs[1, 1].plot(data_Exp['col2'], data_Exp['col4'], 'o-', lw=1, label=Exp_ID)
    for Exp_ID in list_ID_1:
        Exp_mask = np.char.startswith(data['col0'].astype(str), Exp_ID)
        data_Exp = data[Exp_mask]
        Exp_mask = np.char.startswith(data_Exp['col3'].astype(str), "R2")
        data_Exp = data_Exp[Exp_mask]
        axs[2, 1].plot(data_Exp['col2'], data_Exp['col4'], 'o-', lw=1, label=Exp_ID)
    for Exp_ID in list_ID_1:
        Exp_mask = np.char.startswith(data['col0'].astype(str), Exp_ID)
        data_Exp = data[Exp_mask]
        Exp_mask = np.char.startswith(data_Exp['col3'].astype(str), "R3")
        data_Exp = data_Exp[Exp_mask]
        axs[3, 1].plot(data_Exp['col2'], data_Exp['col4'], 'o-', lw=1, label=Exp_ID)

    data = np.loadtxt('comparaisons m/results_10microM_m.txt', dtype=dtypes, skiprows=1, usecols=(0, 1, 2, 3, 4))
    for Exp_ID in list_ID_2:
        Exp_mask = np.char.startswith(data['col0'].astype(str), Exp_ID)
        data_Exp = data[Exp_mask]
        Exp_mask = np.char.startswith(data_Exp['col3'].astype(str), "R0")
        data_Exp = data_Exp[Exp_mask]
        axs[0, 2].plot(data_Exp['col2'], data_Exp['col4'], 'o-', lw=1, label=Exp_ID)
    for Exp_ID in list_ID_2:
        Exp_mask = np.char.startswith(data['col0'].astype(str), Exp_ID)
        data_Exp = data[Exp_mask]
        Exp_mask = np.char.startswith(data_Exp['col3'].astype(str), "R1")
        data_Exp = data_Exp[Exp_mask]
        axs[1, 2].plot(data_Exp['col2'], data_Exp['col4'], 'o-', lw=1, label=Exp_ID)
    for Exp_ID in list_ID_2:
        Exp_mask = np.char.startswith(data['col0'].astype(str), Exp_ID)
        data_Exp = data[Exp_mask]
        Exp_mask = np.char.startswith(data_Exp['col3'].astype(str), "R2")
        data_Exp = data_Exp[Exp_mask]
        axs[2, 2].plot(data_Exp['col2'], data_Exp['col4'], 'o-', lw=1, label=Exp_ID)
    for Exp_ID in list_ID_2:
        Exp_mask = np.char.startswith(data['col0'].astype(str), Exp_ID)
        data_Exp = data[Exp_mask]
        Exp_mask = np.char.startswith(data_Exp['col3'].astype(str), "R3")
        data_Exp = data_Exp[Exp_mask]
        axs[3, 2].plot(data_Exp['col2'], data_Exp['col4'], 'o-', lw=1, label=Exp_ID)

    data = np.loadtxt('comparaisons m/results_50microM_m.txt', dtype=dtypes, skiprows=1, usecols=(0, 1, 2, 3, 4))
    for Exp_ID in list_ID_3:
        Exp_mask = np.char.startswith(data['col0'].astype(str), Exp_ID)
        data_Exp = data[Exp_mask]
        Exp_mask = np.char.startswith(data_Exp['col3'].astype(str), "R0")
        data_Exp = data_Exp[Exp_mask]
        axs[0, 3].plot(data_Exp['col2'], data_Exp['col4'], 'o-', lw=1, label=Exp_ID)
    for Exp_ID in list_ID_3:
        Exp_mask = np.char.startswith(data['col0'].astype(str), Exp_ID)
        data_Exp = data[Exp_mask]
        Exp_mask = np.char.startswith(data_Exp['col3'].astype(str), "R1")
        data_Exp = data_Exp[Exp_mask]
        axs[1, 3].plot(data_Exp['col2'], data_Exp['col4'], 'o-', lw=1, label=Exp_ID)
    for Exp_ID in list_ID_3:
        Exp_mask = np.char.startswith(data['col0'].astype(str), Exp_ID)
        data_Exp = data[Exp_mask]
        Exp_mask = np.char.startswith(data_Exp['col3'].astype(str), "R2")
        data_Exp = data_Exp[Exp_mask]
        axs[2, 3].plot(data_Exp['col2'], data_Exp['col4'], 'o-', lw=1, label=Exp_ID)
    for Exp_ID in list_ID_3:
        Exp_mask = np.char.startswith(data['col0'].astype(str), Exp_ID)
        data_Exp = data[Exp_mask]
        Exp_mask = np.char.startswith(data_Exp['col3'].astype(str), "R3")
        data_Exp = data_Exp[Exp_mask]
        axs[3, 3].plot(data_Exp['col2'], data_Exp['col4'], 'o-', lw=1, label=Exp_ID)

    axs[0, 0].imshow(mpimg.imread('comparaisons m/images coronas/spheroid.png'))
    axs[1, 0].imshow(mpimg.imread('comparaisons m/images coronas/coronas0.png'))
    axs[2, 0].imshow(mpimg.imread('comparaisons m/images coronas/coronas1.png'))
    axs[3, 0].imshow(mpimg.imread('comparaisons m/images coronas/coronas2.png'))
    for i in range(4):
        for j in range(1, 4):
            axs[i, j].set_ylim(0.05, 0.3)
            axs[i, j].set_xlabel('Time (h)', fontsize=22)
            axs[i, j].set_ylabel('m', fontsize=22)
            axs[i, j].legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    axs[0, 0].set_title('image example \n', color='red', fontsize=25, weight='bold')
    axs[0, 1].set_title('control \n', color='red', fontsize=25, weight='bold')
    axs[0, 2].set_title('10 microM \n', color='red', fontsize=25, weight='bold')
    axs[0, 3].set_title('50 microM \n', color='red', fontsize=25, weight='bold')
    figure = plt.gcf()
    figure.set_size_inches(50, 50)
    plt.savefig(output_name + '.pdf')

    return


# plot_m(Exp_ID_control_cleaned, Exp_ID_10mM_cleaned, Exp_ID_50mM_cleaned, 'comparaisons m/m_comparisons')


def plot_MandR(list_ID, output_name):
    dtypes = [('col0', 'U10'), ('col1', 'U10'), ('col2', 'U10'), ('col3', float), ('col4', float)]
    fig, axs = plt.subplots(nrows=2, ncols=2)
    data = np.loadtxt(output_name + '.txt', dtype=dtypes, skiprows=1, usecols=(0, 1, 2, 3, 4))
    for Exp_ID in list_ID:
        Exp_mask = np.char.startswith(data['col0'].astype(str), Exp_ID)
        data_Exp = data[Exp_mask]
        Exp_mask = np.char.startswith(data_Exp['col2'].astype(str), "20")
        data_Exp = data_Exp[Exp_mask]
        # axs[0, 0].plot(data_Exp['col3'], data_Exp['col4'], 'o', lw=1, label=Exp_ID)
        axs[0, 0].plot(data_Exp['col4'], data_Exp['col3'], 'o', lw=1, label=Exp_ID)
    for Exp_ID in list_ID:
        Exp_mask = np.char.startswith(data['col0'].astype(str), Exp_ID)
        data_Exp = data[Exp_mask]
        Exp_mask = np.char.startswith(data_Exp['col2'].astype(str), "30")
        data_Exp = data_Exp[Exp_mask]
        # axs[0, 1].plot(data_Exp['col3'], data_Exp['col4'], 'o', lw=1, label=Exp_ID)
        axs[0, 1].plot(data_Exp['col4'], data_Exp['col3'], 'o', lw=1, label=Exp_ID)
    for Exp_ID in list_ID:
        Exp_mask = np.char.startswith(data['col0'].astype(str), Exp_ID)
        data_Exp = data[Exp_mask]
        Exp_mask = np.char.startswith(data_Exp['col2'].astype(str), "40")
        data_Exp = data_Exp[Exp_mask]
        # axs[1, 0].plot(data_Exp['col3'], data_Exp['col4'], 'o', lw=1, label=Exp_ID)
        axs[1, 0].plot(data_Exp['col4'], data_Exp['col3'], 'o', lw=1, label=Exp_ID)
    for Exp_ID in list_ID:
        Exp_mask = np.char.startswith(data['col0'].astype(str), Exp_ID)
        data_Exp = data[Exp_mask]
        Exp_mask = np.char.startswith(data_Exp['col2'].astype(str), "50")
        data_Exp = data_Exp[Exp_mask]
        # axs[1, 1].plot(data_Exp['col3'], data_Exp['col4'], 'o', lw=1, label=Exp_ID)
        axs[1, 1].plot(data_Exp['col4'], data_Exp['col3'], 'o', lw=1, label=Exp_ID)


    # axs[0, 0].set_title('Pour t = 20h')
    # axs[0, 0].set_xlabel('hauteur h')
    # axs[0, 0].set_ylabel('1 - m(I)')
    # axs[0, 0].legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    # axs[0, 1].set_title('Pour t = 30h')
    # axs[0, 1].set_xlabel('hauteur h')
    # axs[0, 1].set_ylabel('1 - m(I)')
    # axs[0, 1].legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    # axs[1, 0].set_title('Pour t = 40h')
    # axs[1, 0].set_xlabel('hauteur h')
    # axs[1, 0].set_ylabel('1 - m(I)')
    # axs[1, 0].legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    # axs[1, 1].set_title('Pour t = 50h')
    # axs[1, 1].set_xlabel('hauteur h')
    # axs[1, 1].set_ylabel('1 - m(I)')
    # axs[1, 1].legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    # axs[0, 0].set_xlim(0., 0.5)
    # axs[0, 0].set_ylim(0., 0.25)
    # axs[1, 0].set_xlim(0., 0.5)
    # axs[1, 0].set_ylim(0., 0.25)
    # axs[0, 1].set_xlim(0., 0.5)
    # axs[0, 1].set_ylim(0., 0.25)
    # axs[1, 1].set_xlim(0., 0.5)
    # axs[1, 1].set_ylim(0., 0.25)

    axs[0, 0].set_title('Pour t = 20h')
    axs[0, 0].set_ylabel('hauteur h')
    axs[0, 0].set_xlabel('1 - m(I)')
    axs[0, 0].legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    axs[0, 1].set_title('Pour t = 30h')
    axs[0, 1].set_ylabel('hauteur h')
    axs[0, 1].set_xlabel('1 - m(I)')
    axs[0, 1].legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    axs[1, 0].set_title('Pour t = 40h')
    axs[1, 0].set_ylabel('hauteur h')
    axs[1, 0].set_xlabel('1 - m(I)')
    axs[1, 0].legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    axs[1, 1].set_title('Pour t = 50h')
    axs[1, 1].set_ylabel('hauteur h')
    axs[1, 1].set_xlabel('1 - m(I)')
    axs[1, 1].legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    axs[0, 0].set_ylim(0., 0.5)
    axs[0, 0].set_xlim(0.04, 0.25)
    axs[1, 0].set_ylim(0., 0.5)
    axs[1, 0].set_xlim(0., 0.25)
    axs[0, 1].set_ylim(0., 0.5)
    axs[0, 1].set_xlim(0., 0.25)
    axs[1, 1].set_ylim(0., 0.5)
    axs[1, 1].set_xlim(0., 0.25)

    figure = plt.gcf()
    figure.set_size_inches(20, 10)
    plt.savefig(output_name + '_inv' + '.pdf')

    return


# plot_MandR(Exp_ID_control_cleaned,'Find_h/results_control_h')
# plot_MandR(Exp_ID_10mM_cleaned, 'MandR/results_10mM_mandr')
# plot_MandR(Exp_ID_50mM_cleaned, 'MandR/results_50mM_mandr')

# write_in_file(Exp_ID_control_cleaned, "results_control_cleaned")
# write_in_file(Exp_ID_control, "results_control_cleaned")
# write_in_file(Exp_ID_control, "results_control_cleaned")

def gompertz_model(t, a0, a, b):
    A = a0 * np.exp(a/b * (1 - np.exp(-b*(t-20))))
    return A


'''observation + gompertz + PDE'''
def plot_volumes_cleaned():

    dtypes = [('col0', 'U10'), ('col1', 'U10'), ('col2', float), ('col3', float),
              ('col4', float), ('col5', float), ('col6', float), ('col7', float), ('col8', float), ('col9', float)]
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

    data = np.loadtxt('results_control_cleaned' + '.txt', dtype=dtypes, skiprows=1, usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9))
    for Exp_ID in Exp_ID_control_cleaned:
        Exp_mask = np.char.startswith(data['col0'].astype(str), Exp_ID)
        data_Exp = data[Exp_mask]
        ax1.plot(data_Exp['col2'], data_Exp['col7'], 'blue', lw=1, label='Area')
        ax1.plot(data_Exp['col2'], data_Exp['col8'], 'green', lw=1, label='Tumor Area')
        ax1.plot(data_Exp['col2'], data_Exp['col9'], 'red', lw=1, label='AonAT')
        # ax1.legend(loc='best')
        ax1.set_xlim(20, 50)
        ax1.set_ylim(0.2, 1.25)

    data = np.loadtxt("results_10microM_cleaned" + '.txt', dtype=dtypes, skiprows=1, usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9))
    for Exp_ID in Exp_ID_10mM_cleaned:
        Exp_mask = np.char.startswith(data['col0'].astype(str), Exp_ID)
        data_Exp = data[Exp_mask]
        ax2.plot(data_Exp['col2'], data_Exp['col7'], 'blue', lw=1, label='Area')
        ax2.plot(data_Exp['col2'], data_Exp['col8'], 'green', lw=1, label='Tumor Area')
        ax2.plot(data_Exp['col2'], data_Exp['col9'], 'red', lw=1, label='AonAT')
        # ax2.legend(loc='best')
        ax2.set_xlim(20, 50)
        ax2.set_ylim(0.2, 1.25)

    data = np.loadtxt("results_50microM_cleaned" + '.txt', dtype=dtypes, skiprows=1, usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9))
    for Exp_ID in Exp_ID_50mM_cleaned:
        Exp_mask = np.char.startswith(data['col0'].astype(str), Exp_ID)
        data_Exp = data[Exp_mask]
        ax3.plot(data_Exp['col2'], data_Exp['col7'], 'blue', lw=1, label='Area')
        ax3.plot(data_Exp['col2'], data_Exp['col8'], 'green', lw=1, label='Tumor Area')
        ax3.plot(data_Exp['col2'], data_Exp['col9'], 'red', lw=1, label='AonT')
        # ax3.legend(loc='best')
        ax3.set_xlim(20, 50)
        ax3.set_ylim(0.2, 1.25)

    ax1.set(title='results_control')
    ax2.set(title='results_10microM')
    ax3.set(title='results_50microM')

    plt.gca().set_prop_cycle('color', colors)

    figure = plt.gcf()
    figure.set_size_inches(15, 6)
    figure.set_size_inches(15, 6)

    plt.savefig('volumes_and_ratio.pdf')

    return

# write_in_file(Exp_ID_control_cleaned, "results_control_cleaned")
# plot_volumes_cleaned()


def plot_mean():

    dtypes = [('col0', 'U10'), ('col1', 'U10'), ('col2', float), ('col3', float),
              ('col4', float), ('col5', float), ('col6', float), ('col7', float), ('col8', float)]
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

    data = np.loadtxt('results_control' + '.txt', dtype=dtypes, skiprows=1, usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8))
    for Exp_ID in Exp_ID_control:
        Exp_mask = np.char.startswith(data['col0'].astype(str), Exp_ID)
        data_Exp = data[Exp_mask]
        ax1.plot(data_Exp['col2'], data_Exp['col3'], label=Exp_ID[:-1])
        ax1.legend(loc='best')
        ax1.set_ylim(0.015, 0.2)

    data = np.loadtxt("results_10microM" + '.txt', dtype=dtypes, skiprows=1, usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8))
    for Exp_ID in Exp_ID_10mM:
        Exp_mask = np.char.startswith(data['col0'].astype(str), Exp_ID)
        data_Exp = data[Exp_mask]
        ax2.plot(data_Exp['col2'], data_Exp['col3'], label=Exp_ID[:-1])
        ax2.legend(loc='best')
        ax2.set_ylim(0.015, 0.2)

    data = np.loadtxt("results_50microM" + '.txt', dtype=dtypes, skiprows=1, usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8))
    for Exp_ID in Exp_ID_50mM:
        Exp_mask = np.char.startswith(data['col0'].astype(str), Exp_ID)
        data_Exp = data[Exp_mask]
        ax3.plot(data_Exp['col2'], data_Exp['col3'], label=Exp_ID[:-1])
        ax3.legend(loc='best')
        ax3.set_ylim(0.015, 0.2)

    ax1.set(title='results_control')
    ax2.set(title='results_10microM')
    ax3.set(title='results_50microM')

    plt.gca().set_prop_cycle('color', colors)

    figure = plt.gcf()
    figure.set_size_inches(15, 6)

    plt.savefig('mean.pdf')

    return


# plot_file(Exp_ID_control, 'results_control')
# plot_file(Exp_ID_10mM, "results_10microM")
# plot_file(Exp_ID_50mM, "results_50microM")

# plot_volumes()
# plot_mean()
