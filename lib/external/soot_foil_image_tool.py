"""

    soot_foil_image_tool.py

Originally from:
    https://github.com/dannyjalon/soot-foil-image-tool

From 
    Jalontzki, D., Zussman, A., Pendurkar, S., Sharon, G., and Kozak, Y. (2025). *A computer vision
approach for analysis of detonation cellular structures.* Applications in Energy in Combustion 
Science. Volume 23. DOI: [10.1016/j.jaecs.2025.100340](https://www.sciencedirect.com/science/article/pii/S2666352X25000226?via%3Dihub)

    Contains computer vision algorithms and functions that allow measurement of cells for a given 
image.

"""


import sys
import os
import argparse
import cv2
import numpy as np
from scipy.stats import norm
from matplotlib import pyplot as plt
import time
from tqdm import tqdm
import csv


# ----------------------- Utility Printing Functions -----------------------
def print_error_info(string):
    print('\x1b[1;31;40m{0}\x1b[0m'.format(string))


def print_info(string):
    print('\x1b[1;35;40m{0}\x1b[0m'.format(string))


def print_sub(string):
    print('\x1b[1;36;40m{0}\x1b[0m'.format(string))


def get_dimension_flags(dimension_value):
    flag_height, flag_width = 0, 0
    if dimension_value.lower() in ['h', 'height']:
        flag_height = 1
    elif dimension_value.lower() in ['w', 'width']:
        flag_width = 1
    return flag_height, flag_width


# ----------------------- Functions for Command-line Parsing -----------------------

def get_args_from_parser():
    """
        Build & return the argparse parser for:
          - input image paths
          - measurement dimension
          - physical size
          - min_area & divisor_of_max_area
        """

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', nargs='*', help='Path(s) of the input image(s)', required=True)
    parser.add_argument('-d', '--dimension', nargs='*',
                        help='Measurement dimension. Options: h (or height), w (or width)', required=True)
    parser.add_argument('-s', '--size', nargs='*',
                        help='The size of the image according to the dimension (in cm)', required=True)
    parser.add_argument('-min_area', nargs='*',
                        help='Minimum contour area. Default is 40', default=[40], required=False)
    parser.add_argument('-divisor_of_max_area', nargs='*',
                        help='Divisor for max contour area (height*width/<value>). Default is 8',
                        default=[8], required=False)
    args = parser.parse_args()
    return args


def args_for_input_image(args):
    """
        Verify each -i/--input path exists; exit on any missing file.
        """

    if len(args.input) < 1:
        print_error_info('Error: No value provided for -i (--input)')
        sys.exit()
    image_list = []
    for image in args.input:
        if os.path.isfile(image):
            image_list.append(image)
        else:
            print_error_info('Error: Invalid image path "{0}"'.format(image))
            sys.exit()
    return image_list


def args_for_dimension(args):
    """
        Normalize each -d/--dimension to 'h' or 'w'; exit on invalid flag.
        """

    if len(args.dimension) < 1:
        print_error_info('Error: No value provided for -d (--dimension)')
        sys.exit()
    dimension_list = []
    for argument in args.dimension:
        if argument.lower() in ['h', 'height']:
            dimension_list.append('h')
        elif argument.lower() in ['w', 'width']:
            dimension_list.append('w')
        else:
            print_error_info('Error: Invalid dimension (should be h or w)')
            sys.exit()
    return dimension_list


def args_for_size(args):
    """
        Parse each -s/--size as float (cm); exit on invalid numeric strings.
        """

    if len(args.size) < 1:
        print_error_info('Error: No value provided for -s (--size)')
        sys.exit()
    size_list = []
    for argument in args.size:
        argument = str(argument)
        if "." in argument:
            before_dot, after_dot = argument.split(".")[0], argument.split(".")[1]
            if not before_dot.isdigit() or not after_dot.isdigit():
                print_error_info('Error: Invalid size value in -s (--size)')
                sys.exit()
        else:
            if not argument.isdigit():
                print_error_info('Error: Invalid size value in -s (--size)')
                sys.exit()
        size_value = float(argument)
        size_list.append(size_value)
    return size_list


def check_if_lengths_of_lists_are_equal(image_list, dim_list, size_list):
    """
        Ensure counts of images, dimensions & sizes match exactly.
        """

    if len(image_list) != len(dim_list):
        print_error_info('Error: Number of images and dimensions do not match')
        sys.exit()
    if len(image_list) != len(size_list):
        print_error_info('Error: Number of images and sizes do not match')
        sys.exit()
    if len(dim_list) != len(size_list):
        print_error_info('Error: Number of dimensions and sizes do not match')
        sys.exit()


def args_for_min_area(args):
    """
        Parse -min_area list into float values (one per image); exit on errors.
        """

    min_area_list = []
    if len(args.min_area) < 1:
        print_error_info('Error: No value provided for -min_area')
        sys.exit()
    elif len(args.min_area) == 1:
        argument = str(args.min_area[0])
        if "." in argument:
            before_dot, after_dot = argument.split(".")[0], argument.split(".")[1]
            if not before_dot.isdigit() or not after_dot.isdigit():
                print_error_info('Error: Invalid -min_area value')
                sys.exit()
        else:
            if not argument.isdigit():
                print_error_info('Error: Invalid -min_area value')
                sys.exit()
        for image_name in args.input:
            min_area_list.append(float(argument))
    else:
        for argument in args.min_area:
            argument = str(argument)
            if "." in argument:
                before_dot, after_dot = argument.split(".")[0], argument.split(".")[1]
                if not before_dot.isdigit() or not after_dot.isdigit():
                    print_error_info('Error: Invalid -min_area value')
                    sys.exit()
            else:
                if not argument.isdigit():
                    print_error_info('Error: Invalid -min_area value')
                    sys.exit()
            min_area_value = float(argument)
            min_area_list.append(min_area_value)
    return min_area_list


def args_for_divisor_of_max_area(args):
    """
        Parse -divisor_of_max_area list into float divisors (one per image).
        """

    divisor_of_max_area_list = []
    if len(args.divisor_of_max_area) < 1:
        print_error_info('Error: No value provided for -divisor_of_max_area')
        sys.exit()
    elif len(args.divisor_of_max_area) == 1:
        argument = str(args.divisor_of_max_area[0])
        if "." in argument:
            before_dot, after_dot = argument.split(".")[0], argument.split(".")[1]
            if not before_dot.isdigit() or not after_dot.isdigit():
                print_error_info('Error: Invalid -divisor_of_max_area value')
                sys.exit()
        else:
            if not argument.isdigit():
                print_error_info('Error: Invalid -divisor_of_max_area value')
                sys.exit()
        for image_name in args.input:
            divisor_of_max_area_list.append(float(argument))
    else:
        for argument in args.divisor_of_max_area:
            argument = str(argument)
            if "." in argument:
                before_dot, after_dot = argument.split(".")[0], argument.split(".")[1]
                if not before_dot.isdigit() or not after_dot.isdigit():
                    print_error_info('Error: Invalid -divisor_of_max_area value')
                    sys.exit()
            else:
                if not argument.isdigit():
                    print_error_info('Error: Invalid -divisor_of_max_area value')
                    sys.exit()
            divisor_of_max_area_value = float(argument)
            divisor_of_max_area_list.append(divisor_of_max_area_value)
    return divisor_of_max_area_list


# ----------------------- Image Processing Functions -----------------------
def get_rotating_image(image_name, flag_height, flag_width):
    """
        Load the image; rotate 90° clockwise if measuring width.
        """

    original_image = cv2.imread(image_name)
    if original_image is None:
        print_error_info(f"Cannot read image: {image_name}")
        sys.exit()
    if flag_height:
        image = original_image
    elif flag_width:
        image = cv2.rotate(original_image, cv2.ROTATE_90_CLOCKWISE)
    image_copy = image.copy()
    return image, image_copy


def get_image_shape(image):
    """
        Return (height, width, channels) of the given image array.
        """

    height, width, channels = image.shape
    return height, width, channels


def morphological_image_processing(image, debug, verbosity=1):
    """
        Apply CLAHE → background subtraction → Otsu threshold → denoising.
        If debug=1, display the binary result.
        """

    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl1 = clahe.apply(img_gray)
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
    bg = cv2.morphologyEx(cl1, cv2.MORPH_DILATE, se)
    out_gray = cv2.divide(cl1, bg, scale=255)
    out_binary = cv2.threshold(out_gray, 0, 255, cv2.THRESH_OTSU)[1]
    morphology_image = cv2.fastNlMeansDenoising(out_binary, 10.0, 7, 21)
    if debug:
        print("\tShowing morphology image")
        cv2.imshow('morphology_image', morphology_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print("\tClosing morphology preview")
    return morphology_image


def detect_contours_in_image(image, morphology_image, height, width, flag_height, flag_width, min_area, max_area):
    """
        Find all contours in the binary image, filter by MIN_AREA < area < MAX_AREA,
        compute each contour’s extreme points (left/right/top/bottom),
        and return lists plus the percentage Loss metric.
        """

    cnts, _ = cv2.findContours(morphology_image.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    cnts = sorted(cnts, key=lambda x: cv2.contourArea(x))  # sort contours by area: small -> big
    MAX_AREA = max_area
    MIN_AREA = min_area
    cntsfiltered = [cnt for cnt in cnts if MIN_AREA < cv2.contourArea(cnt) and MAX_AREA > cv2.contourArea(cnt)]
    area_list, extLeft_list, extRight_list, extTop_list, extBot_list = [], [], [], [], []
    for c in cntsfiltered:
        area = cv2.contourArea(c)
        extLeft = list(c[c[:, :, 0].argmin()][0])
        extRight = list(c[c[:, :, 0].argmax()][0])
        extTop = list(c[c[:, :, 1].argmin()][0])
        extBot = list(c[c[:, :, 1].argmax()][0])
        if flag_height:
            extLeft[0] -= 5
            extRight[0] += 5
            extTop[1] -= 5
            extBot[1] += 5
        elif flag_width:
            extLeft[0] -= 10
            extRight[0] += 10
            extTop[1] -= 10
            extBot[1] += 10
        area_list.append(area)
        extLeft_list.append(extLeft)
        extRight_list.append(extRight)
        extTop_list.append(extTop)
        extBot_list.append(extBot)
    Loss = (abs((height * width) - sum(area_list)) / (
                height * width) * 100)  # (Image area - area of all contours detected) / Image area (%)
    return area_list, extLeft_list, extRight_list, extTop_list, extBot_list, Loss


def is_junction(extLeft_list, extRight_list, extTop_list, extBot_list, area_list, min_area):
    """
        Identify junction points whose top/bottom/left/right extremities
        cluster within a critical radius r_crit.
        Returns: (junction_list, r_crit, junction_count).
        """

    junction_list = []
    jun_num = 0
    r_crit = (min_area / (np.pi)) ** 0.5  # defines the junction's minimal area
    for i in range(0, len(extTop_list)):
        for j in range(0, len(extBot_list)):
            if j != i:
                top_bot_dist = (((extTop_list[i][0] - extBot_list[j][0]) ** 2 + (extTop_list[i][1] - extBot_list[j][
                    1]) ** 2) ** 0.5)  # calculate the distance between top point to other bottom points
                if top_bot_dist < r_crit:
                    jun_num += 1
                    for k in range(0, len(extLeft_list)):
                        if k != j and k != i:
                            top_left_dist = ((extTop_list[i][0] - extLeft_list[k][0]) ** 2 + (
                                    extTop_list[i][1] - extLeft_list[k][
                                1]) ** 2) ** 0.5  # calculate the distance between top point to other left points
                            if top_left_dist < r_crit:
                                for n in range(0, len(extRight_list)):
                                    if n != k and n != j and n != i:
                                        top_right_dist = ((extTop_list[i][0] - extRight_list[n][0]) ** 2 + (
                                                extTop_list[i][1] - extRight_list[n][
                                            1]) ** 2) ** 0.5  # calculate the distance between top point to other right points
                                        if top_right_dist < r_crit:
                                            junction_list.append(tuple(extTop_list[i]))
                                        else:
                                            n = n + 1
                                    else:
                                        n = n + 1

                            else:
                                k = k + 1
                        else:
                            k = k + 1
                else:
                    j = j + 1
            else:
                j = j + 1
    return junction_list, r_crit, jun_num


def get_parameters_for_drawing():
    """
        Define default BGR colors and line/dot thicknesses for annotations.
        Returns a tuple of (color_left, color_right, … thickness_line).
        """

    global color_left, color_right, color_top, color_bot, color_line_top_bottom, color_line_left_right
    color_left = (0, 135, 255)  # orange
    color_right = (135, 135, 255)  # pink
    color_top = (255, 0, 0)  # blue
    color_bot = (0, 255, 0)  # green
    color_line_top_bottom = (0, 255, 0)  # green
    color_line_left_right = (255, 0, 0)  # blue
    thickness_dot = 1
    thickness_line = 2
    return color_left, color_right, color_top, color_bot, color_line_left_right, color_line_top_bottom, thickness_dot, thickness_line


def draw_circle_for_each_junction(image, junction_list, r_crit):
    """
        Draw red circles at each junction point on a copy of the image.
        """

    image_ = image.copy()
    for junction in junction_list:
        cv2.circle(image_, junction, r_crit, (0, 0, 255), 2)
    #cv2.imshow("junctions", image_)
    #cv2.waitKey(0)
    return image_, image


def draw_contours_in_image(image, morphology_image, height, width, flag_height, flag_width, min_area, max_area):
    """
        Overlay filtered contours on a grayscale BGR image,
        display & save as “contours.png”.
        """

    cnts, _ = cv2.findContours(morphology_image.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    cnts = sorted(cnts, key=lambda x: cv2.contourArea(x))
    MAX_AREA = max_area
    MIN_AREA = min_area
    cntsfiltered = [cnt for cnt in cnts if MIN_AREA < cv2.contourArea(cnt) and MAX_AREA > cv2.contourArea(cnt)]
    area_list, extLeft_list, extRight_list, extTop_list, extBot_list = [], [], [], [], []
    for c in cntsfiltered:
        area = cv2.contourArea(c)
        extLeft = list(c[c[:, :, 0].argmin()][0])
        extRight = list(c[c[:, :, 0].argmax()][0])
        extTop = list(c[c[:, :, 1].argmin()][0])
        extBot = list(c[c[:, :, 1].argmax()][0])
        if flag_height:
            extLeft[0] -= 5
            extRight[0] += 5
            extTop[1] -= 5
            extBot[1] += 5
        elif flag_width:
            extLeft[0] -= 10
            extRight[0] += 10
            extTop[1] -= 10
            extBot[1] += 10
        area_list.append(area)
        extLeft_list.append(extLeft)
        extRight_list.append(extRight)
        extTop_list.append(extTop)
        extBot_list.append(extBot)
    Loss = (abs((height * width) - sum(area_list)) / (height * width) * 100)
    # Draw contours
    img_gray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
    img_gray_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(img_gray_bgr, cntsfiltered, -1, (0, 0, 255), 1)
    cv2.imshow('contours', img_gray_bgr)
    cv2.imwrite("contours.png", img_gray_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return area_list, extLeft_list, extRight_list, extTop_list, extBot_list, Loss


def get_final_image(image_copy, color_left, color_right, color_top, color_bot,
                    color_line_left_right, color_line_top_bottom, thickness_dot, thickness_line, height, width,
                    extTop_list, extBot_list, extLeft_list, extRight_list):
    """
        On the image copy, draw arrowed lines and dots for each
        top/bottom and left/right pair, compute abs & euclidean distances and
        then display & save “Measurements.png”.
        """

    # Final image
    image_copy_ = image_copy.copy()
    w_abs_values, w_euclidean_values = [], []
    for top, bot in zip(extTop_list, extBot_list):
        # Don't include points that are out of image boundaries

        if bot[1] <= height + 2  and top[1] >= -2:
            cv2.circle(image_copy, top, thickness_dot, color_top, -1)
            cv2.circle(image_copy, bot, thickness_dot, color_bot, -1)
            cv2.arrowedLine(image_copy, top, bot, color_line_top_bottom, thickness_line)
            cv2.arrowedLine(image_copy, bot, top, color_line_top_bottom, thickness_line)
            abs_value = [float(abs(top[1] - bot[1]))]
            w_abs_values = np.concatenate((w_abs_values, abs_value), axis=0)
            euclidean_value = [float(np.sqrt((top[1] - bot[1]) ** 2 + (top[0] - bot[0]) ** 2))]
            w_euclidean_values = np.concatenate((w_euclidean_values, euclidean_value), axis=0)
    l_abs_values, l_euclidean_values  = [], []

    for lft, rht in zip(extLeft_list, extRight_list):
        # Don't include points that are out of image boundaries

        if lft[0] >= -2 and rht[0] <= width + 2:
            cv2.circle(image_copy_, lft, thickness_dot, color_left, -1)
            cv2.circle(image_copy_, rht, thickness_dot, color_right, -1)
            cv2.arrowedLine(image_copy, lft, rht, color_line_left_right, thickness_line)
            cv2.arrowedLine(image_copy, rht, lft, color_line_left_right, thickness_line)
            abs_value = [float(abs(rht[0] - lft[0]))]
            l_abs_values = np.concatenate((l_abs_values, abs_value), axis=0)
            euclidean_value = [float(np.sqrt(np.square(lft[1] - rht[1]) + np.square(lft[0] - rht[0])))]
            l_euclidean_values = np.concatenate((l_euclidean_values, euclidean_value), axis=0)

    cv2.imshow('Measurements', image_copy)
    cv2.imwrite('Measurements.png', image_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return l_abs_values, l_euclidean_values, w_abs_values, w_euclidean_values


def get_statistics(args, image_name, size_value, height,
                   l_abs_values, l_euclidean_values, w_abs_values, w_euclidean_values, area_list):
    """
        Convert pixel distances → cm, print summary stats, and
        plot 4 histograms with fitted normal PDFs.
        Save figure as “output_plot_1.png”.
        """

    convert_pixels2cm = size_value / height
    l_abs_values_converted = l_abs_values * convert_pixels2cm
    l_euclidean_values_converted = l_euclidean_values * convert_pixels2cm
    w_abs_values_converted = w_abs_values * convert_pixels2cm
    w_euclidean_values_converted = w_euclidean_values * convert_pixels2cm
    print('--------------------------------------------')
    print_info('image name: %s' % image_name)
    print('--------------------------------------------')
    print_info('Image height in pixels: %.2f' % height)
    print_info('Image height in cm: %.2f' % size_value)
    print_info('Ratio between pixels and cm: 1:%.5f' % convert_pixels2cm)
    l_print_abs = "Length - MEAN distance abs - %.2f cm" % np.mean(l_abs_values_converted)
    l_print_euclidean = "Length - MEAN distance euclidean - %.2f cm" % np.mean(l_euclidean_values_converted)
    w_print_abs = "Width - MEAN distance abs - %.2f cm" % np.mean(w_abs_values_converted)
    w_print_euclidean = "Width - MEAN distance euclidean - %.2f cm" % np.mean(w_euclidean_values_converted)
    print('--------------------------------------------')
    print_info(l_print_abs)
    print_info(l_print_euclidean)
    print('--------------------------------------------')
    print_info(w_print_abs)
    print_info(w_print_euclidean)
    print('--------------------------------------------')
    x_min, x_max = plt.xlim()
    x_min_area, x_max_area = plt.xlim()

    # Create a new figure
    plt.figure(figsize=(12, 8))

    # Subplot 1: Height Abs
    plt.subplot(221)
    l_mu_abs, l_std_abs = norm.fit(l_abs_values_converted)
    x = np.linspace(min(l_abs_values_converted), max(l_abs_values_converted), 1000)
    l_pdf_abs = norm.pdf(x, l_mu_abs, l_std_abs)
    plt.hist(l_abs_values_converted, bins=50, density=True, alpha=0.6, color='g')
    plt.plot(x, l_pdf_abs, 'k', linewidth=2)
    plt.title(f"Length Abs: μ={l_mu_abs:.2f}, σ={l_std_abs:.2f}")
    plt.xlabel("Length [cm]")
    plt.ylabel("Probability Density")
    plt.grid()
    plt.legend(['PDF', 'Histogram'])

    # Subplot 2: Height Euclidean
    plt.subplot(222)
    l_mu_euclidean, l_std_euclidean = norm.fit(l_euclidean_values_converted)
    x = np.linspace(min(l_euclidean_values_converted), max(l_euclidean_values_converted), 1000)
    l_pdf_euclidean = norm.pdf(x, l_mu_euclidean, l_std_euclidean)
    plt.hist(l_euclidean_values_converted, bins=50, density=True, alpha=0.6, color='g')
    plt.plot(x, l_pdf_euclidean, 'k', linewidth=2)
    plt.title(f"Length Euclidean: μ={l_mu_euclidean:.2f}, σ={l_std_euclidean:.2f}")
    plt.xlabel("Length [cm]")
    plt.ylabel("Probability Density")
    plt.grid()
    plt.legend(['PDF', 'Histogram'])

    # Subplot 3: Width Abs
    plt.subplot(223)
    w_mu_abs, w_std_abs = norm.fit(w_abs_values_converted)
    x = np.linspace(min(w_abs_values_converted), max(w_abs_values_converted), 1000)
    w_pdf_abs = norm.pdf(x, w_mu_abs, w_std_abs)
    plt.hist(w_abs_values_converted, bins=50, density=True, alpha=0.6, color='g')
    plt.plot(x, w_pdf_abs, 'k', linewidth=2)
    plt.title(f"Width Abs: μ={w_mu_abs:.2f}, σ={w_std_abs:.2f}")
    plt.xlabel("Width [cm]")
    plt.ylabel("Probability Density")
    plt.grid()
    plt.legend(['PDF', 'Histogram'])

    # Subplot 4: Width Euclidean
    plt.subplot(224)
    w_mu_euclidean, w_std_euclidean = norm.fit(w_euclidean_values_converted)
    x = np.linspace(min(w_euclidean_values_converted), max(w_euclidean_values_converted), 1000)
    w_pdf_euclidean = norm.pdf(x, w_mu_euclidean, w_std_euclidean)
    plt.hist(w_euclidean_values_converted, bins=50, density=True, alpha=0.6, color='g')
    plt.plot(x, w_pdf_euclidean, 'k', linewidth=2)
    plt.title(f"Width Euclidean: μ={w_mu_euclidean:.2f}, σ={w_std_euclidean:.2f}")
    plt.xlabel("Width [cm]")
    plt.ylabel("Probability Density")
    plt.grid()
    plt.legend(['PDF', 'Histogram'])

    # Optimize spacing
    plt.tight_layout()
    plt.savefig("output_plot_1.png", dpi=300)
    plt.show()
    plt.close('all')
    return x_min, x_max, x_min_area, x_max_area, l_abs_values_converted, l_euclidean_values_converted, w_abs_values_converted, w_euclidean_values_converted, area_list


def measure_image(image_path, dimension, size, step=10, debug=0, min_area_input=40, divisor_of_max_area_input=8, verbosity=2 ):
    """
    Process a single image and produce measurement graphs and an annotated final image.

    Parameters:
      image_path: Path to the image file.
      dimension: 'h' (height) or 'w' (width) to indicate the measurement basis.
      size: Physical size (in cm) corresponding to the selected dimension.
      debug: Set to 1 to show intermediate images.
      min_area_input: Fallback minimum area value (default: 40).
      divisor_of_max_area_input: Divisor used to compute the maximum area (default: 8).

    Returns:
      A tuple (annotated_image, measurements) where:
        - annotated_image: The final image with drawn measurements.
        - measurements: A dictionary with measurement arrays.
    """
    start_time = time.time()
    flag_height, flag_width = get_dimension_flags(dimension)
    image, image_copy = get_rotating_image(image_path, flag_height, flag_width)
    height, width, channels = get_image_shape(image)
    if verbosity>0:
        print("Got image shape")
    morphology_image = morphological_image_processing(image, debug)
    if verbosity>0:
        print("Finished morphology")
    (color_left, color_right, color_top, color_bot,
     color_line_left_right, color_line_top_bottom,
     thickness_dot, thickness_line) = get_parameters_for_drawing()
    if verbosity>0:
        print("Moving out of photo pull and manipulation")

    # Optimization loop for minimum area
    Loss_list = []
    jun_percent_min = []
    min_area_candidates = []
    max_area_candidate = int(height * width / divisor_of_max_area_input)
    pbar = tqdm(range(0, max_area_candidate, step), desc="Min. Area Optimization")
    for m in pbar:
        if verbosity>3:
            print(f"\tProgress bar iteration:\t{m}")
        area_list, extLeft_list, extRight_list, extTop_list, extBot_list, Loss = detect_contours_in_image(
            image, morphology_image, height, width, flag_height, flag_width, m, max_area_candidate)
        junction_list, r_crit, jun_num = is_junction(extLeft_list, extRight_list,
                                                     extTop_list, extBot_list, area_list, m)

        Loss_list.append(100 - Loss)
        min_area_candidates.append(m)

        if (100 - Loss) == 0:
            pbar.n = pbar.total
            pbar.refresh()
            pbar.close()
            break

        jun_percent_min.append(len(junction_list))
    if len(jun_percent_min) > 0:
        max_jun_percent_ind = jun_percent_min.index(max(jun_percent_min))
        opt_min_area = min_area_candidates[max_jun_percent_ind]
    else:
        opt_min_area = min_area_input

    # Optimization loop for maximum area
    Loss_list = []
    jun_percent_max = []
    max_area_candidates = []

    step_max = -abs(step) * 300
    pbar = tqdm(range(max_area_candidate, opt_min_area, step_max),  # ↓
                desc="Max. Area Optimization")

    for M in pbar:
        area_list, extLeft_list, extRight_list, extTop_list, extBot_list, Loss = detect_contours_in_image(
            image, morphology_image, height, width, flag_height, flag_width,
            opt_min_area, M)

        junction_list, r_crit, jun_num = is_junction(
            extLeft_list, extRight_list, extTop_list, extBot_list, area_list,
            opt_min_area)

        Loss_list.append(100 - Loss)
        max_area_candidates.append(M)

        if (100 - Loss) == 0:
            pbar.n = pbar.total
            pbar.refresh()
            pbar.close()
            break

        jun_percent_max.append(len(junction_list))
    if len(jun_percent_max) > 0:
        max_jun_percent_ind = jun_percent_max.index(max(jun_percent_max))
        opt_max_area = max_area_candidates[max_jun_percent_ind]
    else:
        opt_max_area = max_area_candidate

    end = time.time()
    runtime = end - start_time
    # print_info(f'Optimal maximum area: {opt_max_area} pixels')
    # print_info(f'Optimal minimum area: {opt_min_area} pixels')
    # print_info(f'Percent of junctions: {max(jun_percent_max) if jun_percent_max else 0}' + '%')
    # print_info(f'r_crit: {r_crit} pixels')
    # print_info(f'Runtime: {runtime:.2f} seconds')

    # Draw junctions using the optimal parameters
    if verbosity>0:
        print("Starting contour detection")
    area_list, extLeft_list, extRight_list, extTop_list, extBot_list, Loss = detect_contours_in_image(
        image, morphology_image, height, width, flag_height, flag_width, opt_min_area, opt_max_area)
    junction_list, r_crit, jun_num = is_junction(extLeft_list, extRight_list, extTop_list, extBot_list, area_list,
                                                 opt_min_area)
    image_annotated, _ = draw_circle_for_each_junction(image, junction_list, int(r_crit))
    if verbosity>0:
        print("Contour detection complete")

    # Draw contours using the optimal parameters
    draw_contours_in_image(image, morphology_image, height, width, flag_height, flag_width, opt_min_area, opt_max_area)

    # Get final measurement values and draw arrows, etc.
    if verbosity>0:
        print("Drawing final image")
    l_abs_values, l_euclidean_values, w_abs_values, w_euclidean_values = get_final_image(
        image_copy, color_left, color_right, color_top, color_bot,
        color_line_left_right, color_line_top_bottom, thickness_dot - 1, thickness_line, height, width,
        extTop_list, extBot_list, extLeft_list, extRight_list)

    # Create a dummy args object for get_statistics (for printing purposes)
    class DummyArgs:
        input = [image_path]

    dummy_args = DummyArgs()
    (x_min, x_max, x_min_area, x_max_area,
     l_abs_values_converted, l_euclidean_values_converted,
     w_abs_values_converted, w_euclidean_values_converted,
     area_list) = get_statistics(dummy_args, image_path, size, height,
                                 l_abs_values, l_euclidean_values, w_abs_values, w_euclidean_values, area_list)

    measurements = {
        'l_abs_values_converted': l_abs_values_converted,
        'l_euclidean_values_converted': l_euclidean_values_converted,
        'w_abs_values_converted': w_abs_values_converted,
        'w_euclidean_values_converted': w_euclidean_values_converted,
    }

    # Save Data
    max_rows = max(len(val) for val in measurements.values())
    with open('measurements.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        # Write header row from dictionary keys
        headers = list(measurements.keys())
        writer.writerow(headers)
        # Write each row, using an empty string for missing entries
        for i in range(max_rows):
            row = [measurements[key][i] if i < len(measurements[key]) else '' for key in headers]
            writer.writerow(row)

    return image_annotated, measurements


# ----------------------- Main Block -----------------------

"""
if __name__ == '__main__':
    # the measurement dimension ('h' for height or 'w' for width), and the physical size.
    image_path = 'path_to_your_image.jpg'
    dimension = 'h'  # Use 'h' for height or 'w' for width
    size = 10.0  # Example: image height is 10 cm
    annotated_image, measurements = measure_image(image_path, dimension, size, debug=0)
    # Display the final annotated image
    cv2.imshow("Annotated Image", annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # Save Data
    max_rows = max(len(val) for val in measurements.values())
    with open('measurements.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        headers = list(measurements.keys())
        writer.writerow(headers)
        for i in range(max_rows):
            row = [measurements[key][i] if i < len(measurements[key]) else '' for key in headers]
            writer.writerow(row)
"""
