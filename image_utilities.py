import numpy as np
import os
import cv2
import errno
import shutil
from PIL import Image


class ImageUtilities:

    @staticmethod
    def make_training_data(image_directory="101_ObjectCategories"):
        """
        Creates two folders, inputs and outputs, that hold grey images (300x300)
        output images also contain white noise
        function takes roughly ~5 min to run
        """

        BASE_DIRECTORY = "/Users/apostolos/Documents/UCLA/Year 1 Q2/EE 194/Image DeNoising/"

        # Get the Images out of the base directory 101_Object Categories and make it greyscale
        # Make a new Temp directory named grays and store the temporary gray images
        shutil.rmtree("inputs", ignore_errors=True)
        shutil.rmtree("outputs", ignore_errors=True)
        ImageUtilities.convert_image_directory_to_greyscale(os.path.join(BASE_DIRECTORY, image_directory),
                                                            new_directory="grays")

        # add noise to the "grays"
        ImageUtilities.add_gaussian_white_noise(
            os.path.join(BASE_DIRECTORY, "grays"),
            noisy_directory_name="noisies",
            variance=10, delete_previous=True)

        # remake the input and output directories
        input_directory = os.path.join(BASE_DIRECTORY, "inputs")
        output_directory = os.path.join(BASE_DIRECTORY, "outputs")

        shutil.rmtree(input_directory, ignore_errors=True)
        shutil.rmtree(output_directory, ignore_errors=True)

        ImageUtilities.make_directory(input_directory)
        ImageUtilities.make_directory(output_directory)

        # Resize images in the grays and noisies category
        grays_dir = os.path.join(BASE_DIRECTORY, "grays")
        for filename in os.listdir(grays_dir):
            if filename.endswith(".png") or filename.endswith(".jpg"):
                filename = os.path.join(grays_dir, filename)
                ImageUtilities.resize_image(image_file=filename, directory=output_directory)

        shutil.rmtree(grays_dir, ignore_errors=True)

        noise_dir = os.path.join(BASE_DIRECTORY, "noisies")
        for filename in os.listdir(noise_dir):
            if filename.endswith(".png") or filename.endswith(".jpg"):
                filename = os.path.join(noise_dir, filename)
                ImageUtilities.resize_image(image_file=filename, directory=input_directory)

        shutil.rmtree(noise_dir, ignore_errors=True)

    @staticmethod
    def convert_to_uint8(image_in: np.array):
        """
        Convert an image array to unsigned 8bit numbers
        :param image_in: numpy array
        :return: image as type uint8
        """
        temp_image = np.float64(np.copy(image_in))
        cv2.normalize(temp_image, temp_image, 0, 255, cv2.NORM_MINMAX, dtype=-1)

        return temp_image.astype(np.uint8)

    @staticmethod
    def display_image(image_file: str, greyscale=False):
        """
        :param image_file: the file path of the image that will be loaded
        :param greyscale: boolean for if you want color with the image or not
        """

        display_format = {
            False: cv2.WINDOW_AUTOSIZE,
            True: cv2.IMREAD_GRAYSCALE
        }

        image = cv2.imread(image_file, display_format[greyscale])
        image_name = image_file.split('/')[-1]
        cv2.imshow(image_name, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    @staticmethod
    def test_image_utils():
        """
        test the various functions that are used in ImageUtilities
        """
        # ImageUtilities.add_gaussian_white_noise(
        #     "/Users/apostolos/Documents/UCLA/Year 1 Q2/EE 194/Image DeNoising/grey_images",
        #     variance=10, delete_previous=True)
        #
        # image_array = ImageUtilities.image_to_nparray(
        #     "/Users/apostolos/Documents/UCLA/Year 1 Q2/EE 194/Image DeNoising/grey_images/grey_image_0010.jpg")
        # image_array = ImageUtilities.__add_gaussian_noise(image_array, noise_sigma=100, display_noise=True)
        ImageUtilities.make_training_data()

    @staticmethod
    def add_gaussian_white_noise(directory: str, variance=50, noisy_directory_name="noisy_images", delete_previous=False):
        """
        Add gaussian white noise to all the images in directory and then save them in a new folder
        :param directory: the directory with all the images
        :param variance: the amount of noise in the image (mess around with multiple values until one works)
        :param noisy_directory_name: the name of the new directory that will be created to hold the noisy images
        :param delete_previous: delete the previous noisy image directory if it exists
        """

        DIRECTORY_PATH = "/Users/apostolos/Documents/UCLA/Year 1 Q2/EE 194/Image DeNoising"

        noisy_image_directory = os.path.join(DIRECTORY_PATH, noisy_directory_name)

        # delete the existing noisy directory
        if delete_previous:
            shutil.rmtree(noisy_image_directory, ignore_errors=True)

        # Make the directory
        ImageUtilities.make_directory(noisy_image_directory)

        test_filename = ""
        for filename in os.listdir(directory):
            if filename.endswith(".png") or filename.endswith(".jpg"):
                print(filename)
                test_filename = filename
                image_array = ImageUtilities.image_to_nparray(os.path.join(directory, filename))
                image_array = ImageUtilities.__add_gaussian_noise(image_array, noise_sigma=variance)
                image_array = ImageUtilities.convert_to_uint8(image_array)
                im = Image.fromarray(image_array)
                im.save(os.path.join(noisy_image_directory, "noisy_" + filename), "PNG")
                print("image", filename, "saved")

        im1 = Image.open(os.path.join(directory, test_filename))
        im1.show()
        image_array = ImageUtilities.image_to_nparray(os.path.join(directory, test_filename))
        image_array = ImageUtilities.__add_gaussian_noise(image_array, noise_sigma=variance, display_noise=True)
        im = Image.fromarray(image_array)
        im.show()
        print("==== Gaussian White Noise Added ====")

    @staticmethod
    def image_to_nparray(file_of_image: str)->np.array:
        """
        converts an image to a numpy array
        :param file_of_image: file of image
        :return: numpy array of image
        """
        im = Image.open(file_of_image).convert('L')
        return np.array(im)

    @staticmethod
    def resize_image(image_file: str, directory: str, width=300, height=300, resize_type="ANTIALIAS"):
        """
        resizes the image
        :param image_file: the image's path
        :param directory: the directory to save the new image
        :param width: the image's new width
        :param height: the image's new Height
        :param resize_type: what type of resize to use:
                Available types:
                NEAREST: use nearest neighbour
                BILINEAR: use linear interpolation in a 2x2 environment
                BICUBIC: cubic spline interpolation in a 4x4 environment
                ANTIALIAS: best down-sizing filter
        """

        im1 = Image.open(image_file)
        ImageUtilities.make_directory(directory)

        # use one of these filter options to resize the image
        if resize_type.upper() == "NEAREST":
            im2 = im1.resize((width, height), Image.NEAREST)  # use nearest neighbour
        elif resize_type.upper() == "BILINEAR":
            im2 = im1.resize((width, height), Image.BILINEAR)  # linear interpolation in a 2x2 environment
        elif resize_type.upper() == "BICUBIC":
            im2 = im1.resize((width, height), Image.BICUBIC)  # cubic spline interpolation in a 4x4 environment
        else:
            im2 = im1.resize((width, height), Image.ANTIALIAS)  # best down-sizing filter

        image_name = image_file.split('/')[-1]
        im2.save(os.path.join(directory, image_name))

    @staticmethod
    def convert_image_directory_to_greyscale(directory: str, new_directory: str):
        """
        :param new_directory: where to store the new images
        :param directory: directory of directories 
        """

        DIRECTORY_PATH = "/Users/apostolos/Documents/UCLA/Year 1 Q2/EE 194/Image DeNoising"
        bad_files = {".DS_Store"}

        ImageUtilities.make_directory(os.path.join(DIRECTORY_PATH, new_directory))

        for dir in os.listdir(directory):
            if os.path.isdir(dir):
                # print the current directory being processed
                print(os.path.join(directory, dir))

                # if it is an image directory
                if dir not in bad_files:
                    temp_directory = os.path.join(directory, dir)
                    for filename in os.listdir(temp_directory):
                        if (filename.endswith(".png") or filename.endswith(".jpg")) and filename not in bad_files:
                            image_array = ImageUtilities.image_to_nparray(os.path.join(temp_directory, filename))
                            image_array = ImageUtilities.__add_gaussian_noise(image_array, noise_sigma=0)
                            image_array = ImageUtilities.convert_to_uint8(image_array)
                            im = Image.fromarray(image_array)

                            # if decided to use a different method of greyscale
                            # color_image_path = os.path.join(temp_directory, filename)
                            # img = Image.open(color_image_path).convert('L')
                            # img.save(os.path.join(os.path.join(DIRECTORY_PATH, new_directory), filename))

                            print("image", filename, "converted")
                            im.save(os.path.join(os.path.join(DIRECTORY_PATH, new_directory), filename), "PNG")

            # if it is just an image
            elif (dir.endswith(".png") or dir.endswith(".jpg")) and dir not in bad_files:
                    filename = os.path.join(directory, dir)
                    image_array = ImageUtilities.image_to_nparray(filename)
                    image_array = ImageUtilities.__add_gaussian_noise(image_array, noise_sigma=0)
                    image_array = ImageUtilities.convert_to_uint8(image_array)
                    im = Image.fromarray(image_array)

                    # if decided to use a different method of greyscale
                    # color_image_path = os.path.join(temp_directory, filename)
                    # img = Image.open(color_image_path).convert('L')
                    # img.save(os.path.join(os.path.join(DIRECTORY_PATH, new_directory), filename))

                    print("image", filename, "converted")
                    im.save(os.path.join(os.path.join(DIRECTORY_PATH, new_directory), filename), "PNG")

        print("==== Images Converted to Greyscale ====")

    @staticmethod
    def make_directory(file_path: str):
        """
        Creates the directory at the path:
        :param file_path: the path of the directory that you want ot create
        """
        if file_path == "\\":
            return 0
        try:
            os.makedirs(file_path, exist_ok=True)
        except OSError as exc:
            if exc.errno == errno.EEXIST and os.path.isdir(file_path):
                pass
            else:
                print("Error while attempting to create a directory.")
                exit(3)

    @staticmethod
    def __add_gaussian_noise(image_in: np.array, noise_sigma: int, display_noise=False)->np.array:
        """
        :param image_in: a image numpy array
        :param noise_sigma: the amount of noise to add to the image
        :param display_noise: display the noise image
        :return: the noisy image array
        """
        temp_image = np.float64(np.copy(image_in))

        h = temp_image.shape[0]
        w = temp_image.shape[1]
        noise = np.random.randn(h, w) * noise_sigma

        if display_noise:
            im = ImageUtilities.convert_to_uint8(noise)
            im = Image.fromarray(im)
            im.show()

        noisy_image = np.zeros(temp_image.shape, np.float64)
        if len(temp_image.shape) == 2:
            noisy_image = temp_image + noise
        else:
            noisy_image[:, :, 0] = temp_image[:, :, 0] + noise
            noisy_image[:, :, 1] = temp_image[:, :, 1] + noise
            noisy_image[:, :, 2] = temp_image[:, :, 2] + noise

        return noisy_image


if __name__ == "__main__":
    ImageUtilities.test_image_utils()
