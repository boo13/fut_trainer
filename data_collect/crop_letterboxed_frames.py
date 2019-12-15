# For the file dir walking code, thanks to https://github.com/jrosebr1/imutils
import os

image_types = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def list_images(basePath, contains=None):
    # return the set of files that are valid
    return list_files(basePath, validExts=image_types, contains=contains)


def list_files(basePath, validExts=None, contains=None):
    # loop over the directory structure
    for (rootDir, dirNames, filenames) in os.walk(basePath):
        # loop over the filenames in the current directory
        for filename in filenames:
            # if the contains string is not none and the filename does not contain
            # the supplied string, then ignore the file
            if contains is not None and filename.find(contains) == -1:
                continue

            # determine the file extension of the current file
            ext = filename[filename.rfind(".") :].lower()

            # check to see if the file is an image and should be processed
            if validExts is None or ext.endswith(validExts):
                # construct the path to the image and yield it
                imagePath = os.path.join(rootDir, filename)
                yield imagePath


if __name__ == "__main__":

    import cv2

    image_counter = 0
    resized_images = 0

    im_list = list(list_images("data"))

    for f in im_list:
        img = cv2.imread(f)

        h, w, channels = img.shape

        if w == 640 and h == 480:
            img = img[60:420, 0:w]
            cv2.imwrite(f, img)
            image_counter += 1
        else:
            resized_images += 1

    print(f"Images of 640 x 480: {image_counter}")
    print(f"Images of other sizes: {resized_images}")
