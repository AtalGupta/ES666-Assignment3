import pdb
import glob
import cv2
import os
import numpy as np

class PanaromaStitcher():
    def __init__(self):
        pass

    def make_panaroma_for_images_in(self, path):
        imf = path
        all_images = sorted(glob.glob(imf + os.sep + '*'))
        print('Found {} Images for stitching'.format(len(all_images)))

        if len(all_images) < 2:
            raise ValueError("Need at least two images to stitch a panorama")

        # Read all images
        images = [cv2.imread(img) for img in all_images]

        # Initialize SIFT detector
        sift = cv2.SIFT_create()

        # List to store homography matrices
        homography_matrix_list = []

        # Initialize the final stitched image as the first image
        stitched_image = images[0]

        for i in range(1, len(images)):
            # Detect keypoints and descriptors
            kp1, des1 = sift.detectAndCompute(stitched_image, None)
            kp2, des2 = sift.detectAndCompute(images[i], None)

            # Match descriptors using FLANN matcher
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(des1, des2, k=2)

            # Store good matches using Lowe's ratio test
            good_matches = []
            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)

            # Extract location of good matches
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            # Compute homography
            H, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
            homography_matrix_list.append(H)

            # Warp the next image to the current stitched image
            stitched_image = cv2.warpPerspective(images[i], H, (stitched_image.shape[1] + images[i].shape[1], stitched_image.shape[0]))
            stitched_image[0:images[i].shape[0], 0:images[i].shape[1]] = images[i]

        return stitched_image, homography_matrix_list

    def say_hi(self):
        raise NotImplementedError('I am an Error. Fix Me Please!')