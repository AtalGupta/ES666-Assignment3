import numpy as np
import cv2
import os

class PanaromaStitcher:
    def __init__(self):
        self.sift = cv2.SIFT_create(nfeatures=1000)  # Further reduced features
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)
        self.max_image_dim = 800  # Maximum dimension for any image

    def resize_image(self, img):
        """Resize image while maintaining aspect ratio"""
        if img is None:
            return None
        h, w = img.shape[:2]
        if max(h, w) > self.max_image_dim:
            scale = self.max_image_dim / max(h, w)
            new_size = (int(w * scale), int(h * scale))
            return cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
        return img

    def detect_and_match_features(self, img1, img2):
        """Detect and match features between two images"""
        try:
            # Convert images to grayscale
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            
            # Detect keypoints and compute descriptors
            kp1, des1 = self.sift.detectAndCompute(gray1, None)
            kp2, des2 = self.sift.detectAndCompute(gray2, None)
            
            if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
                return None, None, []
            
            matches = self.flann.knnMatch(des1, des2, k=2)
            good_matches = []
            
            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
            
            # Limit number of matches to reduce memory usage
            return kp1, kp2, good_matches[:50]
            
        except Exception as e:
            print(f"Feature matching error: {str(e)}")
            return None, None, []

    def estimate_homography(self, kp1, kp2, matches):
        """Estimate homography matrix using RANSAC"""
        if len(matches) < 4:
            return None
        
        try:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            
            best_H = None
            best_inliers = 0
            threshold = 5.0
            
            for _ in range(300):
                idx = np.random.choice(min(len(matches), 20), 4, replace=False)
                H = self.compute_homography(src_pts[idx], dst_pts[idx])
                
                if H is None:
                    continue
                
                # Quick inlier check
                inliers = 0
                for i in range(min(len(matches), 20)):
                    pt1 = src_pts[i]
                    pt2 = dst_pts[i]
                    pt1_transformed = cv2.perspectiveTransform(pt1.reshape(-1, 1, 2), H)
                    error = np.linalg.norm(pt1_transformed - pt2)
                    if error < threshold:
                        inliers += 1
                
                if inliers > best_inliers:
                    best_inliers = inliers
                    best_H = H
            
            return best_H
            
        except Exception as e:
            print(f"Homography estimation error: {str(e)}")
            return None

    def compute_homography(self, src_pts, dst_pts):
        """Compute homography matrix for 4 point correspondences"""
        try:
            A = np.zeros((8, 9), dtype=np.float32)  
            for i in range(4):
                x, y = src_pts[i][0]
                u, v = dst_pts[i][0]
                A[i*2] = [-x, -y, -1, 0, 0, 0, x*u, y*u, u]
                A[i*2+1] = [0, 0, 0, -x, -y, -1, x*v, y*v, v]
            
            _, _, Vt = np.linalg.svd(A)
            H = Vt[-1].reshape(3, 3)
            
            if abs(H[2, 2]) < 1e-8:
                return None
                
            return H / H[2, 2]
            
        except Exception:
            return None

    def warp_images(self, img1, img2, H):
        """Warp img1 onto img2 using homography matrix H"""
        try:
            h1, w1 = img1.shape[:2]
            h2, w2 = img2.shape[:2]
            
            corners1 = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2)
            corners2 = cv2.perspectiveTransform(corners1, H)
            
            all_corners = np.concatenate((
                corners2,
                np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)
            ))
            
            xmin, ymin = np.int32(all_corners.min(axis=0).ravel() - 0.5)
            xmax, ymax = np.int32(all_corners.max(axis=0).ravel() + 0.5)
            
            # Limit output size
            width = min(xmax - xmin, 5000)
            height = min(ymax - ymin, 5000)
            
            t = [-xmin, -ymin]
            Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]], dtype=np.float32)
            
            try:
                result = cv2.warpPerspective(img1, Ht.dot(H), (width, height))
                if t[1] < height and t[0] < width:
                    result[t[1]:min(t[1]+h2, height), t[0]:min(t[0]+w2, width)] = img2[
                        0:min(h2, height-t[1]),
                        0:min(w2, width-t[0])
                    ]
                return result
            except Exception as e:
                print(f"Warping error: {str(e)}")
                return img2
                
        except Exception as e:
            print(f"Warping preparation error: {str(e)}")
            return img2

    def make_panaroma_for_images_in(self, *, path):
        """
        Create panorama from images in the specified folder
        Returns: (stitched_image, list_of_homography_matrices)
        """
        try:
            if not os.path.isdir(path):
                return None, []
                
            image_files = sorted([
                os.path.join(path, f) 
                for f in os.listdir(path) 
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ])
            
            if not image_files:
                return None, []
            
            # Read and resize images
            images = []
            for img_path in image_files:
                try:
                    img = cv2.imread(img_path)
                    if img is not None:
                        img = self.resize_image(img)
                        if img is not None:
                            images.append(img)
                except Exception as e:
                    print(f"Error reading image {img_path}: {str(e)}")
                    continue
            
            if len(images) < 2:
                return images[0] if len(images) == 1 else None, []
            
            result = images[0]
            homography_matrices = []
            
            for i in range(1, len(images)):
                try:
                    kp1, kp2, matches = self.detect_and_match_features(result, images[i])
                    
                    if kp1 is None or kp2 is None or not matches:
                        continue
                    
                    H = self.estimate_homography(kp1, kp2, matches)
                    if H is None:
                        continue
                    
                    homography_matrices.append(H)
                    
                    new_result = self.warp_images(result, images[i], H)
                    if new_result is not None:
                        result = new_result
                    
                except Exception as e:
                    print(f"Error processing image pair {i}: {str(e)}")
                    continue
            
            # Ensure final image size is reasonable
            if result is not None and (result.shape[0] > 5000 or result.shape[1] > 5000):
                scale = min(5000 / result.shape[0], 5000 / result.shape[1])
                result = cv2.resize(result, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            
            return result, homography_matrices
            
        except Exception as e:
            print(f"Error in make_panaroma_for_images_in: {str(e)}")
            return None, []