import cv2
import numpy as np
import matplotlib.pyplot as plt


def select_roi(image):
    """Select ROI on an image and return the cropped ROI and its coordinates."""
    roi = cv2.selectROI("Select ROI", image)
    cv2.destroyWindow("Select ROI")
    cropped = image[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])]
    return cropped, roi


def compute_sift_keypoints_and_descriptors(image):
    """Compute SIFT keypoints and descriptors for a given image."""
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints, descriptors


def match_features(des1, des2):
    """Match features between two sets of descriptors using FLANN."""
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
    return good_matches


def find_homography_and_classify_points(kp1, kp2, matches, roi2, im2_shape):
    """Find homography matrix and classify points as inliers or outliers."""
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Compute homography using RANSAC
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    mask = mask.ravel()  # Flatten mask

    # Transform points using homography
    transformed_pts = cv2.perspectiveTransform(src_pts, M)

    # Classify points
    inliers = [(int(pt[0][0] + roi2[0]), int(pt[0][1] + roi2[1])) for pt, m in zip(transformed_pts, mask) if m]
    outliers = [(int(pt[0][0] + roi2[0]), int(pt[0][1] + roi2[1])) for pt, m in zip(transformed_pts, mask) if not m]

    return inliers, outliers


def draw_points_on_image(image, inliers, outliers):
    """Draw inliers and outliers on the image."""
    image_copy = image.copy()
    for pt in inliers:
        cv2.circle(image_copy, pt, 5, (255, 0, 0), -1)  # Inliers: Blue
    for pt in outliers:
        cv2.circle(image_copy, pt, 5, (0, 0, 255), -1)  # Outliers: Red
    return image_copy


def display_image_with_matches(image, inliers, outliers):
    """Display the image with inliers and outliers."""
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(f'Inliers: {len(inliers)}, Outliers: {len(outliers)}')
    plt.axis('off')
    plt.show()


def feature_matching_with_ransac(im1_path, im2_path):
    """Perform feature matching between two images with RANSAC."""
    # Load images
    im1 = cv2.imread(im1_path)
    im2 = cv2.imread(im2_path)

    # Select ROIs
    roi1_cropped, roi1 = select_roi(im1)
    roi2_cropped, roi2 = select_roi(im2)

    # Convert to grayscale
    gray1_roi = cv2.cvtColor(roi1_cropped, cv2.COLOR_BGR2GRAY)
    gray2_roi = cv2.cvtColor(roi2_cropped, cv2.COLOR_BGR2GRAY)

    # Compute SIFT keypoints and descriptors
    kp1, des1 = compute_sift_keypoints_and_descriptors(gray1_roi)
    kp2, des2 = compute_sift_keypoints_and_descriptors(gray2_roi)

    # Match features
    matches = match_features(des1, des2)

    # Find homography and classify points
    inliers, outliers = find_homography_and_classify_points(kp1, kp2, matches, roi2, im2.shape)

    # Draw inliers and outliers
    im2_with_matches = draw_points_on_image(im2, inliers, outliers)

    # Display the results
    display_image_with_matches(im2_with_matches, inliers, outliers)


# Call the function with paths to the images
feature_matching_with_ransac('im0.png', 'im1.png')