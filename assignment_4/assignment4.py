import cv2
import numpy as np
import os


def harris_corner_detection(reference_image):
    """
    Detect Harris corners in the reference image
    and save the result as harris.png.
    """

    # Load the reference image
    image = cv2.imread(reference_image)

    if image is None:
        raise FileNotFoundError(
            f"Could not load reference image: {reference_image}"
        )

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Harris needs a float32 image
    gray = np.float32(gray)

    # Find corners in the image
    corners = cv2.cornerHarris(
        gray,
        blockSize=2,
        ksize=3,
        k=0.04
    )

    # Make the corners easier to see
    corners = cv2.dilate(corners, None)

    # Mark strong corners in red
    threshold = 0.01 * corners.max()
    image[corners > threshold] = [0, 0, 255]

    # Save the result
    output_file = "harris.png"

    if not cv2.imwrite(output_file, image):
        raise RuntimeError(
            f"Could not save {output_file}"
        )

    print(f"Created: {output_file}")


def align_images(
    image_to_align,
    reference_image,
    max_features,
    good_match_precent
):
    """
    Align image_to_align with reference_image
    using SIFT, FLANN and homography.
    """

    # Load both images
    image = cv2.imread(image_to_align)
    reference = cv2.imread(reference_image)

    if image is None:
        raise FileNotFoundError(
            f"Could not load image to align: {image_to_align}"
        )

    if reference is None:
        raise FileNotFoundError(
            f"Could not load reference image: {reference_image}"
        )

    # Convert both images to grayscale
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    reference_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)

    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Find keypoints and their descriptors
    keypoints_image, descriptors_image = sift.detectAndCompute(
        image_gray,
        None
    )

    keypoints_reference, descriptors_reference = sift.detectAndCompute(
        reference_gray,
        None
    )

    if descriptors_image is None or descriptors_reference is None:
        raise RuntimeError("SIFT could not find features in one of the images.")

    print(f"SIFT features in image_to_align: {len(keypoints_image)}")
    print(f"SIFT features in reference_image: {len(keypoints_reference)}")

    # Set up FLANN for feature matching
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Find the two closest matches for each feature
    matches = flann.knnMatch(
        descriptors_image,
        descriptors_reference,
        k=2
    )

    # Keep only matches that pass Lowe's ratio test (good_match_precent = 0.7)
    good_matches = []
    for match_pair in matches:
        if len(match_pair) < 2:
            continue
        first_match = match_pair[0]
        second_match = match_pair[1]
        if first_match.distance < (good_match_precent * second_match.distance):
            good_matches.append(first_match)

    # Sort matches by distance (best matches first)
    good_matches = sorted(good_matches, key=lambda x: x.distance)

    # Limit to max_features (max_features = 10)
    if max_features > 0 and len(good_matches) > max_features:
        good_matches = good_matches[:max_features]

    print(f"Good matches used: {len(good_matches)}")

    if len(good_matches) < 4:
        raise RuntimeError("Not enough good matches were found.")

    # Get coordinates of matched points
    src_pts = np.float32([
        keypoints_image[m.queryIdx].pt for m in good_matches
    ]).reshape(-1, 1, 2)

    dst_pts = np.float32([
        keypoints_reference[m.trainIdx].pt for m in good_matches
    ]).reshape(-1, 1, 2)

    # Find homography
    homography, mask = cv2.findHomography(
        src_pts,
        dst_pts,
        cv2.RANSAC,
        5.0
    )

    if homography is None:
        raise RuntimeError("Could not calculate homography.")

    # Transform image
    height, width = reference.shape[:2]
    aligned = cv2.warpPerspective(image, homography, (width, height))

    # Filter to draw ONLY inlier matches
    inlier_matches = [
        good_matches[i] for i in range(len(good_matches)) if mask[i][0] == 1
    ]

    matches_image = cv2.drawMatches(
        image,
        keypoints_image,
        reference,
        keypoints_reference,
        inlier_matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    # Save output
    cv2.imwrite("aligned.png", aligned)
    cv2.imwrite("matches.png", matches_image)

    print("Created: aligned.png")
    print("Created: matches.png")


def main():
    reference_image = "reference_img.png"
    image_to_align = "align_this.jpg"

    if not os.path.exists(reference_image) or not os.path.exists(image_to_align):
        raise FileNotFoundError("Input images not found.")

    print("Part 1: Harris Corner Detection")
    harris_corner_detection(reference_image)

    print("\nPart 2: Feature-Based Image Alignment")
    print("max_features = 10")
    print("good_match_precent = 0.7\n")

    # Call with max_features=10 and good_match_precent=0.7 as requested
    align_images(
        image_to_align,
        reference_image,
        max_features=10,
        good_match_precent=0.7
    )


if __name__ == "__main__":
    main()