import cv2
import numpy as np

def main():

    # Last inn bildet
    image = cv2.imread('iris-1.png')

    if image is None:
        print("Feil: Fant ikke 'iris-1.png'. Sjekk filnavn og sti.")
        return

    # Finn størrelse
    height, width = image.shape[:2]

    print(f"Bilde lastet: Bredde={width}, Høyde={height}")

    # Bildebehandlingsfunksjoner

    def padding(image, border_width):
        """Legger til en reflektert kant."""

        padded_image = cv2.copyMakeBorder(
            image,
            border_width,
            border_width,
            border_width,
            border_width,
            cv2.BORDER_REFLECT
        )

        cv2.imwrite('padded_iris-1.png', padded_image)
        print("Lagret: padded_iris-1.png")

        return padded_image

    def crop(image, x_0, x_1, y_0, y_1):
        """Beskjærer bildet."""

        cropped_image = image[y_0:y_1, x_0:x_1]

        cv2.imwrite('cropped_iris-1.png', cropped_image)
        print("Lagret: cropped_iris-1.png")

        return cropped_image

    def resize(image, width, height):
        """Endrer bildets størrelse."""

        resized_image = cv2.resize(image, (width, height))

        cv2.imwrite('resized_iris-1.png', resized_image)
        print("Lagret: resized_iris-1.png")

        return resized_image

    def copy(image, emptyPictureArray):
        """Kopierer pikslene manuelt."""

        height, width, channels = image.shape

        for y in range(height):
            for x in range(width):
                for channel in range(channels):
                    emptyPictureArray[y, x, channel] = image[y, x, channel]

        cv2.imwrite('copied_iris-1.png', emptyPictureArray)
        print("Lagret: copied_iris-1.png")

        return emptyPictureArray

    def grayscale(image):
        """Gjør bildet til gråtone."""

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        cv2.imwrite('grayscale_iris-1.png', gray_image)
        print("Lagret: grayscale_iris-1.png")

        return gray_image

    def hsv(image):
        """Konverterer bildet til HSV."""

        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        cv2.imwrite('hsv_iris-1.png', hsv_image)
        print("Lagret: hsv_iris-1.png")

        return hsv_image

    def hue_shifted(image, emptyPictureArray, hue):
        """Endrer fargeverdiene."""

        height, width, channels = image.shape

        for y in range(height):
            for x in range(width):
                for channel in range(channels):

                    original_value = int(image[y, x, channel])
                    new_value = (original_value + hue) % 256

                    emptyPictureArray[y, x, channel] = new_value

        cv2.imwrite('hue_shifted_iris-1.png', emptyPictureArray)
        print("Lagret: hue_shifted_iris-1.png")

        return emptyPictureArray

    def smoothing(image):
        """Gjør bildet uskarpt."""

        smoothed_image = cv2.GaussianBlur(
            image,
            (15, 15),
            0
        )

        cv2.imwrite('smoothed_iris-1.png', smoothed_image)
        print("Lagret: smoothed_iris-1.png")

        return smoothed_image

    def rotation(image, rotation_angle):
        """Roterer bildet."""

        if rotation_angle == 90:
            rotated_image = cv2.rotate(
                image,
                cv2.ROTATE_90_CLOCKWISE
            )

        elif rotation_angle == 180:
            rotated_image = cv2.rotate(
                image,
                cv2.ROTATE_180
            )

        else:
            print("Bare 90 og 180 grader støttes.")
            return image

        cv2.imwrite(
            f'rotated_{rotation_angle}_iris-1.png',
            rotated_image
        )

        print(f"Lagret: rotated_{rotation_angle}_iris-1.png")

        return rotated_image

    # Kjør funksjonene

    # 1. Padding
    padding(image, 100)

    # 2. Beskjæring
    x_0 = 200
    x_1 = width - 130
    y_0 = 200
    y_1 = height - 130

    crop(image, x_0, x_1, y_0, y_1)

    # 3. Resize
    resize(image, 200, 200)

    # 4. Manuell kopiering
    emptyPictureArray = np.zeros(
        (height, width, 3),
        dtype=np.uint8
    )

    copy(image, emptyPictureArray)

    # 5. Grayscale
    grayscale(image)

    # 6. HSV
    hsv(image)

    # 7. Fargeskift med 50
    emptyPictureArray = np.zeros(
        (height, width, 3),
        dtype=np.uint8
    )

    hue_shifted(image, emptyPictureArray, 50)

    # 8. Smoothing
    smoothing(image)

    # 9. Rotasjon 180 grader
    rotation(image, 180)

    print("\nAlle oppgavene er fullført!")

if __name__ == "__main__":
    main()
