import cv2
import numpy
import os
def print_image_information(image):
    height = image.shape[0]
    width = image.shape[1]
    channels = image.shape[2] if len(image.shape) == 3 else 1
    size = image.size
    data_type = image.dtype

    print(f"Height: {height}")
    print(f"Width: {width}")
    print(f"Channels: {channels}")
    print(f"Size: {size}")
    print(f"Data type: {data_type}")


def save_camera_info():
    # Åpne webkamera
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Feil: Kunne ikke åpne webkamera")
        return

    # Hent informasjon
    fps = cap.get(cv2.CAP_PROP_FPS)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    # Lukk kameraet
    cap.release()

    # Lagre til tekstfil
    output_path = "solutions/camera_outputs.txt"

    # Opprett solutionsmappen hvis den ikke finnes
    os.makedirs("solutions", exist_ok=True)

    with open(output_path, 'w') as f:
        f.write(f"fps: {fps}\n")
        f.write(f"height: {height}\n")
        f.write(f"width: {width}\n")
    # Printe info
    print(f"Kamerainformasjon lagret til: {output_path}")
    print(f"fps: {fps}")
    print(f"height: {height}")
    print(f"width: {width}")


def main():
    # Del IV Bildeinformasjon
    print(" Del IV: Bildeinformasjon ")
    image_path = "iris-1.jpg"
    image = cv2.imread(image_path)

    if image is None:
        print("Feil: Kunne ikke laste bildet. Sjekk filbanen.")
        return

    print_image_information(image)
    print("\n")

    # Del V Kamerainformasjon
    print("Del V: Kamerainformasjon ")
    save_camera_info()


if __name__ == "__main__":
    main()