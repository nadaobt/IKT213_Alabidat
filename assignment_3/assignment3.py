import cv2
import numpy as np
import os


# Finn mappen der dette programmet ligger
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
def sobel_edge_detection(image: np.ndarray) -> np.ndarray:
    # Gjør bildet grått hvis det er i farger
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    # Fjern litt støy før Sobel
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    # Finn kanter med kravene fra oppgaven
    sobel = cv2.Sobel(
        blurred,
        cv2.CV_16S,
        dx=1,
        dy=1,
        ksize=1
    )
    # Gjør resultatet om til et vanlig bilde
    result = cv2.convertScaleAbs(sobel)
    # Lagre resultatet
    save_path = os.path.join(BASE_DIR, "sobel_edges.png")
    cv2.imwrite(save_path, result)
    return result
def canny_edge_detection(
        image: np.ndarray,
        threshold_1: int = 50,
        threshold_2: int = 50
) -> np.ndarray:

    # Gjør bildet grått hvis det er i farger
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Fjern litt støy før Canny
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Finn kanter med tersklene fra oppgaven
    edges = cv2.Canny(
        blurred,
        threshold_1,
        threshold_2
    )

    # Lagre resultatet
    save_path = os.path.join(BASE_DIR, "canny_edges.png")
    cv2.imwrite(save_path, edges)

    return edges


def template_match(
        image: np.ndarray,
        template: np.ndarray
) -> np.ndarray:

    # Gjør både bildet og malen grå
    if len(image.shape) == 3:
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        image_gray = image.copy()

    if len(template.shape) == 3:
        template_gray = cv2.cvtColor(
            template,
            cv2.COLOR_BGR2GRAY
        )
    else:
        template_gray = template.copy()

    # Finn hvor godt malen passer i bildet
    result = cv2.matchTemplate(
        image_gray,
        template_gray,
        cv2.TM_CCOEFF_NORMED
    )

    # Oppgaven krever terskel 0.9
    threshold = 0.9

    # Finn alle punkter som har godt nok treff
    y_coords, x_coords = np.where(result >= threshold)

    # Tegn på originalbildet
    output_image = image.copy()

    template_height, template_width = template_gray.shape

    # Lag en liste med områder som allerede er markert
    detected_areas = []

    for x, y in zip(x_coords, y_coords):

        # Lag området rundt dette treffet
        current_box = (
            x,
            y,
            x + template_width,
            y + template_height
        )

        # Sjekk om dette treffet overlapper et tidligere treff
        overlaps = False

        for old_box in detected_areas:

            old_x1, old_y1, old_x2, old_y2 = old_box
            new_x1, new_y1, new_x2, new_y2 = current_box

            if (
                new_x1 < old_x2
                and new_x2 > old_x1
                and new_y1 < old_y2
                and new_y2 > old_y1
            ):
                overlaps = True
                break

        # Tegn bare hvis området ikke allerede er markert
        if not overlaps:
            x1, y1, x2, y2 = current_box

            cv2.rectangle(
                output_image,
                (x1, y1),
                (x2, y2),
                (0, 0, 255),
                2
            )

            detected_areas.append(current_box)

    # Lagre resultatet
    save_path = os.path.join(
        BASE_DIR,
        "template_matched.png"
    )
    cv2.imwrite(save_path, output_image)

    return output_image


def resize(
        image: np.ndarray,
        scale_factor: int,
        up_or_down: str
) -> np.ndarray:

    # Sjekk at faktoren er et positivt heltall
    if not isinstance(scale_factor, int) or scale_factor <= 0:
        raise ValueError(
            "scale_factor må være et positivt heltall."
        )

    # Gjør teksten liten og enkel å sjekke
    direction = up_or_down.lower()

    if direction == "up":

        # pyrUp gjør bildet dobbelt så stort
        result = cv2.pyrUp(
            image,
            dstsize=(
                image.shape[1] * scale_factor,
                image.shape[0] * scale_factor
            )
        )

    elif direction == "down":

        # pyrDown gjør bildet halvparten så stort
        result = cv2.pyrDown(
            image,
            dstsize=(
                image.shape[1] // scale_factor,
                image.shape[0] // scale_factor
            )
        )

    else:
        raise ValueError(
            "up_or_down må være 'up' eller 'down'."
        )

    # Lagre resultatet
    filename = f"resized_{direction}.png"
    save_path = os.path.join(BASE_DIR, filename)
    cv2.imwrite(save_path, result)

    return result


def load_image(filename: str) -> np.ndarray:
    # Finn bildet i samme mappe som programmet
    path = os.path.join(BASE_DIR, filename)

    # Last inn bildet med cv2.imread()
    image = cv2.imread(path)

    # Stopp hvis bildet ikke kunne lastes
    if image is None:
        raise FileNotFoundError(
            f"Kunne ikke laste bildet: {path}"
        )

    return image


if __name__ == "__main__":

    # Last inn bildene som brukes i oppgaven
    lambo_image = load_image("lambo.png")
    shapes_image = load_image("shapes.png")
    template_image = load_image("shapes_template.jpg")

    # Del 1: Sobel edge detection
    sobel_edge_detection(lambo_image)

    # Del 2: Canny edge detection
    canny_edge_detection(
        lambo_image,
        threshold_1=50,
        threshold_2=50
    )

    # Del 3: Template matching
    template_match(
        shapes_image,
        template_image
    )

    # Del 4: Resize opp
    resize(
        lambo_image,
        scale_factor=2,
        up_or_down="up"
    )

    # Del 4: Resize ned
    resize(
        lambo_image,
        scale_factor=2,
        up_or_down="down"
    )
    print("Alle oppgavene er fullført.")