###########################
try:
    import cv2.cv2 as cv2
except Exception as e:
    import cv2
###########################
from utils import example
from pathlib import Path


def capture_and_save(im):
    im = example(mediapipe=True, im=im, show_result=False)

    m = 0
    p = Path("images")
    for imp in p.iterdir():
        if imp.suffix == ".png" and imp.stem not in ["last", "table"]:
            num = imp.stem.split("_")[1]
            try:
                num = int(num)
                if num > m:
                    m = num
            except:
                print("Error reading image number for", str(imp))
    m += 1
    lp = Path("images/last.png")
    if lp.exists() and lp.is_file():
        np = Path("images/img_{}.png".format(m))
        np.write_bytes(lp.read_bytes())
    cv2.imwrite("images/last.png", im)


def test_capture_and_save():
    im_test = cv2.imread("images/test.jpg")
    capture_and_save(im_test)
    print("done")


if __name__ == "__main__":
    test_capture_and_save()
