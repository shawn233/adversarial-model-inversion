import os
import shutil


def main():
    for root, dirs, files in os.walk("./", topdown=True):
        for filename in files:
            if "classifier" in filename or "inversion" in filename:
                os.makedirs(os.path.join("../ckpt", root), exist_ok=True)
                shutil.copy2(os.path.join(root, filename), os.path.join("../ckpt", root))
                os.remove(os.path.join(root, filename))


if __name__ == "__main__":
    main()