#!/usr/bin/env python3
"""Печатает путь к подпапке sparse/N с наибольшим числом зарегистрированных изображений."""
import os
import sys


def count_images(images_txt):
    n = 0
    with open(images_txt, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 10:
                continue
            try:
                int(parts[0])
            except ValueError:
                continue
            n += 1
    return n


def main():
    sparse_root = sys.argv[1]
    best = None
    best_n = -1
    for name in sorted(os.listdir(sparse_root), key=lambda x: (len(x), x)):
        p = os.path.join(sparse_root, name, "images.txt")
        if not os.path.isfile(p):
            continue
        n = count_images(p)
        if n > best_n:
            best_n = n
            best = os.path.join(sparse_root, name)
    if best is None:
        print("", end="")
        sys.exit(1)
    print(best, end="")


if __name__ == "__main__":
    main()
