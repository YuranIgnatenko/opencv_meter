from extractor.extractor import extract_value

def main():
    mode = True
    test_names = [
        "testimg/i1.jpg",
        "testimg/i2.jpg",
        "testimg/i3.jpg",
        "testimg/i4.jpg",
        "testimg/i5.jpg",
        "testimg/i6.jpg",
            ]

    for name in test_names:
        result = extract_value(name, mode)
        print(f"[image:{name}] [angle:{result}]")

if __name__ == "__main__":
    main()