import numpy as np
from functools import cmp_to_key

def IntegerConversionFunction(ch):
    """
    Convert a single character into its corresponding integer grayscale level.
    ' ' -> 0
    '+' -> 1
    '#' -> 2
    """
    if ch == ' ':
        return 0
    elif ch == '+':
        return 1
    elif ch == '#':
        return 2

def convertToInteger(data):
    """
    Recursively convert all elements in 'data' into their integer representations.
    If 'data' is an ndarray, map the conversion function to each element.
    Otherwise, convert a single character.
    """
    if not isinstance(data, np.ndarray):
        return IntegerConversionFunction(data)
    else:
        return np.array(list(map(convertToInteger, data)))

def AsciiGrayscaleConversionFunction(val):
    """
    Convert integer grayscale values back to their ASCII characters.
    0 -> ' '
    1 -> '+'
    2 -> '#'
    """
    if val == 0:
        return ' '
    elif val == 1:
        return '+'
    elif val == 2:
        return '#'

def convertToAscii(data):
    """
    Recursively convert an integer ndarray back into ASCII representation.
    """
    if not isinstance(data, np.ndarray):
        return AsciiGrayscaleConversionFunction(data)
    else:
        return np.array(list(map(convertToAscii, data)))

def loadDataFileRandomly(filepath: str, randomOrder: list, width: int, height: int):
    """
    Load specific images from a file using the provided indices in randomOrder.
    Each image is 'height' lines of 'width' characters.
    """
    items = []
    with open(filepath, "r") as f:
        for idx in randomOrder:
            f.seek(idx * (width + 1) * height, 0)
            image_lines = []
            for _ in range(height):
                line = f.readline()
                # Exclude newline and convert line to list of characters
                image_lines.append([c for c in line if c != '\n'])
            items.append(Picture(np.array(image_lines), width, height))
    return items

def loadLabelFileRandomly(filepath: str, randomOrder: list):
    """
    Load labels from a label file using random indices.
    Each label is assumed to occupy 2 bytes (character + newline).
    """
    labels = []
    with open(filepath, "r") as f:
        for idx in randomOrder:
            f.seek(2 * idx, 0)
            labels.append(f.read(1))
    return labels

def loadDataFile(filePath: str, totalPicNum: int, width: int, height: int):
    """
    Load a contiguous set of images from a file.
    """
    items = []
    with open(filePath, "r") as f:
        for i in range(totalPicNum):
            f.seek(i * (width + 1) * height, 0)
            image_data = []
            for _ in range(height):
                line = f.readline()
                image_data.append([ch for ch in line if ch != '\n'])
            items.append(Picture(np.array(image_data), width, height))
    return items

def loadLabelFile(filePath: str, totalPicNum: int):
    """
    Load a contiguous set of labels from a file.
    Each label is stored as one character plus a newline (2 bytes).
    """
    labels = []
    with open(filePath, "r") as f:
        for i in range(totalPicNum):
            f.seek(2 * i, 0)
            labels.append(f.read(1))
    return labels

class Picture:
    """
    Picture class represents an image with binary (ASCII) data converted into integer pixels.
    It can be rotated and converted back to ASCII for visualization.
    """
    def __init__(self, data, width: int, height: int):
        self.width = width
        self.height = height
        if data is None:
            # If no data is provided, initialize with empty spaces
            data = [[' ' for _ in range(self.width)] for _ in range(self.height)]
        # Convert to integers and rotate
        self.pixels = np.rot90(convertToInteger(data), -1)

    def getPixel(self, column, row):
        return self.pixels[column][row]

    def getPixels(self):
        return self.pixels

    def getAsciiString(self):
        # Rotate back to original orientation before converting to ASCII
        rotated_back = np.rot90(self.pixels, 1)
        ascii_art = convertToAscii(rotated_back)
        return '\n'.join(''.join(map(str, line)) for line in ascii_art)

    def __str__(self):
        return self.getAsciiString()

def sign(x):
    """
    Return 1 if x >= 0, else -1.
    """
    return 1 if x >= 0 else -1

class Counter(dict):
    """
    A dictionary that defaults missing values to 0 and supports various arithmetic and
    utility methods often needed in machine learning tasks.
    """
    def __getitem__(self, index):
        if index not in self:
            dict.__setitem__(self, index, 0)
        return dict.__getitem__(self, index)

    def incrementALL(self, keys, count):
        for key in keys:
            self[key] += count

    def argMax(self):
        if len(self) == 0:
            return None
        # Find key with the maximum value
        items_list = list(self.items())
        values = [pair[1] for pair in items_list]
        max_idx = values.index(max(values))
        return items_list[max_idx][0]

    def sortedKeys(self):
        # Sort by value in descending order
        sorted_items = sorted(self.items(), key=lambda x: x[1], reverse=True)
        return [key for key, val in sorted_items]

    def totalCount(self):
        return sum(self.values())

    def normalize(self):
        total = float(self.totalCount())
        if total == 0:
            return
        for k in self.keys():
            self[k] = self[k] / total

    def divideAll(self, divisor):
        divisor = float(divisor)
        for k in self:
            self[k] /= divisor

    def copy(self):
        return Counter(dict.copy(self))

    def __mul__(self, y):
        # Dot product of two Counters
        result = 0
        x = self
        # Iterate over smaller one to be efficient
        if len(x) > len(y):
            x, y = y, x
        for key in x:
            if key in y:
                result += x[key] * y[key]
        return result

    def __radd__(self, y):
        # Right-add: increment this counter by y's items
        for key, value in y.items():
            self[key] += value

    def __add__(self, y):
        # Add corresponding counts
        sum_counter = Counter()
        for k in self:
            sum_counter[k] = self[k] + y.get(k, 0)
        for k in y:
            if k not in self:
                sum_counter[k] = y[k]
        return sum_counter

    def __sub__(self, y):
        # Subtract y from self
        diff_counter = Counter()
        for k in self:
            diff_counter[k] = self[k] - y.get(k, 0)
        for k in y:
            if k not in self:
                diff_counter[k] = -y[k]
        return diff_counter

if __name__ == '__main__':
    # Example usage to load and display images
    Width, Height = 60, 70
    dataPath = 'data/facedata/facedatatrain'
    # Just load first 4 images to test
    pictures = loadDataFile(dataPath, 4, Width, Height)
    print("Loaded Images:")
    for i, img in enumerate(pictures):
        print(f"Image {i}:")
        print(img)
        print("-------")
