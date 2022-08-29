

class StringBuffer:

    def __init__(self):
        self.array = list()

    def append(self, s):
        self.array.append(s)

    def append_numpy_2d_matrix(self, matrix, cell_size=5):
        for y in range(matrix.shape[0]):
            for x in range(matrix.shape[1]):
                self.array.append()


    def get_string(self, delimiter=""):
        return delimiter.join(self.array)
