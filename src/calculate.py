import numpy as np
from scipy.linalg import lu_factor, lu_solve

__author__ = "Khashayar Ghamati"
__email__ = "khashayarghamati@gmail.com"


class Calculate(object):
    def __init__(self, matrix, n):
        self.matrix = matrix
        self.n = n
        self.last_matrix = None
        self.L = np.identity(self.n)

    def changePositionOfrows(self, matrix, from_row, to_row):
        i = np.argsort(matrix)
        matrix = i[:, from_row, to_row]
        return matrix

    def producePivotElement(self):
        below_diagonal = np.tril(self.matrix, -1)
        below_diagonal = np.delete(below_diagonal, 0)

        diagonal = self.matrix.diagonal()

        for i, m in enumerate(self.matrix):
            if len(m) > len(self.matrix[0]):
                self.changePositionOfrows(matrix=self.matrix, from_row=i, to_row=0)

        for i, diag in enumerate(diagonal):

            pivot = []
            for j in below_diagonal:
                denominator = diag
                if below_diagonal[i][j] != 0:
                    pivot.append(-1 * below_diagonal[i][j] / denominator)

            if len(pivot) > 0:
                self.produceL(pivot, i)
                e = self.produceElementaryLowerTriangular(i, pivot)

                if self.last_matrix is None:
                    self.last_matrix = self.multiplyMatrix(e, self.matrix)
                else:
                    self.last_matrix = self.multiplyMatrix(e, self.last_matrix)

    def produceElementaryLowerTriangular(self, i, pivot):
        e = np.identity(self.n)
        col = self.getSpeceficCol(i)
        if i == 0:
            v = np.place(col, col == 0, pivot)
            e[:, i] = v

        else:
            temp = col
            for idx, row in enumerate(col):
                if row != 1:
                    temp = np.delete(temp, idx)
                else:
                    break

            v = np.place(temp, temp == 0, pivot)

            while len(v) != len(col):
                v = np.insert(v, 0, 0)

            e[:, i] = v

        return e

    def multiplyMatrix(self, e, matrix):
        return np.matmul(e, matrix)

    def getU(self):
        return self.last_matrix

    def getSpeceficCol(self, col=0):
        return self.matrix[:, col]

    def produceL(self, pivot, col_index):
        col = self.getSpeceficCol(col_index)
        pivot = -1 * pivot
        v = np.place(col, col == 0, pivot)
        self.L[:, col_index] = v

    def getL(self):
        return self.L

    def getLU(self):
        self.lu, self.piv = lu_factor(self.matrix)

    def getSolution(self, b):
        return lu_solve((self.lu, self.piv), b)
