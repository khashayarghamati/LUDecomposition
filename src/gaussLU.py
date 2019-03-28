import ast
import re

import numpy as np
import scipy.linalg

from src.calculate import Calculate

__author__ = "Khashayar Ghamati"
__email__ = "khashayarghamati@gmail.com"


class GaussLU(object):

    def getEquation(self):
        n = int(input("insert number of row: "))
        m = int(input("insert number of column: "))

        if n != m:
            print("Your Matrix is not square !")
        else:
            self.value = []
            self.matrix = []

            for i in range(n):
                equation = input(f"insert vector {i+1} :")
                vector = self.convertStringToList(list=equation)

                if len(self.matrix) == 0:
                    self.matrix = np.array(vector)
                else:
                    self.matrix = np.vstack([self.matrix, vector])

            self.value = self.convertStringToList(input(f"insert values :"))

            isMatrixValid, msg = self._isMatrixValid()
            if isMatrixValid:
                gwp = Calculate(matrix=self.matrix, n=n)
                P, L, U = scipy.linalg.lu(self.matrix)

                print(f"\nYour matrix is: \n {self.matrix}")
                print(f"\nP is: \n {P}")
                print(f"\nL is: \n {L}")
                print(f"\nU is: \n {U}")

                gwp.getLU()
                solution = gwp.getSolution(self.value)
                print(f"\nSolution is: \n {solution}")

            else:
                print(msg)

    def _isEquationValid(self, equation):

        if '=' not in equation:
            return "Your equation must equel to a scaler"
        elif '+' not in equation:
            return "Your equation is not valid"

        return True

    def convertStringToList(self, list):
        return ast.literal_eval(list)

    def _parseEquation(self, equation):
        elements = equation.split('+')
        value = int(re.search(r'-?\d+', equation.split('=')[1]).group())

        vector = []
        for idx, element in enumerate(elements):
            vector.append(int(re.search(r'-?\d+', element).group()))

        return vector, value

    def _isMatrixValid(self):

        if np.linalg.det(self.matrix) == 0:
            return False, "the determinant must not be zero"

        diagonal = self.matrix.diagonal()

        for idx, element in enumerate(diagonal):
            if idx != 0 and element == 0:
                return False, "the main diagonal must not contain zero"

        return True, "Your matrix is ready :)"


if __name__ == '__main__':
    lu = GaussLU()
    lu.getEquation()
