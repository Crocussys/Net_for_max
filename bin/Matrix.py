from typing import Union


def in_func(mtrx, func, var, *args, **kwargs):
    for i in range(mtrx.n):
        for j in range(mtrx.m):
            kwargs.update({var: mtrx.array[i][j]})
            mtrx.array[i][j] = func(*args, **kwargs)
    return mtrx


class Matrix:
    def __init__(self, n: int = 0, m: int = 0, val: Union[int, float] = 0):
        self.n = n
        self.m = m
        self.array = [[val for _ in range(m)] for _ in range(n)]

    def __str__(self):
        string = ""
        for i in range(self.n):
            for j in range(self.m):
                string += str(self.array[i][j]) + ", "
            string = f"[{string[:-2]}]\n"
        return string[:-1]

    def __add__(self, other):
        if isinstance(other, self.__class__):
            if self.n != other.n or self.m != other.m:
                raise ValueError
            for i in range(self.n):
                for j in range(self.m):
                    self.array[i][j] += other.array[i][j]
        elif isinstance(other, Union[int, float]):
            return self + Matrix(self.n, self.m, other)
        else:
            raise TypeError

    def __sub__(self, other):
        if isinstance(other, self.__class__):
            if self.n != other.n or self.m != other.m:
                raise ValueError
            for i in range(self.n):
                for j in range(self.m):
                    self.array[i][j] -= other.array[i][j]
        elif isinstance(other, Union[int, float]):
            return self + Matrix(self.n, self.m, other)
        else:
            raise TypeError

    def __setattr__(self, attr, val):
        if attr == "array":
            if type(val) != list():
                raise AttributeError
            self.n = len(val)
            self.m = 0
            for i in range(self.n):
                line = val[i]
                if type(line) != list():
                    raise AttributeError
                if i == 0:
                    self.m = len(line)
                else:
                    if len(line) != self.m:
                        raise AttributeError
                for elem in line:
                    if type(elem) != Union[int, float]:
                        raise AttributeError
            self.__dict__[attr] = val
        elif attr == "n" or attr == "m":
            if type(val) != int or val < 0:
                raise AttributeError
            self.__dict__[attr] = val
        else:
            raise AttributeError

    def __getitem__(self, i):
        return self.array[i]

    def __mul__(self, other: Union[int, float]):
        for i in range(self.n):
            for j in range(self.m):
                self.array[i][j] *= other

    def __truediv__(self, other: Union[int, float]):
        for i in range(self.n):
            for j in range(self.m):
                self.array[i][j] /= other

    def __floordiv__(self, other: Union[int, float]):
        for i in range(self.n):
            for j in range(self.m):
                self.array[i][j] //= other

    def __matmul__(self, other):
        if isinstance(other, self.__class__):
            if self.m != other.n:
                raise ValueError
            ans = Matrix(self.n, other.m)
            for i in range(self.n):
                for j in range(other.m):
                    summ = 0
                    for k in range(other.n):
                        summ += self[i][k] * other[k][j]
                    ans.set(summ, i, j)
            return ans
        else:
            raise TypeError

    def set(self, val, i, j):
        val = float(val)
        if 0 <= i < self.n and 0 <= j < self.m:
            self.array[i][j] = val
        else:
            raise ValueError

    def transpose(self):
        temp = list()
        for j in range(self.m):
            temp.append(list())
            for i in range(self.n):
                temp[j].append(self.array[i][j])
        self.array = temp


class Identity(Matrix):
    def __init__(self, n, m):
        super(Identity, self).__init__(n, m)
        self.array = list()
        for i in range(self.n):
            self.array.append(list())
            for j in range(self.m):
                if i == j:
                    self.array[i].append(1)
                else:
                    self.array[i].append(0)
