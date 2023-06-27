class Vector:
    def __init__(self, size: int = 0, val=0):
        if type(val) == int or type(val) == float:
            if size < 0:
                raise ValueError
            self.__size = size
            self.__array = [val for _ in range(size)]
        else:
            raise TypeError

    def __str__(self):
        string = ""
        for i in range(self.__size):
            string += str(self.__array[i])
            if i != self.__size - 1:
                string += ", "
        return "[" + string + "]"

    def __len__(self):
        return self.__size

    def __setitem__(self, i, val):
        if type(val) == int or type(val) == float:
            if 0 <= i < self.__size:
                self.__array[i] = val
            else:
                raise IndexError
        else:
            raise TypeError

    def __getitem__(self, i):
        if 0 <= i < self.__size:
            return self.__array[i]
        else:
            raise IndexError

    def __add__(self, other):
        if isinstance(other, self.__class__):
            if self.__size != other.__size:
                raise IndexError
            for i in range(self.__size):
                self.__array[i] += other.__array[i]
            return self
        elif type(other) == int or type(other) == float:
            for i in range(self.__size):
                self.__array[i] += other
            return self
        else:
            raise TypeError

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        if isinstance(other, self.__class__):
            if self.__size != other.__size:
                raise IndexError
            for i in range(self.__size):
                self.__array[i] -= other.__array[i]
            return self
        elif type(other) == int or type(other) == float:
            for i in range(self.__size):
                self.__array[i] -= other
            return self
        else:
            raise TypeError

    def __rsub__(self, other):
        return self - other

    def __mul__(self, other):
        if isinstance(other, self.__class__):
            if self.__size != other.__size:
                raise IndexError
            for i in range(self.__size):
                self.__array[i] *= other.__array[i]
            return self
        elif type(other) == int or type(other) == float:
            for i in range(self.__size):
                self.__array[i] *= other
            return self
        else:
            raise TypeError

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        if isinstance(other, self.__class__):
            if self.__size != other.__size:
                raise IndexError
            for i in range(self.__size):
                self.__array[i] /= other.__array[i]
            return self
        elif type(other) == int or type(other) == float:
            for i in range(self.__size):
                self.__array[i] /= other
            return self
        else:
            raise TypeError

    def __rtruediv__(self, other):
        return self / other

    def __floordiv__(self, other):
        if isinstance(other, self.__class__):
            if self.__size != other.__size:
                raise IndexError
            for i in range(self.__size):
                self.__array[i] //= other.__array[i]
            return self
        elif type(other) == int or type(other) == float:
            for i in range(self.__size):
                self.__array[i] //= other
            return self
        else:
            raise TypeError

    def __rfloordiv__(self, other):
        return self // other

    def __matmul__(self, other):
        if isinstance(other, self.__class__) or type(other) == int or type(other) == float:
            return self * other
        elif isinstance(other, Matrix):
            n, m = other.get_size()
            if self.__size != n:
                raise IndexError
            ans = Vector(m)
            for j in range(m):
                summ = 0
                for k in range(n):
                    summ += self[k] * other[k][j]
                ans[j] = summ
            return ans
        else:
            raise TypeError

    def __rmatmul__(self, other):
        if type(other) == int or type(other) == float:
            return self * other
        else:
            raise TypeError

    def __pow__(self, other):
        if isinstance(other, self.__class__):
            if self.__size != other.__size:
                raise IndexError
            for i in range(self.__size):
                self.__array[i] **= other.__array[i]
            return self
        elif type(other) == int or type(other) == float:
            for i in range(self.__size):
                self.__array[i] **= other
            return self
        else:
            raise TypeError

    def copy(self):
        new_vector = Vector(self.__size)
        for i in range(self.__size):
            new_vector[i] = self[i]
        return new_vector

    def transpose(self):
        new_matrix = Matrix(self.__size, 1)
        for i in range(self.__size):
            new_matrix[i][0] = self.__array[i]
        return new_matrix

    def get_size(self):
        return self.__size

    def resize(self, new_size: int):
        if new_size < 0:
            raise ValueError
        if new_size < self.__size:
            for i in range(self.__size - new_size):
                self.__array.pop()
        elif new_size > self.__size:
            for i in range(new_size - self.__size):
                self.__array.append(0)
        self.__size = new_size

    def append(self, val):
        if type(val) == int or type(val) == float:
            self.__array.append(val)
            self.__size += 1
        else:
            raise TypeError

    def pop(self):
        if self.__size == 0:
            raise IndexError
        self.__array.pop()
        self.__size -= 1

    def in_list(self):
        return self.__array.copy()

    def from_list(self, array: list):
        for elem in array:
            if type(elem) != int and type(elem) != float:
                raise TypeError
        self.__size = len(array)
        self.__array = array
        return self

    def from_matrix(self, mtrx):
        if isinstance(mtrx, Matrix):
            n, m = mtrx.get_size()
            if n != 1:
                raise ValueError
            self.__size = m
            self.__array = mtrx[0]
            return self
        else:
            raise TypeError

    def in_func(self, func, var_func, *args, **kwargs):
        for i in range(self.__size):
            kwargs.update({var_func: self.__array[i]})
            self.__array[i] = func(*args, **kwargs)
        return self


class Matrix:
    def __init__(self, n: int = 0, m: int = 0, val=0):
        if type(val) == int or type(val) == float:
            if n < 0 or m < 0:
                raise ValueError
            self.__n = n
            self.__m = m
            self.__matrix = [Vector(m, val) for _ in range(n)]
        else:
            raise TypeError

    def __str__(self):
        string = ""
        for i in range(self.__n):
            string += str(self.__matrix[i]) + "\n"
        return string

    def __len__(self):
        return self.__n, self.__m

    def __getitem__(self, i):
        if 0 <= i < self.__n:
            return self.__matrix[i]
        else:
            raise IndexError

    def __call__(self, i, j):
        if 0 <= i < self.__n and 0 <= j < self.__m:
            return self.__matrix[i][j]
        else:
            raise IndexError

    def __add__(self, other):
        if isinstance(other, self.__class__):
            if self.__n != other.__n:
                raise IndexError
            for i in range(self.__n):
                self.__matrix[i] += other.__matrix[i]
            return self
        elif isinstance(other, Vector):
            if self.__m != other.get_size():
                raise IndexError
            for i in range(self.__n):
                self.__matrix[i] += other
            return self
        elif type(other) == int or type(other) == float:
            for i in range(self.__n):
                self.__matrix[i][i] += other
            return self
        else:
            raise TypeError

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        if isinstance(other, self.__class__):
            if self.__n != other.__n:
                raise IndexError
            for i in range(self.__n):
                self.__matrix[i] -= other.__matrix[i]
            return self
        elif isinstance(other, Vector):
            if self.__m != other.get_size():
                raise IndexError
            for i in range(self.__n):
                self.__matrix[i] -= other
            return self
        elif type(other) == int or type(other) == float:
            for i in range(self.__n):
                self.__matrix[i][i] -= other
            return self
        else:
            raise TypeError

    def __rsub__(self, other):
        return self - other

    def __mul__(self, other):
        if type(other) == int or type(other) == float:
            for i in range(self.__n):
                self.__matrix[i] *= other
            return self
        else:
            raise TypeError

    def __rmul__(self, other):
        if type(other) == int or type(other) == float:
            return self * other
        else:
            raise TypeError

    def __truediv__(self, other):
        if type(other) == int or type(other) == float:
            for i in range(self.__n):
                self.__matrix[i] /= other
            return self
        else:
            raise TypeError

    def __rtruediv__(self, other):
        if type(other) == int or type(other) == float:
            return self / other
        else:
            raise TypeError

    def __floordiv__(self, other):
        if type(other) == int or type(other) == float:
            for i in range(self.__n):
                self.__matrix[i] //= other
            return self
        else:
            raise TypeError

    def __rfloordiv__(self, other):
        if type(other) == int or type(other) == float:
            return self // other
        else:
            raise TypeError

    def __matmul__(self, other):
        if isinstance(other, self.__class__):
            if self.__m != other.__n:
                raise IndexError
            ans = Matrix(self.__n, other.__m)
            for i in range(self.__n):
                for j in range(other.__m):
                    summ = 0
                    for k in range(other.__n):
                        summ += self[i][k] * other[k][j]
                    ans[i][j] = summ
            return ans
        elif isinstance(other, Vector):
            new_other = Matrix().from_vector(other)
            return self @ new_other
        elif type(other) == int or type(other) == float:
            return self * other
        else:
            raise TypeError

    def __rmatmul__(self, other):
        if type(other) == int or type(other) == float:
            return self * other
        else:
            raise TypeError

    def __pow__(self, other):
        if type(other) == int or type(other) == float:
            for i in range(self.__n):
                self.__matrix[i] **= other
            return self
        else:
            raise TypeError

    def copy(self):
        new_matrix = Matrix(self.__n, self.__m)
        for i in range(self.__n):
            for j in range(self.__m):
                new_matrix[i][j] = self[i][j]
        return new_matrix

    def transpose(self):
        new_matrix = [Vector(self.__n) for _ in range(self.__m)]
        for j in range(self.__m):
            for i in range(self.__n):
                new_matrix[j][i] = self.__matrix[i][j]
        self.__n, self.__m = self.__m, self.__n
        self.__matrix = new_matrix
        return self

    def get_size(self):
        return self.__n, self.__m

    def from_list(self, array: list):
        self.__n = len(array)
        self.__m = 0
        self.__matrix = list()
        if self.__n == 0:
            return
        self.__m = len(array[0])
        for line in array:
            if len(line) != self.__m:
                raise ValueError
            self.__matrix.append(Vector().from_list(line))
        return self

    def from_vector(self, vect):
        if isinstance(vect, Vector):
            self.__n = 1
            self.__m = len(vect)
            self.__matrix = [vect]
            return self
        else:
            raise TypeError

    def in_list(self):
        array = list()
        for line in self.__matrix:
            array.append(line.in_list())
        return array

    def in_func(self, func, var, *args, **kwargs):
        for i in range(self.__n):
            self.__matrix[i].in_func(func, var, *args, **kwargs)
        return self


class Identity(Matrix):
    def __init__(self, n):
        super(Identity, self).__init__(n, n)
        for i in range(n):
            self[i][i] = 1
