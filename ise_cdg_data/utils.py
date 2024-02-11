import math
import os
import random


class LazyFileName:
    def __init__(self, path_to_file) -> None:
        self.__path = path_to_file
        self.__base_name, self.__name, self.__extension = [None] * 3

    def __set_fields(self):
        if self.__base_name is not None:  # Guard
            return
        self.__base_name = os.path.basename(self.__path)
        self.__name, self.__extension = os.path.splitext(self.__base_name)

    @property
    def base_name(self):
        self.__set_fields()
        return self.__base_name

    @property
    def name(self):
        self.__set_fields()
        return self.__name

    @property
    def extension(self):
        self.__set_fields()
        return self.__extension


def get_range(batch_size, current_batch, size):
    return range(
        current_batch * batch_size, min(size, (current_batch + 1) * batch_size)
    )


def train_test_split(size: int, test_portion: float):
    test_size = int(math.ceil(test_portion * size))
    test_indices = random.sample(range(0, size), test_size)
    train_indices = [ind for ind in range(0, size) if ind not in test_indices]
    return train_indices, test_indices
