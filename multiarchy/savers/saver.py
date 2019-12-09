"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from abc import ABC, abstractmethod


class Saver(ABC):

    @abstractmethod
    def save(
        self,
    ):
        return NotImplemented

    @abstractmethod
    def load(
        self,
    ):
        return NotImplemented
