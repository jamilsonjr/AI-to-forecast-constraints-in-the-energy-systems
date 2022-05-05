from abc import ABC, abstractmethod

# Implement the Stretegy design pattern.
class Strategy(ABC):
    @abstractmethod
    def do_algorithm(self, data: list) -> None:
        pass    
class Strategy_1(Strategy):
    def do_algorithm(self, data: list) -> None:
        print("Strategy 1: ", data)
        return data * 100

class Context():
    def __init__(self, strategy: Strategy) -> None:
        self._strategy = strategy
    @property
    def strategy(self) -> Strategy:
        return self._strategy
    @strategy.setter
    def strategy(self, strategy: Strategy) -> None:
        self._strategy = strategy
    def do_algorithm(self, data: list) -> None:
        return self._strategy.do_algorithm(data)
        