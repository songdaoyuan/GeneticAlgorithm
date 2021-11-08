# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


class GeneticAlgorithm():
    def __init__(self) -> None:
        self.DNA_SIZE: int = 24  # DNA双链长度，奇数位存储X1，偶数位存储X2
        self.POP_SIZE: int = 200 # 总群容量
        self.CROSSOVER_RATE: float = 0.8  # 交叉概率
        self.MUTATION_RATE: float = 0.01  # 变异概率
        self.N_GENERATION: int = 200  # 迭代次数
        self.X1_BOUND: list = [-32, 32]  # 变量X1定义域
        self.X2_BOUND: list = [-32, 32]  # 变量X2定义域
        self.ADAPTABILITY = []


    def target_function(self, x1, x2) -> float:
        formula = -20 * np.exp(-0.2 * np.sqrt(0.5 * (x1**2 + x2**2))) - np.exp(
            0.5 * (np.cos(2 * np.pi * x1) + np.cos(2 * np.pi * x2))) + np.e + 20

        return formula

    def translate_DNA(self, pop) -> np.array:
        X1_pop = pop[:, 1::2]
        X2_pop = pop[:, ::2]
        x1 = X1_pop.dot(2**np.arange(self.DNA_SIZE)[::-1]) / float(
            2**self.DNA_SIZE - 1) * (self.X1_BOUND[1] - self.X1_BOUND[0]) + self.X1_BOUND[0]
        x2 = X2_pop.dot(2**np.arange(self.DNA_SIZE)[::-1]) / float(
            2**self.DNA_SIZE - 1) * (self.X2_BOUND[1] - self.X2_BOUND[0]) + self.X2_BOUND[0]

        return x1, x2

    def survival_of_the_fittest(self, pop) -> np.array:
        x1, x2 = self.translate_DNA(pop)
        #print(x1, x2)
        predected_value = self.target_function(x1, x2)

        return -(predected_value - np.max(predected_value)) + 1e-3

    def crossover_and_mutation(self, pop, CROSSOVER_RATE, MUTATION_RATE) -> list:
        CROSSOVER_RATE = self.CROSSOVER_RATE
        MUTATION_RATE = self.MUTATION_RATE
        new_pop = []

        def crossover(father, mother, CROSSOVER_RATE) -> np.array:
            cross_point = np.random.randint(low=0, high=self.DNA_SIZE*2)
            child = np.append(father[cross_point:], mother[:cross_point]) if np.random.rand(
            ) < CROSSOVER_RATE else father

            return child

        def mutation(child, MUTATION_RATE) -> list:
            mutation_point = np.random.randint(low=0, high=self.DNA_SIZE)
            child[mutation_point] = child[mutation_point] ^ 1 if np.random.rand(
            ) < MUTATION_RATE else child[mutation_point]

            return child

        for father in pop:
            mother = pop[np.random.randint(self.POP_SIZE)]
            child = crossover(father, mother, CROSSOVER_RATE)
            child = mutation(child, MUTATION_RATE)
            new_pop.append(child)

        return new_pop

    def select(self, pop, fitness) -> list:
        idx = np.random.choice(np.arange(self.POP_SIZE), size=self.POP_SIZE, replace=1, p=(fitness)/(fitness.sum()))

        return pop[idx]

    def run(self, pop) -> None:
        fitness = self.survival_of_the_fittest(pop)
        self.ADAPTABILITY.append(fitness.mean())
        best_fitness_index = np.argmax(fitness)
        #print(best_fitness_index)
        #print(fitness)
        print("ackley适应度为:", fitness[best_fitness_index])
        x1, x2 = self.translate_DNA(pop)
        print("(x1, x2):", (x1[best_fitness_index], x2[best_fitness_index]))




if __name__ == '__main__':
    GA = GeneticAlgorithm()
    pop = np.random.randint(2, size=(GA.POP_SIZE, GA.DNA_SIZE*2))
    GA.run(pop)

    for _ in range(GA.N_GENERATION):
        print(f'The {_} steps generation')
        x1, x2 = GA.translate_DNA(pop)
        pop = np.array(GA.crossover_and_mutation(pop, GA.CROSSOVER_RATE, GA.MUTATION_RATE))
        fitness = GA.survival_of_the_fittest(pop)
        pop = GA.select(pop, fitness)
        GA.run(pop)

    
    x=np.arange(0, GA.N_GENERATION+1)
    y = np.array(GA.ADAPTABILITY)
    plt.figure()
    plt.title('Fitness and number of iterations')
    plt.xlabel('Number of iterations')
    plt.ylabel('Fitness')
    plt.plot(x,y)
    plt.show()