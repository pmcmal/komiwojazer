"""
Algorytm poszukujący ścieżki do celu
Paweł Malec i Miłosz Darecky 2020
"""
 
import matplotlib.pyplot as plt  # Biblioteka do wykresów, rozszerzenie numpy
import numpy as np               # Biblioteka do wielowymiarowych tablic i macierzy
 
N_CITIES = 15        # Wielkość DNA
CROSS_RATE = 0.5     # Wskaźnik krzyżowy
MUTATE_RATE = 0.05   # Współczynnik mutacji
POP_SIZE = 100       # Ilość populacji
N_GENERATIONS = 100  # Liczba generacji
 
 
class GA(object):
    def __init__(self, DNA_size, cross_rate, mutation_rate, pop_size, ):
        self.DNA_size = DNA_size
        self.cross_rate = cross_rate
        self.mutate_rate = mutation_rate
        self.pop_size = pop_size
 
        self.pop = np.vstack([np.random.permutation(DNA_size) for _ in range(pop_size)])
 
    def translateDNA(self, DNA, city_position):     # Uzyskujemy współrzedne miast w kolejności
        line_x = np.empty_like(DNA, dtype=np.float64)
        line_y = np.empty_like(DNA, dtype=np.float64)
        for i, d in enumerate(DNA):
            city_coord = city_position[d]
            line_x[i, :] = city_coord[:, 0]
            line_y[i, :] = city_coord[:, 1]
        return line_x, line_y
 
    def get_fitness(self, line_x, line_y):
        total_distance = np.empty((line_x.shape[0],), dtype=np.float64)
        for i, (xs, ys) in enumerate(zip(line_x, line_y)):
            total_distance[i] = np.sum(np.sqrt(np.square(np.diff(xs)) + np.square(np.diff(ys))))
        fitness = np.exp(self.DNA_size * 2 / total_distance)
        return fitness, total_distance
 
    def select(self, fitness):
        idx = np.random.choice(np.arange(self.pop_size), size=self.pop_size, replace=True, p=fitness / fitness.sum())
        return self.pop[idx]
 
    def crossover(self, parent, pop):
        if np.random.rand() < self.cross_rate:
            i_ = np.random.randint(0, self.pop_size, size=1)                        # Wybierz inną osobę z populacji
            cross_points = np.random.randint(0, 2, self.DNA_size).astype(np.bool)   # Wybierz punkt crossover
            keep_city = parent[~cross_points]                                       # Znajdź numer miasta
            swap_city = pop[i_, np.isin(pop[i_].ravel(), keep_city, invert=True)]
            parent[:] = np.concatenate((keep_city, swap_city))
        return parent
 
    def mutate(self, child):
        for point in range(self.DNA_size):
            if np.random.rand() < self.mutate_rate:
                swap_point = np.random.randint(0, self.DNA_size)
                swapA, swapB = child[point], child[swap_point]
                child[point], child[swap_point] = swapB, swapA
        return child
 
    def evolve(self, fitness):
        pop = self.select(fitness)
        pop_copy = pop.copy()
        for parent in pop:  # Dla każdego z rodziców
            child = self.crossover(parent, pop_copy)
            child = self.mutate(child)
            parent[:] = child
        self.pop = pop
 
 
class TravelSalesPerson(object):
    def __init__(self, n_cities):
        self.city_position = np.random.rand(n_cities, 2)
        plt.ion()                        # Uruchom tryb interkatywny biblioteki matplotlib
 
    def plotting(self, lx, ly, total_d): # Tworzymy wykres w bibliotece matplotlib
        plt.cla()
        plt.scatter(self.city_position[:, 0].T, self.city_position[:, 1].T, s=100, c='k')
        plt.plot(lx.T, ly.T, 'b-')
        plt.text(0.3, -0.225, "Łączny dystans=%.2f" % total_d, fontdict={'size': 10, 'color': 'black'})
        plt.text(0.2, 1.15, "Paweł Malec i Miłosz Darecky 2020", fontdict={'size': 10, 'color': 'black'})
        plt.xlim((-0.1, 1.1))
        plt.ylim((-0.1, 1.1))
        plt.pause(0.01)
 
 
ga = GA(DNA_size=N_CITIES, cross_rate=CROSS_RATE, mutation_rate=MUTATE_RATE, pop_size=POP_SIZE)
 
env = TravelSalesPerson(N_CITIES)
for generation in range(N_GENERATIONS):
    lx, ly = ga.translateDNA(ga.pop, env.city_position)
    fitness, total_distance = ga.get_fitness(lx, ly)
    ga.evolve(fitness)
    best_idx = np.argmax(fitness)
    print('Generacja:', generation, '| Najlepsze dopasowanie: %.2f' % fitness[best_idx],)
 
    env.plotting(lx[best_idx], ly[best_idx], total_distance[best_idx])
 
plt.ioff()    # Wyłącz tryb interaktywny biblioteki matplotlib
plt.show()    # Wyświetl wykres biblioteki matplolib