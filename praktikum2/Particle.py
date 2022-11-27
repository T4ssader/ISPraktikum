import random
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
import copy


''' 
Fitness Funktion zur Bestimmung der Eignung der Position eines Partikels
@:param x1: Variable1 x2: Vriable2
'''


def fitness_function(x1, x2):
    return x1 ** 2.0 + x2 ** 2.0


'''
Aktualisieren der Geschwindigkeit eines Partikels
@:param  
    particle: Partikel dessen Geschwindigkeit aktualisiert werden soll
    velocity: momentane Geschwindigkeit
    pbest: beste bisher bekannte Position des Partikels
'''


def update_velocity(particle, velocity, pbest, gbest, w_min=0.5, max=1.0, c=0.1):
    # Initial velocity [0,0]
    new_velocity = np.array([0.0 for i in range(len(particle))])

    # Randomly generate r1, r2 and inertia
    r1 = random.uniform(0, max)
    r2 = random.uniform(0, max)
    w = 0.8
    c1 = 0.15
    c2 = c

    # Calculate new velocity
    for i in range(len(particle)):
        new_velocity[i] = w * velocity[i] + c1 * r1 * (pbest[i] - particle[i]) + c2 * r2 * (gbest[i] - particle[i])


    return new_velocity


'''
Aktualisiert die Position des Partikels
@:param particle: Partikel welches aktualisiert werden soll  velocity: Geschwindigkeit des aktuellen Partikels
'''


def update_position(particle, velocity, position_min, position_max):
    new_particle = particle + velocity
    if new_particle[0] > position_max or new_particle[0] < position_min:
        new_particle[0] = particle[0]
    if new_particle[1] > position_max or new_particle[1] < position_min:
        new_particle[1] = particle[1]
    return new_particle


'''
Bereitet die Animation vor, muss einmal aufgerufen werden, bevor images gesichert werden können
'''


def plotting_preparation(position_min, position_max, fitness_function):
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    x = np.linspace(position_min, position_max)
    y = np.linspace(position_min, position_max)
    X, Y = np.meshgrid(x, y)
    Z = fitness_function(X, Y)

    ax.plot_wireframe(X, Y, Z, color='r', linewidth=0.35)
    return [ax, fig]


'''
Die Partikelschwarmoptimierung (derzeit nur 2d)
@:param
    population: Populationsgröße/Anzahl der Partikel
    dimension: 2  ->  Anzahl der Variablen
    position_min: Untere Grenze des zu betrachteten Bereichs der Fitnessfunktion
    position_max: Obere Grenze des zu betrachteten Bereichs der Fitnessfunktion
    generation: obere Grenze der Generationen bis abgebrochen wird
    fitness_criterion: Der Wert, welcher erreicht werden muss, damit der Algorithmus als Erfolg abbrechen darf
'''


def pso_2d(population, position_min, position_max, generation, fitness_criterion, fitness_function=fitness_function,
           file_name='pso_main'):
    # Population
    particles = [[random.uniform(position_min, position_max) for j in range(2)] for i in range(population)]
    # Particle best position
    particle_best_positions = copy.deepcopy(particles)
    # Fitness
    particle_best_fitness = [fitness_function(p[0], p[1]) for p in particles]
    # Global best particle position
    global_best_position = particles[np.argmin(particle_best_fitness)]
    # Velocity (starting from 0 speed)
    particle_velocity = [[0.0 for j in range(2)] for i in range(population)]
    # Animation image placeholder
    images = []
    ax_fig = plotting_preparation(position_min, position_max, fitness_function)
    ax = ax_fig[0]
    fig = ax_fig[1]

    # Loop for the number of generation
    for t in range(generation):
        # Stop if the average fitness value reached a predefined success criterion   fitness_function(global_best_position[0],global_best_position[1])<= fitness_criterion:
        if np.average(particle_best_fitness) <= fitness_criterion:
            break
        else:
            for n in range(population):
                # Update the velocity of each particle
                if fitness_function(particles[n][0], particles[n][1]) < fitness_function(particle_best_positions[n][0],
                                                                                         particle_best_positions[n][1]):
                    particle_best_positions[n] = particles[n]


                particle_velocity[n] = update_velocity(particles[n], particle_velocity[n], particle_best_positions[n],
                                                        global_best_position)
                # Move the particles to new position
                particles[n] = update_position(particles[n], particle_velocity[n], position_min, position_max)
        # Calculate fitness
        particle_best_fitness = [fitness_function(p[0], p[1]) for p in particles]
        # Find the index of the best particle
        gbest_index = np.argmin(particle_best_fitness)
        # Update the position of the best particle
        if fitness_function(particles[gbest_index][0], particles[gbest_index][1]) < fitness_function(
                global_best_position[0], global_best_position[1]):
            global_best_position = particles[gbest_index]

        # Add plot for each generation (within the generation for-loop)
        if t < 100:
            image = ax.scatter3D([
                particles[n][0] for n in range(population)],
                [particles[n][1] for n in range(population)],
                [fitness_function(particles[n][0], particles[n][1]) for n in range(population)], c='b')
            images.append([image])

    # Results
    print('Global Best Position: ', global_best_position)
    print('Best Fitness Value: ', min(particle_best_fitness))
    print('Average Particle Best Fitness Value: ', np.average(particle_best_fitness))
    print('Number of Generation: ', t)

    # Generate the animation image and save
    animated_image = animation.ArtistAnimation(fig, images)
    animated_image.save('./'+file_name+'.gif', writer='pillow')
    print('\nGIF saved')
    return global_best_position, min(particle_best_fitness)


if __name__ == "__main__":
    population = 100
    dimension = 2
    position_min = 0
    position_max = 2
    generation = 400
    fitness_criterion = 0.00004

    pso_2d(population, position_min, position_max, generation, fitness_criterion)
