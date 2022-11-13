from unittest import TestCase
from numpy import exp
from numpy import sqrt
from numpy import pi, e, sin, cos, absolute
import unittest

from Particle import pso_2d


def simple_unimodal(x1, x2):
    return x1 ** 2.0 + x2 ** 2.0


def complex_multimodal(x1, x2):
    return -20.0 * exp(-0.2 * sqrt(0.5 * (x1 ** 2 + x2 ** 2))) - exp(
        0.5 * (cos(2 * pi * x1) + cos(2 * pi * x2))) + e + 20

def geogebra(x1, x2):
    return sin(x1) + sin(x2) + 0.01*(absolute(x1) + absolute(x2))**2 + 1.9


def wavy_multimodal(x1, x2):
    return (x1 ** 2 + x2 - 11) ** 2 + (x1 + x2 ** 2 - 7) ** 2


def multiple_mins_multimodal(x1, x2):
    return -absolute(sin(x1) * cos(x2) * exp(absolute(1 - (sqrt(x1 ** 2 + x2 ** 2) / pi)))) + 19.2


class Test(TestCase):
    def test_animation_simple_unimodal(self):
        text = ' Simple Unimodal Function test '
        print('\n\n' + 10 * "_" + text + 10 * "_")
        population = 100
        dimension = 2
        position_min = -4
        position_max = 4
        generation = 400
        fitness_criterion = 0.00004

        pso_2d(population, position_min, position_max, generation, fitness_criterion, simple_unimodal,
               'TestingAnimations/testSimpleUnimodal')

        self.assertEqual(1, 1)

    def test_animation_complex_multimodal(self):
        text = ' Ackley\'s Function test '
        print('\n\n' + 10 * "_" + text + 10 * "_")
        population = 100
        dimension = 2
        position_min = -4
        position_max = 4
        generation = 400
        fitness_criterion = 0.00004

        pso_2d(population, position_min, position_max, generation, fitness_criterion, complex_multimodal,
               'TestingAnimations/testComplexMultimodal')
        self.assertEqual(1, 1)


    def test_animation_geogebra(self):
        text = ' Geogebra Function test '
        print('\n\n' + 10 * "_" + text + 10 * "_")
        population = 100
        dimension = 2
        position_min = -6
        position_max = 6
        generation = 400
        fitness_criterion = 0.00004

        pso_2d(population, position_min, position_max, generation, fitness_criterion, geogebra,
               'TestingAnimations/testGeogebra')
        self.assertEqual(1, 1)

    def test_animation_wavy_multimodal(self):
        text = ' Himmelblau\'s Function test '
        print('\n\n' + 10 * "_" + text + 10 * "_")
        population = 100
        dimension = 2
        position_min = -4
        position_max = 4
        generation = 400
        fitness_criterion = 0.00004

        pso_2d(population, position_min, position_max, generation, fitness_criterion, wavy_multimodal,
               'TestingAnimations/testWavyMultimodal')
        self.assertEqual(1, 1)

    def test_animation_multiple_mins_multimodal(self):
        text = ' Holder\'s Table Function test '
        print('\n\n' + 10 * "_" + text + 10 * "_")
        population = 100
        dimension = 2
        position_min = -10
        position_max = 10
        generation = 400
        fitness_criterion = 0.00004

        pso_2d(population, position_min, position_max, generation, fitness_criterion, multiple_mins_multimodal,
               'TestingAnimations/testMultipleMinsMultimodal')
        self.assertEqual(1, 1)
