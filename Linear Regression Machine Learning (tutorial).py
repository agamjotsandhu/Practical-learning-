from numpy import *

def compute_error_for_line_given_points(b, m, points):
    totalError = 0
    for i in range(len(points)):
        # get the x value
        x = points[i, 0]
        # get the y value
        y = points[i, 1]

        # get the difference, square it, add it to the total error
        totalError += (y - (m*x + b))**2

    # get the average
    return totalError / float(len(points))

def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
    # starting b and m
    b = starting_b
    m = starting_m

    # perform gradient descent
    for i in range(num_iterations):
        # update b and m with more accurate b and m
        b, m = step_gradient(b, m, array(points), learning_rate)

    return b, m

def step_gradient(current_b, current_m, points, learning_rate):
    b_gradient = 0
    m_gradient = 0

    for i in range(len(points)):
        x = points[i, 0]
        y = points[i, 1]

        N = float(len(points))
        # direction with respect to b and m
        # computing

        b_gradient += -(2/N) * (y-(current_m*x) + current_b)
        m_gradient += -(2/N) * x * (y-(current_m*x) + current_b)

    # update b and m values using our partial derivaties
    new_b = current_b - (learning_rate * b_gradient) 
    new_m = current_m - (learning_rate * m_gradient)

    return [new_b, new_m]


def run():
    # step 1: collect data
    points = genfromtxt('data.csv', delimiter = ',')

    # step 2: define our hyper parameters
    # how fast should our model converge - balance
    learning_rate = 0.0001

    # y = mx + b
    initial_b = 0
    initial_m = 0

    num_iterations = 1000

    # step 3: train our model

    error = compute_error_for_line_given_points(initial_b, initial_m, points)
    print(f'starting gradient descent b = {initial_b}, m = {initial_m} error = {error}')
    [b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)
    error = compute_error_for_line_given_points(b, m, points)
    print(f'end gradient descent b = {b}, m = {m} error = {error}')


# where the meat of the code goes
if __name__ == "__main__":

    # shows what we're doing at a high level
    run()