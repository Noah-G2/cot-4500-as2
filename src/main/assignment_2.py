import numpy as np

def nevilles_method(x_points, y_points, x):
    # must specify the matrix size (this is based on how many columns/rows you want)
    matrix = np.zeros((3,3))
    # fill in value (just the y values because we already have x set)
    for counter, row in enumerate(matrix):
        row[0] = y_points[counter]
    # the end of the first loop are how many columns you have...
    num_of_points = len (x_points)
    # populate final matrix (this is the iterative version of the recursion explained in class)
    # the end of the second loop is based on the first loop...
    for i in range(1, num_of_points):
        
        for j in range(1,i+1):
            
            first_multiplication = (x - x_points[i-j]) * matrix[i][j-1]
            #print(first_multiplication)
            second_multiplication = (x - x_points[i]) * matrix[i-1][j-1]
            #print(second_multiplication)
            denominator = x_points[i] - x_points[i-j]
            #print(denominator)
            # this is the value that we will find in the matrix
            coefficient = (first_multiplication - second_multiplication)/denominator
            #print(coefficient)
            matrix[i][j] = coefficient
            
    print(matrix[len(x_points)-1][len(x_points)-1])
    
    return matrix

def divided_difference_table(x_points, y_points):
    # set up the matrix
    size: int = len(x_points)
        
    matrix: np.array = np.zeros((4,4))
    # fill the matrix
    for index, row in enumerate(matrix):
        
        row[0] = y_points[index]
    # populate the matrix (end points are based on matrix size and max operations we're using)
    for i in range(1, size):
        
        for j in range(1, i+1):
            # the numerator are the immediate left and diagonal left indices...
            numerator = matrix[i][j-1] - matrix[i-1][j-1]
            # the denominator is the X-SPAN...
            denominator = x_points[i] - x_points[i-j]
            operation = numerator / denominator
            #print(operation)
            # cut it off to view it more simpler
            matrix[i][j] = (operation)
    #print(matrix)
    return matrix

def get_approximate_result(matrix, x_points, value):
    # p0 is always y0 and we use a reoccuring x to avoid having to recalculate x 
    reoccuring_x_span = 1
    reoccuring_px_result = matrix[0][0]
    
    # we only need the diagonals...and that starts at the first row...
    for index in range(1, len(x_points)):
        #print(index)
        polynomial_coefficient = matrix[index][index]
        # we use the previous index for x_points....
        reoccuring_x_span *= (value - x_points[index-1])
        #print(reoccuring_x_span)
        
        # get a_of_x * the x_span
        mult_operation = polynomial_coefficient * reoccuring_x_span
        # add the reoccuring px result
        reoccuring_px_result += mult_operation
    
    # final result
    return reoccuring_px_result

np.set_printoptions(precision=7, suppress=True, linewidth=100
                   )
def apply_div_dif(matrix: np.array):
    size = len(matrix)
    for i in range(2, size):
        for j in range(2, i+2):
            # skip if value is prefilled (we dont want to accidentally recalculate...)
            if j >= len(matrix[i]) or matrix[i][j] != 0:
                continue
            
            # get left cell entry
            #print(matrix[i][j-1])
            left: float = matrix[i][j-1]
            # get diagonal left entry
            diagonal_left: float = matrix[i-1][j-1]
            # order of numerator is SPECIFIC.
            numerator: float = ( left - diagonal_left )
            # denominator is current i's x_val minus the starting i's x_val....
            denominator = matrix[i][0]- matrix[i-j+1][0]
            # something save into matrix
            operation = numerator / denominator
            matrix[i][j] = operation
    
    return matrix

def hermite_interpolation():
    x_points = [3.6, 3.8, 3.9]
    y_points = [1.675, 1.436, 1.318]
    slopes = [-1.195, -1.188, -1.182]
    # matrix size changes because of "doubling" up info for hermite 
    num_of_points = 2*len(x_points)
    matrix = np.zeros((num_of_points, num_of_points))
    # populate x values (make sure to fill every TWO rows)
    counter = 0
    #print(matrix)
    for x in range(0, num_of_points,2):
        matrix[x][0] = x_points[counter]
        matrix[x+1][0] = x_points[counter]
        counter +=1
    # prepopulate y values (make sure to fill every TWO rows)
    counter = 0
    for x in range(0,num_of_points, 2):
        matrix[x][1] = y_points[counter]
        matrix[x+1][1] = y_points[counter]
        counter +=1
    #prepopulate with derivates (make sure to fill every TWO rows. starting row CHANGES.)
    counter = 0
    for x in range(1,num_of_points,2):
        matrix[x][2] = slopes[counter]
        counter +=1
    
    filled_matrix = apply_div_dif(matrix)
    print(filled_matrix)

def CubicSplineInterpolation ():
    
    x_points = np.array([2, 5, 8, 10])
    y_points = np.array([3, 5, 7, 9])

    size = len(x_points)

    Matrix = np.zeros((size, size))
    #These are given and constant 
    Matrix[0, 0] = 1
    Matrix[size-1, size-1] = 1
    
    for i in range(1, size-1):
        Matrix[i, i-1] = x_points[i] - x_points[i-1]
        Matrix[i, i] = 2 * (x_points[i+1] - x_points[i-1])
        Matrix[i, i+1] = x_points[i+1] - x_points[i]

    Array = np.zeros(size)
    
    for i in range(1, size-1):
        Array[i] = 3 * (y_points[i+1] - y_points[i]) / (x_points[i+1] - x_points[i]) - \
        3 * (y_points[i] - y_points[i-1]) / (x_points[i] - x_points[i-1])

    XArray = np.linalg.solve(Matrix, Array)
    
    print(Matrix)
    print("\n")
    print(Array)
    print("\n")
    print(XArray)
    
if __name__ == "__main__":
    
    # point setup
    
    x_points = [3.6, 3.8, 3.9]
    y_points = [1.675, 1.436, 1.318]
    approximating_value = 3.7
    
    nevilles_method(x_points, y_points, approximating_value)
    print("\n")
    
    # point setup for number 2
    x_points = [7.2, 7.4, 7.5, 7.6]
    y_points = [23.5492, 25.3913, 26.8224, 27.4589]
    divided_table = divided_difference_table(x_points, y_points)
    
    # find approximation
    approximating_x = 7.3
    final_approximation = get_approximate_result(divided_table, x_points, approximating_x)
    
    AnswerArray = []
    for i in range(1, len(x_points)):
        AnswerArray.append(divided_table[i][i])

    print(AnswerArray)
    print("\n")
    print(final_approximation)
    print("\n")
    hermite_interpolation()
    print("\n")
    CubicSplineInterpolation()
