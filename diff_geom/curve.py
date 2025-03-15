import numpy as np
import scipy.linalg as la

def curve_length(c, N: int = 1000) -> float:
    """
    Compute the length of a curve defined by a function c: [0,1] -> R^n,
    approximated by summing the distances between consecutive points.
    
    Parameters:
    c: function
       A function that defines the curve.
    N: int
       Number of points to use for the numerical approximation.
    """
    # Generate N evenly spaced points in [0,1]
    t = np.linspace(0, 1, N)
    # Evaluate the curve at these points; assume c(t) returns an array of shape (d, N)
    x = c(t)
    # Compute differences between consecutive points along the last axis
    dx = np.diff(x, axis=1)
    # Compute the Euclidean distance for each segment
    segment_lengths = la.norm(dx, ord=2, axis=0)
    # Sum up the segment lengths to approximate the total curve length
    return np.sum(segment_lengths)

if __name__ == "__main__":
    # Example curve: c(t) = [2*t + 1, -(t**2)]
    c = lambda t: np.array([2*t + 1, -(t**2)])
    print(curve_length(c, N=1000))
