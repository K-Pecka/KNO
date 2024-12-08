import numpy as np
import tensorflow as tf
import sys

def rotate_point(x, y, alpha):
    alpha_rad = tf.constant(np.radians(alpha), dtype=tf.float32)
    rotation_matrix = tf.stack([[tf.cos(alpha_rad), -tf.sin(alpha_rad)],
                                [tf.sin(alpha_rad), tf.cos(alpha_rad)]])
    point = tf.constant([x, y], dtype=tf.float32)
    rotated_point = tf.linalg.matvec(rotation_matrix, point)
    return rotated_point

x,y,alpha = 5.0,5.0,80

#print(rotate_point(x,y,alpha).numpy())

def solve_linear(a,b):
    return tf.linalg.solve(a,b)

A=tf.constant([[3,1],[4,1]],dtype=tf.float32)
B=tf.constant([[3],[1]],dtype=tf.float32)

#print(solve_linear(A,B).numpy())

def parse_arguments():

    args = sys.argv[1:]
    n = int(len(args) ** 0.5)
    a=tf.constant([float(args[i]) for i in range(n*n)], shape = (n,n),dtype=tf.float32)
    b=tf.constant([float(args[i]) for i in range(n*n,len(args))], shape = (n,1),dtype=tf.float32)

    return a,b

try:
    if len(sys.argv) > 1:
        A,B = parse_arguments()
        print(solve_linear(A,B).numpy())
    else:
        print(solve_linear(A, B).numpy())
except:
        print("Brak rozwiązania takiego układu!")
