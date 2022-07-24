import numpy as np

def solve_image_ldl3(A11, A12, A13, A22, A23, A33, b1, b2, b3):
  # An unrolled LDL solver for a 3x3 symmetric linear system.
    d1 = A11
    L12 = A12/d1
    d2 = A22 - L12*A12
    L13 = A13/d1
    L23 = (A23 - L13*A12)/d2
    d3 = A33 - L13*A13 - L23*L23*d2
    y1 = b1
    y2 = b2 - L12*y1
    y3 = b3 - L13*y1 - L23*y2
    x3 = y3/d3
    x2 = y2/d2 - L23*x3
    x1 = y1/d1 - L12*x2 - L13*x3
    return x1, x2, x3

def planar_filter(Z, filt, eps):
    xy_shape = np.array(Z.shape[-2:])
    xy_scale = 2 / np.mean(xy_shape-1)  # Scaling the x, y coords to be in ~[0, 1]
    x, y = np.meshgrid(*[(np.arange(s) - (s-1)/2) * xy_scale for s in xy_shape], indexing='ij')
    [F1, Fx, Fy, Fz, Fxx, Fxy, Fxz, Fyy, Fyz] = [
        filt(t) for t in [
        np.ones_like(x), x, y, Z, x**2, x*y, x*Z, y**2, y*Z]]
    A11 = F1*x**2 - 2*x*Fx + Fxx + eps**2
    A22 = F1*y**2 - 2*y*Fy + Fyy + eps**2
    A12 = F1*y*x - x*Fy - y*Fx + Fxy
    A13 = F1*x - Fx
    A23 = F1*y - Fy
    A33 = F1# + eps**2
    b1 = Fz*x - Fxz
    b2 = Fz*y - Fyz
    b3 = Fz
    Zx, Zy, Zz = solve_image_ldl3(A11, A12, A13, A22, A23, A33, b1, b2, b3)
    return -Zx*xy_scale, -Zy*xy_scale, Zz

def blur(X, alpha):
  # Do an exponential decay filter on the outermost two dimensions of X.
  # Equivalent to convolving an image with a Laplacian blur.
    Y = X.copy()
    for i in range(Y.shape[-1]-1):
        Y[...,i+1] += alpha * Y[...,i]

    for i in range(Y.shape[-1]-1)[::-1]:
        Y[...,i] += alpha * Y[...,i+1]

    for i in range(Y.shape[-2]-1):
        Y[...,i+1,:] += alpha * Y[...,i,:]

    for i in range(Y.shape[-2]-1)[::-1]:
        Y[...,i,:] += alpha * Y[...,i+1,:]
    return Y


def run_filter(filename):
    depth = np.load(filename)
    mask = depth != 0
    W = np.float32(mask)
    # Define a blur function.
    alpha = 0.5
    filt = lambda x : blur(x * W, alpha) / blur(W, alpha)
    Zx0, Zy0, Zf_recon = planar_filter(depth, filt, 1e15)
    np.save(filename, Zf_recon)
    return 0