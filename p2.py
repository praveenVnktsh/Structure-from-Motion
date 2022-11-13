# Helper functions for this assignment. DO NOT MODIFY!!!
"""
Helper functions.

Written by Chen Kong, 2018.
Modified by Zhengyi (Zen) Luo, 2021
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy
import nbimporter

# Helper functions for this assignment. DO NOT MODIFY!!!
"""
Helper functions.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy
import scipy.optimize

def eightpoint(pts1, pts2, M):
    '''
    Q1.1: Eight Point Algorithm for calculating the fundamental matrix
        Input:  pts1, Nx2 Matrix containing the corresponding points from image1
                pts2, Nx2 Matrix containing the corresponding points from image2
                M, a scalar parameter computed as max (imwidth, imheight)
        Output: F, the fundamental matrix of shape (3, 3)
    
    ***
    HINTS:
    (1) Normalize the input pts1 and pts2 using the matrix T.
    (2) Setup the eight point algorithm's equation.
    (3) Solve for the least square solution using SVD. 
    (4) Use the function `_singularize` (provided) to enforce the singularity condition. 
    (5) Use the function `refineF` (provided) to refine the computed fundamental matrix. 
        (Remember to usethe normalized points instead of the original points)
    (6) Unscale the fundamental matrix

    '''

    F = None # output fundamental matrix 
    N = pts1.shape[0] # Extrating the number of points  
    T = np.diag([1/M, 1/M, 1])
    
    # Converting the points to homogenous coordinates
    pts1_homogenous, pts2_homogenous = toHomogenous(pts1), toHomogenous(pts2)
    pts1_homogenous = np.dot(T, pts1_homogenous.T).T
    pts2_homogenous = np.dot(T, pts2_homogenous.T).T
    # Computing the 3x3 matrix used to normalize corresponding points. 
    
    A = []
    for i in range(len(pts1_homogenous)):
        temp = []
        pt1 = pts1_homogenous[i]
        pt2 = pts2_homogenous[i]

        temp.append(pt1[0] * pt2[0])
        temp.append(pt1[0] * pt2[1])
        temp.append(pt1[0] )
        temp.append(pt1[1] * pt2[0])
        temp.append(pt1[1] * pt2[1])
        temp.append(pt1[1] )
        temp.append(pt2[0])
        temp.append(pt2[1])
        temp.append(1)
        A.append(temp)

    A = np.array(A)

    U, S, V = np.linalg.svd(A)
    F = V[-1].reshape(3, 3).T
    
    F = _singularize(F)  
    
    F = refineF(F, pts1/M, pts2/M)
    F = np.dot(np.dot(T.T, F), T)
    
    F = F/F[2,2] #Finding the unique fundamental matrix by setting the scale to 1. 
    return F




def _epipoles(E):
    U, S, V = np.linalg.svd(E)
    e1 = V[-1, :]
    U, S, V = np.linalg.svd(E.T)
    e2 = V[-1, :]
    return e1, e2

def displayEpipolarF(I1, I2, F):
    matplotlib.use('TkAgg')
    e1, e2 = _epipoles(F)

    sy, sx, _ = I2.shape

    f, [ax1, ax2] = plt.subplots(1, 2, figsize=(12, 9))
    ax1.imshow(I1)
    ax1.set_title('Select a point in this image')
    ax1.set_axis_off()
    ax2.imshow(I2)
    ax2.set_title('Verify that the corresponding point \n is on the epipolar line in this image')
    ax2.set_axis_off()

    while True:
        plt.sca(ax1)
        x, y = plt.ginput(1, timeout=3600, mouse_stop=2)[0]

        xc = x
        yc = y
        v = np.array([xc, yc, 1])
        l = F.dot(v)
        s = np.sqrt(l[0]**2+l[1]**2)


        l = l/s

        if l[0] != 0:
            ye = sy-1
            ys = 0
            xe = -(l[1] * ye + l[2])/l[0]
            xs = -(l[1] * ys + l[2])/l[0]
        else:
            xe = sx-1
            xs = 0
            ye = -(l[0] * xe + l[2])/l[1]
            ys = -(l[0] * xs + l[2])/l[1]

        # plt.plot(x,y, '*', 'MarkerSize', 6, 'LineWidth', 2);
        ax1.plot(x, y, '*', markersize=6, linewidth=2)
        ax2.plot([xs, xe], [ys, ye], linewidth=2)
        plt.draw()


def _singularize(F):
    U, S, V = np.linalg.svd(F)
    S[-1] = 0
    F = U.dot(np.diag(S).dot(V))
    return F

def _objective_F(f, pts1, pts2):
    F = _singularize(f.reshape([3, 3]))
    num_points = pts1.shape[0]
    hpts1 = np.concatenate([pts1, np.ones([num_points, 1])], axis=1)
    hpts2 = np.concatenate([pts2, np.ones([num_points, 1])], axis=1)
    Fp1 = F.dot(hpts1.T)
    FTp2 = F.T.dot(hpts2.T)

    r = 0
    for fp1, fp2, hp2 in zip(Fp1.T, FTp2.T, hpts2):
        r += (hp2.dot(fp1))**2 * (1/(fp1[0]**2 + fp1[1]**2) + 1/(fp2[0]**2 + fp2[1]**2))
    return r

def refineF(F, pts1, pts2):
    f = scipy.optimize.fmin_powell(
        lambda x: _objective_F(x, pts1, pts2), F.reshape([-1]),
        maxiter=100000,
        maxfun=10000,
        disp = False
    )
    return _singularize(f.reshape([3, 3]))

def camera2(E):
    U,S,V = np.linalg.svd(E)
    m = S[:2].mean()
    E = U.dot(np.array([[m,0,0], [0,m,0], [0,0,0]])).dot(V)
    U,S,V = np.linalg.svd(E)
    W = np.array([[0,-1,0], [1,0,0], [0,0,1]])

    if np.linalg.det(U.dot(W).dot(V))<0:
        W = -W

    M2s = np.zeros([3,4,4])
    M2s[:,:,0] = np.concatenate([U.dot(W).dot(V), U[:,2].reshape([-1, 1])/abs(U[:,2]).max()], axis=1)
    M2s[:,:,1] = np.concatenate([U.dot(W).dot(V), -U[:,2].reshape([-1, 1])/abs(U[:,2]).max()], axis=1)
    M2s[:,:,2] = np.concatenate([U.dot(W.T).dot(V), U[:,2].reshape([-1, 1])/abs(U[:,2]).max()], axis=1)
    M2s[:,:,3] = np.concatenate([U.dot(W.T).dot(V), -U[:,2].reshape([-1, 1])/abs(U[:,2]).max()], axis=1)
    return M2s


def calc_epi_error(pts1_homo, pts2_homo, F):
    '''
    Helper function to calculate the sum of squared distance between the corresponding points and the estimated epipolar lines. 
    Expect pts1 and pts2 are in homogeneous coordinates and not normalized. 
    '''
    line1s = pts1_homo.dot(F.T)
    dist1 = np.square(np.divide(np.sum(np.multiply(
        line1s, pts2_homo), axis=1), np.linalg.norm(line1s[:, :2], axis=1)))

    line2s = pts2_homo.dot(F)
    dist2 = np.square(np.divide(np.sum(np.multiply(
        line2s, pts1_homo), axis=1), np.linalg.norm(line2s[:, :2], axis=1)))

    ress = (dist1 + dist2).flatten()
    return ress


def toHomogenous(pts):
    return np.vstack([pts[:,0],pts[:,1],np.ones(pts.shape[0])]).T.copy()

def epipolarMatchGUI(I1, I2, F):
    matplotlib.use('TkAgg')
    e1, e2 = _epipoles(F)

    sy, sx, _ = I2.shape

    f, [ax1, ax2] = plt.subplots(1, 2, figsize=(12, 9))
    ax1.imshow(I1)
    ax1.set_title('Select a point in this image')
    ax1.set_axis_off()
    ax2.imshow(I2)
    ax2.set_title('Verify that the corresponding point \n is on the epipolar line in this image')
    ax2.set_axis_off()

    while True:
        plt.sca(ax1)
        x, y = plt.ginput(1, mouse_stop=2)[0]

        xc = int(x)
        yc = int(y)
        v = np.array([xc, yc, 1])
        l = F.dot(v)
        s = np.sqrt(l[0]**2+l[1]**2)


        l = l/s;

        if l[0] != 0:
            ye = sy-1
            ys = 0
            xe = -(l[1] * ye + l[2])/l[0]
            xs = -(l[1] * ys + l[2])/l[0]
        else:
            xe = sx-1
            xs = 0
            ye = -(l[0] * xe + l[2])/l[1]
            ys = -(l[0] * xs + l[2])/l[1]

        # plt.plot(x,y, '*', 'MarkerSize', 6, 'LineWidth', 2);
        ax1.plot(x, y, '*', MarkerSize=6, linewidth=2)
        ax2.plot([xs, xe], [ys, ye], linewidth=2)

        # draw points
        x2, y2 = epipolarCorrespondence(I1, I2, F, xc, yc)
        ax2.plot(x2, y2, 'ro', MarkerSize=8, linewidth=2)
        plt.draw()

def plot_3D(P):
    matplotlib.use('TkAgg')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(P[:,0], P[:,1], P[:,2])
    # ax.set_zlim([-100, 100])
    # ax.set_xlim([-5, 5])
    # ax.set_ylim([-5, 5])
    plt.show()
    # while True:
    #     x, y = plt.ginput(1, mouse_stop=2)[0]
    #     plt.draw()

        

def essentialMatrix(F, K1, K2):
    '''
    Q1.1: Compute the essential matrix E given the fundamental matrix and camera intrinsics
        Input:  F, fundamental matrix
                K1, internal camera calibration matrix of camera 1
                K2, internal camera calibration matrix of camera 2
        Output: E, the essential matrix
    '''
    # ----- TODO -----
    # YOUR CODE HERE
    E = np.dot(K1.T, np.dot(F, K2))
    return E


def triangulate(C1, pts1, C2, pts2):
    '''
    Q2.2: Triangulate a set of 2D coordinates in the image to a set of 3D points.
        Input:  C1, the 3x4 camera matrix
                pts1, the Nx2 matrix with the 2D image coordinates per row
                C2, the 3x4 camera matrix
                pts2, the Nx2 matrix with the 2D image coordinates per row
        Output: P, the Nx3 matrix with the corresponding 3D points per row
                err, the reprojection error.
    
    ***
    Hints:
    (1) For every input point, form A using the corresponding points from pts1 & pts2 and C1 & C2
    (2) Solve for the least square solution using np.linalg.svd
    (3) Calculate the reprojection error using the calculated 3D points and C1 & C2 (do not forget to convert from 
        homogeneous coordinates to non-homogeneous ones)
    (4) Keep track of the 3D points and projection error, and continue to next point 
    (5) You do not need to follow the exact procedure above. 
    '''
    # ----- TODO -----
    # YOUR CODE HERE
    p1 = C1[0]
    p2 = C1[1]
    p3 = C1[2]

    p1t = C2[0]
    p2t = C2[1]
    p3t = C2[2]

    A = []
    P = []
    err = np.zeros((len(pts1), )) 
    for i in range(len(pts1)):
        x1, y1 = pts1[i]
        x2, y2 = pts2[i]

        A = np.array(
            [
                [y1*p3 - p2],
                [p1 - x1*p3],
                [y2*p3t - p2t],
                [p1t - x2*p3t]
            ]
        ).squeeze()


        U, S, V = np.linalg.svd(A)
        val = V[-1]
        val /= val[3]

        proj1 = np.dot(C1, val)
        proj1 /= proj1[2]

        proj2 = np.dot(C2, val)
        proj2 /= proj2[2]
        err[i] =np.linalg.norm((np.array([x1, y1, 1]) - proj1 )) ** 2  +\
              np.linalg.norm((np.array([x2, y2, 1]) - proj2)) ** 2

        P.append(val)
        

    P = np.array(P)

    err = np.sum(err)

    return P, err
    

def find_M2(F, pts1, pts2, intrinsics):
    '''
    Q2.2: Function to find the camera2's projective matrix given correspondences
        Input:  F, the pre-computed fundamental matrix
                pts1, the Nx2 matrix with the 2D image coordinates per row
                pts2, the Nx2 matrix with the 2D image coordinates per row
                intrinsics, the intrinsics of the cameras, load from the .npz file
        Output: [M2, C2, P] the computed M2 (3x4) camera projective matrix, C2 (3x4) K2 * M2, and the 3D points P (Nx3)
    
    ***
    Hints:
    (1) Loop through the 'M2s' and use triangulate to calculate the 3D points and projection error. Keep track 
        of the projection error through best_error and retain the best one. 
    (2) Remember to take a look at camera2 to see how to correctly reterive the M2 matrix from 'M2s'. 

    '''
    K1 = intrinsics["K1"]
    K2 = intrinsics["K2"]
    E = essentialMatrix(F, K1, K2)
    M1 = np.hstack((np.identity(3), np.zeros(3)[:,np.newaxis]))
    C1 = K1.dot(M1)
    M2s = camera2(E)
    best_error = np.finfo('float').max
    # ----- TODO -----
    # YOUR CODE HERE
    M_final, C_final, P_final = None, None, None
    # for i in range(M2s.shape[-1]):
    M2 = M2s[:,:,0]
    C2 = K2.dot(M2)    
    P, err = triangulate(C1, pts1, C2, pts2)
    if err < best_error:
        M_final = M2
        C_final = C2
        P_final = P
        best_error = err


    
    print(f"Best Error {best_error}")
    return M_final, C_final, P_final
    
def epipolarCorrespondence(im1, im2, F, x1, y1):
    '''
    Q2.3 3D visualization of the temple images.
        Input:  im1, the first image
                im2, the second image
                F, the fundamental matrix
                x1, x-coordinates of a pixel on im1
                y1, y-coordinates of a pixel on im1
        Output: x2, x-coordinates of the pixel on im2
                y2, y-coordinates of the pixel on im2
    
    ***
    Hints:
    (1) Given input [x1, x2], use the fundamental matrix to recover the corresponding epipolar line on image2
    (2) Search along this line to check nearby pixel intensity (you can define a search window) to 
        find the best matches
    (3) Use guassian weighting to weight the pixel simlairty
    
    '''
    # ----- TODO -----
    # YOUR CODE HERE
    window_size = 15

    l_prime = F @ np.array([[x1], [y1], [1]])
    
    get_x_coord = lambda y: (-l_prime[1]/l_prime[0]) * y - l_prime[2]/l_prime[0]

    x2 = []
    y2 = []

    lower_bound = int(window_size//2)
    upper_bound = im2.shape[0] - window_size//2 - 1

    min_val = float("inf")
    x2, y2 = -1, -1

    for y in range(lower_bound, upper_bound):
        x = get_x_coord(y)
        x = int(np.round(x))

        offset = int(window_size // 2)
        sad = np.sum(np.abs(im1[y1 - offset: y1 + offset + 1, x1 - offset: x1 + offset + 1].flatten() - im2[y - offset: y + offset + 1, x - offset: x + offset + 1].flatten()))

        if sad < min_val:
            min_val = sad
            x2 = x
            y2 = y


    return x2, y2

def compute3D_pts(pts1, intrinsics, F, im1, im2):
    '''
    Q2.4: Finding the 3D position of given points based on epipolar correspondence and triangulation
        Input:  pts1, chosen points from im1
                intrinsics, the intrinsics dictionary for calling epipolarCorrespondence
                F, the fundamental matrix
                im1, the first image
                im2, the second image
        Output: P (Nx3) the recovered 3D points
    
    ***
    Hints:
    (1) Use epipolarCorrespondence to find the corresponding point for [x1 y1] (find [x2, y2])
    (2) Now you have a set of corresponding points [x1, y1] and [x2, y2], you can compute the M2
        matrix and use triangulate to find the 3D points. 
    (3) Use the function findM2 to find the 3D points P (do not recalculate fundamental matrices)
    (4) As a reference, our solution's bet error is around ~2000 on the 3D points. 
    '''
    x1s_temple, y1s_temple = pts1[:, 0], pts1[:, 1]
    P = []
    pts1 = []
    pts2 = []
    for i in range(len(x1s_temple)):
        x1, y1 = x1s_temple[i], y1s_temple[i]
        x2, y2 = epipolarCorrespondence(im1, im2, F, x1, y1)
        pts1.append([x1, y1])
        pts2.append([x2, y2])

    M2, C2, P = find_M2(F, np.array(pts1), np.array(pts2), intrinsics)
    print(P)
    return P
correspondence = np.load('data/some_corresp.npz') # Loading correspondences
intrinsics = np.load('data/intrinsics.npz') # Loading the intrinscis of the camera
templeCoords = np.load("data/templeCoords.npz")
pts1, pts2 = correspondence['pts1'], correspondence['pts2']
im1, im2 = plt.imread('data/im1.png'), plt.imread('data/im2.png')

F = eightpoint(pts1, pts2, M=np.max([*im1.shape, *im2.shape]))

pts1 = np.hstack([templeCoords["x1"], templeCoords["y1"]])
P = compute3D_pts(pts1, intrinsics, F, im1, im2)

plot_3D(P)
