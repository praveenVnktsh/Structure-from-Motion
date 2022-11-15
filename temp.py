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
