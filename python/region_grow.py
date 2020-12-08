import numpy as np
import math
import valid_pixel
import angle_diff


def region_grow(LLA, P, tau, status):
    """
    input:  LLA:            M*N     ---angles at each pixel
            seed pixel P:   2*1     ---pixel to grow from in rc space
            angle tau:      R       ---scalar heper-parameter
            status:         M*N     ---binary mask, used or not_used
    output: region:         2*Var   ---coordinate of pixels in this region
    rc space
    """
    used = 1
    not_used = 0
    M, N = LLA.shape
    region = np.zeros((2, 1), dtype=int)
    added = np.zeros((2, 1), dtype=int)
    region[:, 0] = P
    added[:, 0] = P
    r = P[0]
    c = P[1]
    status[r, c] = used
    theta_region = LLA[r, c]
    Sx = math.cos(theta_region)
    Sy = math.sin(theta_region)

    # eight neighbor pixels are used as offset
    offset = np.array([[-1, -1, -1, 0, 0, 1, 1, 1],
                       [-1,  0, 1, -1, 1, -1, 0, 1]])
    """
        what's offset?
        * * *
        * P *
        * * *
    """
    # BFS for region grow
    has_new_pixel = 1
    while has_new_pixel:
        has_new_pixel = 0
        new_added = np.zeros((2, 1), dtype=int)
        for i in range(len(added[0, :])):
            # has_new_pixel = 0
            pixel = added[:, i]
            for j in range(len(offset[0, :])):
                neighbor = np.array([pixel + offset[:, j]]).T
                if valid_pixel.valid_pixel(neighbor, M, N) and status[int(neighbor[0]), int(neighbor[1])] == not_used:
                    if angle_diff.angle_diff(theta_region, LLA[int(neighbor[0]), int(neighbor[1])]) < tau:
                        # add a new column to region
                        # no issue in angle_diff
                        # the issue is that the size of each region is too small
                        """:cvar
                            neighbor: shape---(2, 1)
                            region:   shape---(2, ...)
                        """
                        # region = np.append(region, neighbor[:, None], axis=1)
                        region = np.concatenate((region, neighbor), axis=1)
                        status[int(neighbor[0]), int(neighbor[1])] = used
                        Sx = Sx + math.cos(LLA[int(neighbor[0]), int(neighbor[1])])
                        Sy = Sy + math.sin(LLA[int(neighbor[0]), int(neighbor[1])])
                        theta_region = math.atan2(Sy, Sx)

                        if len(new_added[0, :]) == 1:
                            new_added = neighbor
                            has_new_pixel = 1
        added = new_added       
        """               
                        new_added = np.concatenate((new_added, neighbor), axis=1)
                        has_new_pixel = 1
        new_added = np.delete(new_added, 0, axis=1)
        added = np.zeros((2, 1), dtype=int)
        added = np.delete(np.concatenate((added, new_added), axis=1), 0, axis=1)
        """
    return region, status
