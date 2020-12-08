def valid_pixel(p, m, n):
    # check whether pixel = img[r, c] is a valid pixel
    rst = 1
    r = int(p[0])
    c = int(p[1])
    if r < 0 or c < 0 or r > m-1 or c > n-1:
        rst = 0
    return rst
