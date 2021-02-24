import math_tools as mt

class clustering():

    def __init__(self, A):
        self.A = A

    def dist_set(self):
        # dtype(A) = list of tuples
        D = []
        D_unsorted = []
        D_sorted = []
        endpoints = []
        endpoints_sorted = []
        # numbers = []
        for k in range(len(self.A)):
            tag = k
            temp = tag
            while tag <= len(self.A) - 2:
                """
                k = 0, idx = tag - temp
                k = 1, idx = len(A) - 1 + tag - temp
                k = 2, idx = len(A) - 1 + len(A) - 2 + tag - temp
                .
                .
                .
                k = k, idx = sum_function + tag - temp = k * len(A) - (1 + k) * k / 2 + tag - temp
                """
                idx = k * len(self.A) - (1 + k) * k / 2 + tag - temp
                tag = tag + 1
                D.append(mt.dist(self.A[k], self.A[tag]))
                # D.append(mt.dist((1, 1), (2, 2)))
                endpoints.append((k, k + tag - temp))
                # numbers.append((self.A[k], self.A[tag]))
        for element in D:
            D_unsorted.append(element)
        # """
        D.sort()
        for element in D:
            D_sorted.append(element)
        for pair in range(len(endpoints)):
            # endpoints_sorted[pair] = endpoints_unsorted[D_unsorted.index(D_sorted[pair])]
            endpoints_sorted.append(endpoints[D_unsorted.index(D_sorted[pair])])
        return D_unsorted, D_sorted, endpoints_sorted
        # """
        # return D_unsorted, len(self.A), endpoints, numbers

    def single_linkage(self, threshold):
        D_unsorted, D_sorted, endpoints_sorted = self.dist_set()
        # cluster = [endpoints(D.index(min(D)))[0], endpoints(D.index(min(D)))[1]]
        """
        in itr0, cluster = [endpoints_sorted[0][0], endpoints_sorted[0][1]]
        in itr1, if endpoints_sorted[1][0] is in cluster:
                    cluster = [end.., end, endpoints_sorted[1][1]]
                 if endpoints_sorted[1][1] is in cluster:
                    cluster = [end.., end, endpoints_sorted[1][0]]
                 else:
                    cluster = [end.., end.., end.., end..]
        """
        for k in range(len(D_sorted)):
            if k == 0:
                cluster = [endpoints_sorted[0][0], endpoints_sorted[0][1]]
            elif D_sorted[k] > threshold: 
                break
            elif endpoints_sorted[k][0] in cluster:
                cluster.append(endpoints_sorted[k][1])
            elif endpoints_sorted[k][1] in cluster:
                cluster.append(endpoints_sorted[k][0])
            else:
                cluster.append(endpoints_sorted[k][0])
                cluster.append(endpoints_sorted[k][1])

        return cluster