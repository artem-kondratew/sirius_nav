from misc_tools import *

''' performs closest point matching of two point sets
      
    Arguments:
    q -- reference point set
    p -- point set to be matched with the reference
    
    Output:
    p_matched -- reordered p, so that the elements in p match the elements in q
'''


def closest_point_matching(q, p):

    q_used = np.zeros(q.shape[1])

    p_matched = np.zeros_like(p)

    for i in range(p.shape[1]):
        dist = np.inf
        idx = -1
        for j in range(q.shape[1]):
            new_dist = np.linalg.norm(p[:, i] - q[:, j])
            if new_dist < dist and q_used[j] != 1:
                dist = new_dist
                idx = j
        p_matched[:, idx] = p[:, i]
        q_used[idx] = 1
    
    return p_matched


def icp(q, p, do_matching):
    print(p.shape)
    p0 = p
    for i in range(10):
        # calculate RMSE
        rmse = 0
        for j in range(p.shape[1]):
            rmse += math.pow(p[0, j] - q[0, j], 2) + math.pow(p[1, j] - q[1, j], 2)
        rmse = math.sqrt(rmse / p.shape[1])

        # print and plot
        print("Iteration:", i, " RMSE:", rmse)
        plot_icp(q, p, p0, i, rmse)

        # data association
        if do_matching:
            p = closest_point_matching(q, p)

        # subtract center of mass
        mq = np.transpose([np.mean(q, 1)])
        mp = np.transpose([np.mean(p, 1)])
        q_prime = q - mq
        p_prime = p - mp

        # singular value decomposition
        H = np.dot(p_prime, q_prime.T)
        U, S, Vt = np.linalg.svd(H)

        print(U)

        # calculate rotation and translation
        R = np.dot(Vt.T, U.T)

        if np.linalg.det(R) < 0:
            U[:, -1] *= -1
            R = np.dot(Vt.T, U.T)

        t = mq - np.dot(R, mp)

        # apply transformation
        p = np.dot(R, p) + t

def main():
    x, p1, p2, p3, p4 = generate_data()

    # icp(x, p1, False)
    # icp(x, p2, False)
    icp(x, p3,  True)
    # icp(x, p4, True)

    plt.waitforbuttonpress()


if __name__ == "__main__":
    main()
