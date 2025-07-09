from scipy.spatial import distance as dist

def mouth_aspect_ratio(mouth):
    horizontal = dist.euclidean(mouth[0], mouth[1])
    vertical = dist.euclidean(mouth[2], mouth[3])
    return vertical / horizontal