from skimage.morphology import skeletonize
import numpy as np

try:
    from skimage.morphology import skeletonize_3d
except Exception:
    skeletonize_3d = None

def cl_score(v, s):
    """[this function computes the skeleton volume overlap]

    Args:
        v ([bool]): [image]
        s ([bool]): [skeleton]

    Returns:
        [float]: [computed skeleton volume intersection]
    """
    denom = np.sum(s)
    if denom == 0:
        return 0.0
    return np.sum(v * s) / denom


def clDice(v_p, v_l):
    """[this function computes the cldice metric]

    Args:
        v_p ([bool]): [predicted image]
        v_l ([bool]): [ground truth image]

    Returns:
        [float]: [cldice metric]
    """
    if len(v_p.shape)==2:
        tprec = cl_score(v_p,skeletonize(v_l))
        tsens = cl_score(v_l,skeletonize(v_p))
    elif len(v_p.shape)==3:
        if skeletonize_3d is None:
            raise ImportError(
                "skeletonize_3d is unavailable in this scikit-image version. "
                "Install a version providing skeletonize_3d for 3D clDice."
            )
        tprec = cl_score(v_p,skeletonize_3d(v_l))
        tsens = cl_score(v_l,skeletonize_3d(v_p))
    denom = tprec + tsens
    if denom == 0:
        return 0.0
    return 2 * tprec * tsens / denom