import numpy as np


def cmat_iso0():
    # Initialize a 4D tensor of size 3x3x3x3
    tensor = np.zeros((3, 3, 3, 3))
    # Populate the tensor with the provided values
    tensor[0, 0, 0, 0] = 2.6666666666666661
    tensor[0, 0, 1, 1] = -1.3333333333333335
    tensor[0, 0, 2, 2] = -1.3333333333333335
    tensor[0, 1, 0, 1] = 2.0
    tensor[0, 1, 1, 0] = 2.0
    tensor[0, 2, 0, 2] = 2.0
    tensor[0, 2, 2, 0] = 2.0

    tensor[1, 0, 0, 1] = 2.0
    tensor[1, 0, 1, 0] = 2.0
    tensor[1, 1, 0, 0] = -1.3333333333333335
    tensor[1, 1, 1, 1] = 2.6666666666666661
    tensor[1, 1, 2, 2] = -1.3333333333333335
    tensor[1, 2, 1, 2] = 2.0
    tensor[1, 2, 2, 1] = 2.0

    tensor[2, 0, 0, 2] = 2.0
    tensor[2, 0, 2, 0] = 2.0
    tensor[2, 1, 1, 2] = 2.0
    tensor[2, 1, 2, 1] = 2.0
    tensor[2, 2, 0, 0] = -1.3333333333333335
    tensor[2, 2, 1, 1] = -1.3333333333333335
    tensor[2, 2, 2, 2] = 2.6666666666666661
    return tensor


def cmat_iso_uni():
    """
    Fixture to create a 4D tensor for isotropic uniaxial stretch with the given values.
    """
    # Initialize a 4D tensor of size 3x3x3x3
    tensor = np.zeros((3, 3, 3, 3))

    # Populate the tensor with the corrected values
    tensor[0, 0, 0, 0] = -8.4133516232281630e-02
    tensor[0, 0, 1, 1] = -2.7160493827160477
    tensor[0, 0, 2, 2] = -2.7160493827160477
    tensor[0, 1, 0, 1] = tensor[0, 1, 1, 0] = 2.1481481481481466
    tensor[0, 2, 0, 2] = tensor[0, 2, 2, 0] = 2.1481481481481466
    tensor[1, 0, 0, 1] = tensor[1, 0, 1, 0] = 2.1481481481481466
    tensor[1, 1, 0, 0] = -2.7160493827160477
    tensor[1, 1, 1, 1] = 146.66666666666652
    tensor[1, 1, 2, 2] = 30.666666666666629
    tensor[1, 2, 2, 1] = tensor[1, 2, 1, 2] = 57.999999999999950
    tensor[2, 1, 2, 1] = tensor[2, 1, 1, 2] = 57.999999999999950
    tensor[2, 0, 0, 2] = tensor[2, 0, 2, 0] = 2.1481481481481466
    tensor[2, 2, 0, 0] = -2.7160493827160477
    tensor[2, 2, 1, 1] = 30.666666666666629
    tensor[2, 2, 2, 2] = 146.66666666666652
    return tensor


def cmat_iso_arbitrary():
    # Initialize a 3x3x3x3 numpy array
    cmat = np.zeros((3, 3, 3, 3))

    # Full tensor data
    data = [
        (1, 1, 1, 1, 0.30193858284457098),
        (1, 1, 1, 2, -1.4370489436583369),
        (1, 1, 1, 3, -1.4435382478426262),
        (1, 1, 2, 1, -1.4370489436583369),
        (1, 1, 2, 2, 2.6354050581880939),
        (1, 1, 2, 3, 3.4990181431837080),
        (1, 1, 3, 1, -1.4435382478426262),
        (1, 1, 3, 2, 3.4990181431837080),
        (1, 1, 3, 3, 2.4260744143504436),
        (1, 2, 1, 1, -1.4370489436583369),
        (1, 2, 1, 2, 8.1421289164817097),
        (1, 2, 1, 3, 5.4210394221196125),
        (1, 2, 2, 1, 8.1421289164817097),
        (1, 2, 2, 2, -31.131706731899918),
        (1, 2, 2, 3, -20.720414983420653),
        (1, 2, 3, 1, 5.4210394221196125),
        (1, 2, 3, 2, -20.720414983420653),
        (1, 2, 3, 3, -19.732406613789706),
        (1, 3, 1, 1, -1.4435382478426262),
        (1, 3, 1, 2, 5.4210394221196125),
        (1, 3, 1, 3, 9.3658760679831481),
        (1, 3, 2, 1, 5.4210394221196125),
        (1, 3, 2, 2, -17.768839217120835),
        (1, 3, 2, 3, -23.629499827717773),
        (1, 3, 3, 1, 9.3658760679831481),
        (1, 3, 3, 2, -23.629499827717773),
        (1, 3, 3, 3, -39.103655664503229),
        (2, 1, 1, 1, -1.4370489436583369),
        (2, 1, 1, 2, 8.1421289164817097),
        (2, 1, 1, 3, 5.4210394221196125),
        (2, 1, 2, 1, 8.1421289164817097),
        (2, 1, 2, 2, -31.131706731899918),
        (2, 1, 2, 3, -20.720414983420653),
        (2, 1, 3, 1, 5.4210394221196125),
        (2, 1, 3, 2, -20.720414983420653),
        (2, 1, 3, 3, -19.732406613789706),
        (2, 2, 1, 1, 2.6354050581880935),
        (2, 2, 1, 2, -31.131706731899918),
        (2, 2, 1, 3, -17.768839217120835),
        (2, 2, 2, 1, -31.131706731899918),
        (2, 2, 2, 2, 185.33923639898384),
        (2, 2, 2, 3, 83.202888729367345),
        (2, 2, 3, 1, -17.768839217120835),
        (2, 2, 3, 2, 83.202888729367345),
        (2, 2, 3, 3, 78.788228348662926),
        (2, 3, 1, 1, 3.4990181431837080),
        (2, 3, 1, 2, -20.720414983420653),
        (2, 3, 1, 3, -23.629499827717773),
        (2, 3, 2, 1, -20.720414983420653),
        (2, 3, 2, 2, 83.202888729367345),
        (2, 3, 2, 3, 113.82682354865585),
        (2, 3, 3, 1, -23.629499827717773),
        (2, 3, 3, 2, 113.82682354865585),
        (2, 3, 3, 3, 104.03898308454664),
        (3, 1, 1, 1, -1.4435382478426262),
        (3, 1, 1, 2, 5.4210394221196125),
        (3, 1, 1, 3, 9.3658760679831481),
        (3, 1, 2, 1, 5.4210394221196125),
        (3, 1, 2, 2, -17.768839217120835),
        (3, 1, 2, 3, -23.629499827717773),
        (3, 1, 3, 1, 9.3658760679831481),
        (3, 1, 3, 2, -23.629499827717773),
        (3, 1, 3, 3, -39.103655664503229),
        (3, 2, 1, 1, 3.4990181431837080),
        (3, 2, 1, 2, -20.720414983420653),
        (3, 2, 1, 3, -23.629499827717773),
        (3, 2, 2, 1, -20.720414983420653),
        (3, 2, 2, 2, 83.202888729367345),
        (3, 2, 2, 3, 113.82682354865585),
        (3, 2, 3, 1, -23.629499827717773),
        (3, 2, 3, 2, 113.82682354865585),
        (3, 2, 3, 3, 104.03898308454664),
        (3, 3, 1, 1, 2.4260744143504436),
        (3, 3, 1, 2, -19.732406613789706),
        (3, 3, 1, 3, -39.103655664503229),
        (3, 3, 2, 1, -19.732406613789706),
        (3, 3, 2, 2, 78.788228348662926),
        (3, 3, 2, 3, 104.03898308454664),
        (3, 3, 3, 1, -39.103655664503229),
        (3, 3, 3, 2, 104.03898308454664),
        (3, 3, 3, 3, 289.84208473072965),
    ]

    for i, j, k, ll, value in data:
        cmat[i - 1, j - 1, k - 1, ll - 1] = value  # Adjust for 0-based indexing

    return cmat


def cmat_iso_arbitrary2():
    """
    Fixture representing a 4D tensor with arbitrary isotropic data.
    The tensor shape corresponds to the 3x3x3x3 structure implied by the provided data.
    """
    data = [
        [1, 1, 1, 1, 0.60472436752719028],
        [1, 1, 1, 2, -3.1494209150312722],
        [1, 1, 1, 3, -1.6438110747420629],
        [1, 1, 2, 1, -3.1494209150312722],
        [1, 1, 2, 2, 10.293213981001511],
        [1, 1, 2, 3, 5.5785936405303111],
        [1, 1, 3, 1, -1.6438110747420629],
        [1, 1, 3, 2, 5.5785936405303111],
        [1, 1, 3, 3, 2.5501822048834013],
        [1, 2, 1, 1, -3.1494209150312722],
        [1, 2, 1, 2, 19.024376838885505],
        [1, 2, 1, 3, 8.0558318916305289],
        [1, 2, 2, 1, 19.024376838885505],
        [1, 2, 2, 2, -89.653784669207511],
        [1, 2, 2, 3, -36.745780464383401],
        [1, 2, 3, 1, 8.0558318916305289],
        [1, 2, 3, 2, -36.745780464383401],
        [1, 2, 3, 3, -19.192671848160998],
        [1, 3, 1, 1, -1.6438110747420629],
        [1, 3, 1, 2, 8.0558318916305289],
        [1, 3, 1, 3, 6.1599002820183753],
        [1, 3, 2, 1, 8.0558318916305289],
        [1, 3, 2, 2, -34.051435541355758],
        [1, 3, 2, 3, -22.501972722776095],
        [1, 3, 3, 1, 6.1599002820183753],
        [1, 3, 3, 2, -22.501972722776095],
        [1, 3, 3, 3, -16.957448329859460],
        [2, 1, 1, 1, -3.1494209150312722],
        [2, 1, 1, 2, 19.024376838885505],
        [2, 1, 1, 3, 8.0558318916305289],
        [2, 1, 2, 1, 19.024376838885505],
        [2, 1, 2, 2, -89.653784669207511],
        [2, 1, 2, 3, -36.745780464383401],
        [2, 1, 3, 1, 8.0558318916305289],
        [2, 1, 3, 2, -36.745780464383401],
        [2, 1, 3, 3, -19.192671848160998],
        [2, 2, 1, 1, 10.293213981001511],
        [2, 2, 1, 2, -89.653784669207511],
        [2, 2, 1, 3, -34.051435541355758],
        [2, 2, 2, 1, -89.653784669207511],
        [2, 2, 2, 2, 557.09229294461079],
        [2, 2, 2, 3, 188.76978511010742],
        [2, 2, 3, 1, -34.051435541355758],
        [2, 2, 3, 2, 188.76978511010742],
        [2, 2, 3, 3, 91.357116815294887],
        [2, 3, 1, 1, 5.5785936405303111],
        [2, 3, 1, 2, -36.745780464383401],
        [2, 3, 1, 3, -22.501972722776092],
        [2, 3, 2, 1, -36.745780464383401],
        [2, 3, 2, 2, 188.76978511010742],
        [2, 3, 2, 3, 119.19089730643179],
        [2, 3, 3, 1, -22.501972722776092],
        [2, 3, 3, 2, 119.19089730643179],
        [2, 3, 3, 3, 68.407412144454938],
        [3, 1, 1, 1, -1.6438110747420629],
        [3, 1, 1, 2, 8.0558318916305289],
        [3, 1, 1, 3, 6.1599002820183753],
        [3, 1, 2, 1, 8.0558318916305289],
        [3, 1, 2, 2, -34.051435541355758],
        [3, 1, 2, 3, -22.501972722776095],
        [3, 1, 3, 1, 6.1599002820183753],
        [3, 1, 3, 2, -22.501972722776095],
        [3, 1, 3, 3, -16.957448329859460],
        [3, 2, 1, 1, 5.5785936405303111],
        [3, 2, 1, 2, -36.745780464383401],
        [3, 2, 1, 3, -22.501972722776092],
        [3, 2, 2, 1, -36.745780464383401],
        [3, 2, 2, 2, 188.76978511010742],
        [3, 2, 2, 3, 119.19089730643179],
        [3, 2, 3, 1, -22.501972722776092],
        [3, 2, 3, 2, 119.19089730643179],
        [3, 2, 3, 3, 68.407412144454938],
        [3, 3, 1, 1, 2.5501822048834013],
        [3, 3, 1, 2, -19.192671848160998],
        [3, 3, 1, 3, -16.957448329859460],
        [3, 3, 2, 1, -19.192671848160998],
        [3, 3, 2, 2, 91.357116815294887],
        [3, 3, 2, 3, 68.407412144454938],
        [3, 3, 3, 1, -16.957448329859460],
        [3, 3, 3, 2, 68.407412144454938],
        [3, 3, 3, 3, 73.087161609769680],
    ]
    tensor = np.zeros((3, 3, 3, 3))
    for i, j, k, ll, value in data:
        tensor[i - 1, j - 1, k - 1, ll - 1] = value
    return tensor


def cmat_vol0():
    # Initialize a 4D tensor of size 3x3x3x3 filled with zeros
    tensor = np.zeros((3, 3, 3, 3))

    # Populate the diagonal elements as per the pattern
    for i in range(3):
        tensor[i, i, 0, 0] = 1000.0
        tensor[i, i, 1, 1] = 1000.0
        tensor[i, i, 2, 2] = 1000.0
    return tensor


def cmat_vol_uni():
    # Initialize a 4D tensor of zeros
    tensor = np.zeros((3, 3, 3, 3))

    # Assign values based on the provided indices and values
    tensor[0, 0, 0, 0] = 12.345679012345677
    tensor[0, 0, 1, 1] = 333.33333333333331
    tensor[0, 0, 2, 2] = 333.33333333333331
    tensor[1, 1, 0, 0] = 333.33333333333331
    tensor[1, 1, 1, 1] = 9000.0
    tensor[1, 1, 2, 2] = 9000.0
    tensor[2, 2, 0, 0] = 333.33333333333331
    tensor[2, 2, 1, 1] = 9000.0
    tensor[2, 2, 2, 2] = 9000.0
    return tensor


def cmat_vol_arbitrary():
    # Initialize a 3x3x3x3 tensor filled with zeros
    tensor = np.zeros((3, 3, 3, 3))

    # Data provided (indexing is 1-based in the data)
    data = [
        (1, 1, 1, 1, 44.445675011086962),
        (1, 1, 1, 2, -108.81092068807229),
        (1, 1, 1, 3, -109.30227984882566),
        (1, 1, 2, 1, -108.81092068807229),
        (1, 1, 2, 2, 646.38849467428531),
        (1, 1, 2, 3, 289.85640344508829),
        (1, 1, 3, 1, -109.30227984882566),
        (1, 1, 3, 2, 289.85640344508829),
        (1, 1, 3, 3, 802.13311745983515),
        (1, 2, 1, 1, -108.81092068807229),
        (1, 2, 1, 2, 274.51822056886755),
        (1, 2, 1, 3, 268.06776652944984),
        (1, 2, 2, 1, 274.51822056886755),
        (1, 2, 2, 2, -1622.2800785174763),
        (1, 2, 2, 3, -730.77910015729617),
        (1, 2, 3, 1, 268.06776652944984),
        (1, 2, 3, 2, -730.77910015729617),
        (1, 2, 3, 3, -1966.1074443718628),
        (1, 3, 1, 1, -109.30227984882568),
        (1, 3, 1, 2, 268.06776652944990),
        (1, 3, 1, 3, 280.20993744243071),
        (1, 3, 2, 1, 268.06776652944990),
        (1, 3, 2, 2, -1591.9523665066845),
        (1, 3, 2, 3, -741.92997282057490),
        (1, 3, 3, 1, 280.20993744243071),
        (1, 3, 3, 2, -741.92997282057490),
        (1, 3, 3, 3, -2028.7527950495598),
        (2, 1, 1, 1, -108.81092068807229),
        (2, 1, 1, 2, 274.51822056886755),
        (2, 1, 1, 3, 268.06776652944984),
        (2, 1, 2, 1, 274.51822056886755),
        (2, 1, 2, 2, -1622.2800785174763),
        (2, 1, 2, 3, -730.77910015729617),
        (2, 1, 3, 1, 268.06776652944984),
        (2, 1, 3, 2, -730.77910015729617),
        (2, 1, 3, 3, -1966.1074443718628),
        (2, 2, 1, 1, 646.38849467428531),
        (2, 2, 1, 2, -1622.2800785174766),
        (2, 2, 1, 3, -1591.9523665066845),
        (2, 2, 2, 1, -1622.2800785174766),
        (2, 2, 2, 2, 9879.5282295227371),
        (2, 2, 2, 3, 4335.7208142542913),
        (2, 2, 3, 1, -1591.9523665066845),
        (2, 2, 3, 2, 4335.7208142542913),
        (2, 2, 3, 3, 11872.770511144739),
        (2, 3, 1, 1, 289.85640344508829),
        (2, 3, 1, 2, -730.77910015729617),
        (2, 3, 1, 3, -741.92997282057490),
        (2, 3, 2, 1, -730.77910015729617),
        (2, 3, 2, 2, 4335.7208142542913),
        (2, 3, 2, 3, 2116.0688457999659),
        (2, 3, 3, 1, -741.92997282057490),
        (2, 3, 3, 2, 2116.0688457999659),
        (2, 3, 3, 3, 5397.6891454291426),
        (3, 1, 1, 1, -109.30227984882568),
        (3, 1, 1, 2, 268.06776652944990),
        (3, 1, 1, 3, 280.20993744243071),
        (3, 1, 2, 1, 268.06776652944990),
        (3, 1, 2, 2, -1591.9523665066845),
        (3, 1, 2, 3, -741.92997282057490),
        (3, 1, 3, 1, 280.20993744243071),
        (3, 1, 3, 2, -741.92997282057490),
        (3, 1, 3, 3, -2028.7527950495598),
        (3, 2, 1, 1, 289.85640344508829),
        (3, 2, 1, 2, -730.77910015729617),
        (3, 2, 1, 3, -741.92997282057490),
        (3, 2, 2, 1, -730.77910015729617),
        (3, 2, 2, 2, 4335.7208142542913),
        (3, 2, 2, 3, 2116.0688457999659),
        (3, 2, 3, 1, -741.92997282057490),
        (3, 2, 3, 2, 2116.0688457999659),
        (3, 2, 3, 3, 5397.6891454291426),
        (3, 3, 1, 1, 802.13311745983515),
        (3, 3, 1, 2, -1966.1074443718630),
        (3, 3, 1, 3, -2028.7527950495596),
        (3, 3, 2, 1, -1966.1074443718630),
        (3, 3, 2, 2, 11872.770511144739),
        (3, 3, 2, 3, 5397.6891454291426),
        (3, 3, 3, 1, -2028.7527950495596),
        (3, 3, 3, 2, 5397.6891454291426),
        (3, 3, 3, 3, 15311.908577538014),
    ]

    # Populate the tensor with the data
    for i, j, k, ll, value in data:
        tensor[i - 1, j - 1, k - 1, ll - 1] = value
    return tensor


def cmat_vol_arbitrary2():
    # Initialize a 3x3x3x3 tensor filled with zeros
    tensor = np.zeros((3, 3, 3, 3))

    # Fill the tensor manually with the provided values
    values = [
        # Format: (i, j, k, l, value)
        (0, 0, 0, 0, 67.508178029118369),
        (0, 0, 0, 1, -229.54442014054172),
        (0, 0, 0, 2, -119.80858391185254),
        (0, 0, 1, 0, -229.54442014054172),
        (0, 0, 1, 1, 1610.5075230282532),
        (0, 0, 1, 2, 502.37867210759879),
        (0, 0, 2, 0, -119.80858391185254),
        (0, 0, 2, 1, 502.37867210759879),
        (0, 0, 2, 2, 615.12752451668928),
        (0, 1, 0, 0, -229.54442014054172),
        (0, 1, 0, 1, 697.23093204981114),
        (0, 1, 0, 2, 397.84701410404227),
        (0, 1, 1, 0, 697.23093204981114),
        (0, 1, 1, 1, -4909.8001398834967),
        (0, 1, 1, 2, -1528.0079339111026),
        (0, 1, 2, 0, 397.84701410404227),
        (0, 1, 2, 1, -1528.0079339111026),
        (0, 1, 2, 2, -2057.7527951098828),
        (0, 2, 0, 0, -119.80858391185254),
        (0, 2, 0, 1, 397.84701410404227),
        (0, 2, 0, 2, 172.24339455425223),
        (0, 2, 1, 0, 397.84701410404227),
        (0, 2, 1, 1, -2793.3911623752092),
        (0, 2, 1, 2, -737.35290453864172),
        (0, 2, 2, 0, 172.24339455425223),
        (0, 2, 2, 1, -737.35290453864172),
        (0, 2, 2, 2, -948.34179591263842),
        (1, 0, 0, 0, -229.54442014054172),
        (1, 0, 0, 1, 697.23093204981114),
        (1, 0, 0, 2, 397.84701410404227),
        (1, 0, 1, 0, 697.23093204981114),
        (1, 0, 1, 1, -4909.8001398834967),
        (1, 0, 1, 2, -1528.0079339111026),
        (1, 0, 2, 0, 397.84701410404227),
        (1, 0, 2, 1, -1528.0079339111026),
        (1, 0, 2, 2, -2057.7527951098828),
        (1, 1, 0, 0, 1610.5075230282532),
        (1, 1, 0, 1, -4909.8001398834958),
        (1, 1, 0, 2, -2793.3911623752092),
        (1, 1, 1, 0, -4909.8001398834958),
        (1, 1, 1, 1, 30885.208280980769),
        (1, 1, 1, 2, 10337.789093444811),
        (1, 1, 2, 0, -2793.3911623752092),
        (1, 1, 2, 1, 10337.789093444811),
        (1, 1, 2, 2, 13430.228675432221),
        (1, 2, 0, 0, 502.37867210759890),
        (1, 2, 0, 1, -1528.0079339111026),
        (1, 2, 0, 2, -737.35290453864195),
        (1, 2, 1, 0, -1528.0079339111026),
        (1, 2, 1, 1, 10337.789093444811),
        (1, 2, 1, 2, 2459.9062512694927),
        (1, 2, 2, 0, -737.35290453864195),
        (1, 2, 2, 1, 2459.9062512694927),
        (1, 2, 2, 2, 3825.6703971537963),
        (2, 0, 0, 0, -119.80858391185254),
        (2, 0, 0, 1, 397.84701410404227),
        (2, 0, 0, 2, 172.24339455425223),
        (2, 0, 1, 0, 397.84701410404227),
        (2, 0, 1, 1, -2793.3911623752092),
        (2, 0, 1, 2, -737.35290453864172),
        (2, 0, 2, 0, 172.24339455425223),
        (2, 0, 2, 1, -737.35290453864172),
        (2, 0, 2, 2, -948.34179591263842),
        (2, 1, 0, 0, 502.37867210759890),
        (2, 1, 0, 1, -1528.0079339111026),
        (2, 1, 0, 2, -737.35290453864195),
        (2, 1, 1, 0, -1528.0079339111026),
        (2, 1, 1, 1, 10337.789093444811),
        (2, 1, 1, 2, 2459.9062512694927),
        (2, 1, 2, 0, -737.35290453864195),
        (2, 1, 2, 1, 2459.9062512694927),
        (2, 1, 2, 2, 3825.6703971537963),
        (2, 2, 0, 0, 615.12752451668928),
        (2, 2, 0, 1, -2057.7527951098828),
        (2, 2, 0, 2, -948.34179591263842),
        (2, 2, 1, 0, -2057.7527951098828),
        (2, 2, 1, 1, 13430.228675432223),
        (2, 2, 1, 2, 3825.6703971537963),
        (2, 2, 2, 0, -948.34179591263842),
        (2, 2, 2, 1, 3825.6703971537963),
        (2, 2, 2, 2, 4229.7071553603892),
    ]

    for i, j, k, ll, value in values:
        tensor[i, j, k, ll] = value

    return tensor


# Define the PK2 tensors isochoric
NULL2 = np.zeros((3, 3))
PK2_ISO0 = NULL2
PK2_ISO_UNI = np.array(
    [
        [1.2839506172839503, 0, 0],
        [0, -17.333333333333325, 0],
        [0, 0, -17.333333333333325],
    ],
)
PK2_ISO_ARBITRARY = np.array(
    [
        [0.47698872071292914, 3.7974644907036370, 3.8146126481050007],
        [3.7974644907036370, -21.098062333064327, -10.149138565098266],
        [3.8146126481050007, -10.149138565098266, -26.762466478176837],
    ],
)
PK2_ISO_ARBITRARY2 = np.array(
    [
        [5.5268336167262878e-002, 6.1233382806947079, 3.1960197886296906],
        [6.1233382806947079, -36.662877397196567, -12.892944575515303],
        [3.1960197886296906, -12.892944575515303, -12.398475045842547],
    ],
)
# Define the PK2 tensors volumetric
PK2_VOL0 = np.zeros((3, 3))
PK2_VOL_UNI = np.array(
    [
        [1.2841973335116454e-010, 0, 0],
        [0, -1.7326671995185574e-009, 0],
        [0, 0, -1.7326671995185574e-009],
    ],
)
PK2_VOL_ARBITRARY = np.array(
    [
        [-4.325244400, 10.58896770, 10.63678470],
        [10.58896770, -64.48578600, -28.30017360],
        [10.63678470, -28.30017360, -80.28059200],
    ],
)
# FIX VALUES
PK2_VOL_ARBITRARY2 = np.array(
    [
        [32.61332040, -110.8933159, -57.87973900],
        [-110.8933159, 697.5769000, 233.4905044],
        [-57.87973900, 233.4905044, 258.1499070],
    ],
)

# Define the material isochoric tangent tensors
CMAT_ISO0 = cmat_iso0()
CMAT_ISO_UNI = cmat_iso_uni()
CMAT_ISO_ARBITRARY = cmat_iso_arbitrary()
CMAT_ISO_ARBITRARY2 = cmat_iso_arbitrary2()
# Define the material volumetric tangent tensors (dummy placeholders)
CMAT_VOL0 = cmat_vol0()
CMAT_VOL_UNI = cmat_vol_uni()
CMAT_VOL_ARBITRARY = cmat_vol_arbitrary()
CMAT_VOL_ARBITRARY2 = cmat_vol_arbitrary2()