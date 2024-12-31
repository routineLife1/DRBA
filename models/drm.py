from models.utils.tools import *

if check_cupy_env():
    from models.softsplat.softsplat import softsplat as warp
else:
    print("System does not have CUDA installed, falling back to PyTorch")
    from models.softsplat.softsplat_torch import softsplat as warp


def calc_drm_gmfss(flow10, flow12, metric10, metric12):
    warp_method = 'soft' if (metric10 is not None and metric12 is not None) else 'avg'
    # Compute the distance using the optical flow and distance calculator
    d10 = distance_calculator(flow10)
    d12 = distance_calculator(flow12)

    # Calculate the distance ratio map
    drm10 = d10 / (d10 + d12)
    drm12 = d12 / (d10 + d12)

    # Theoretically, the distance ratio can only be controlled by flow emitted from the center to sides.
    drm01_unaligned = 1 - drm10
    drm21_unaligned = 1 - drm12

    # The distance ratio map (drm) is initially aligned with I1.
    # To align it with I0 and I2, we need to warp the drm maps.
    # Note: To reverse the direction of the drm map, use 1 - drm and then warp it.
    drm01 = warp(drm01_unaligned, flow10, metric10, warp_method)
    drm21 = warp(drm21_unaligned, flow12, metric12, warp_method)

    # Create a mask with all ones to identify the holes in the warped drm maps
    ones_mask = torch.ones_like(drm01, device=drm01.device)

    # Warp the ones mask
    warped_ones_mask01 = warp(ones_mask, flow10, metric10, warp_method)
    warped_ones_mask21 = warp(ones_mask, flow12, metric12, warp_method)

    # Identify holes in warped drm map
    holes01 = warped_ones_mask01 < 0.999
    holes21 = warped_ones_mask21 < 0.999

    # Fill the holes in the warped drm maps with the inverse of the original drm maps
    drm01[holes01] = drm01_unaligned[holes01]
    drm21[holes21] = drm21_unaligned[holes21]

    return drm01, drm10, drm12, drm21


def get_drm_t(drm, _t, precision=1e-3):
    """
    DRM is a tensor with dimensions b, 1, h, w, where for any value x (0 < x < 1).
    We define the timestep of the entire DRM tensor as 0.5, want the entire DRM approach t,
    but during this process, all values in the tensor must maintain their original proportions.

    Example:
        Input:
            drm = [0.1, 0.7, 0.4, 0.2, ...]
            init_t = 0.5
            target_t = 0.8

        Iteration Outputs:
            [0.1900(0.1 + (1 - 0.1) * 0.1), 0.9100, ...]  t = 0.5 + (1 - 0.5) * 0.5 = 0.75\n
            [0.2710(0.19 + (1 - 0.19) * 0.1), 0.9730, ...]  t = 0.75 + (1 - 0.75) * 0.5 = 0.875\n
            [0.2629(0.271 - (0.271 - 0.19) * 0.1), 0.9289, ...]  t = 0.875 - (0.875 - 0.75) * 0.5 = 0.8125\n
            [0.2556(0.2629 - (0.2629 - 0.19) * 0.1), 0.9157, ...]  t = 0.8125 - (0.8125 - 0.75) * 0.5 = 0.78125\n
            [0.2563(0.2556 + (0.2629 - 0.2556) * 0.1), 0.9249, ...]  t = 0.78125 + (0.8125 - 0.78125) * 0.5 = 0.796875\n
            [0.2570(0.2563 + (0.2629 - 0.2563) * 0.1), 0.9277, ...]  t = 0.796875 + (0.8125 - 0.796875) * 0.5 = 0.8046875\n
            ...

        Final Output:
            drm_t = [0.2569, 0.9258, 0.7106, 0.4486, ...]  t = 0.80078125
    """
    dtype = drm.dtype

    _x, b = 0.5, 0.5
    l, r = 0, 1

    # float is suggested for drm calculation to avoid overflow
    x_drm, b_drm = drm.float().clone(), drm.float().clone()
    l_drm, r_drm = torch.zeros_like(x_drm, device=x_drm.device), torch.ones_like(x_drm, device=x_drm.device)

    while abs(_x - _t) > precision:
        if _x > _t:
            r = _x
            # print(f"{_x} - ({_x} - {l}) * {b}")
            _x = _x - (_x - l) * b

            r_drm = x_drm.clone()
            x_drm = x_drm - (x_drm - l_drm) * b_drm
            # print(_x, x_drm)

        if _x < _t:
            l = _x
            # print(f"{_x} + ({r} - {_x}) * {b}")
            _x = _x + (r - _x) * b

            l_drm = x_drm.clone()
            x_drm = x_drm + (r_drm - x_drm) * b_drm
            # print(_x, x_drm)

    return x_drm.to(dtype)


def calc_drm_rife(_t, flow01, flow10, flow12, flow21):
    # Compute the distance using the optical flow and distance calculator
    d10 = distance_calculator(flow10) + 1e-4
    d12 = distance_calculator(flow12) + 1e-4

    # Calculate the distance ratio map
    drm10 = d10 / (d10 + d12)
    drm12 = d12 / (d10 + d12)

    # Theoretically, the distance ratio can only be controlled by flow emitted from the center to sides.
    drm01_unaligned = 1 - drm10
    drm21_unaligned = 1 - drm12

    ones_mask = torch.ones_like(drm10, device=drm10.device)
    warp_method = 'avg'

    drm0t1_unaligned = get_drm_t(drm01_unaligned, _t)
    drm1t0_unaligned = get_drm_t(drm10, _t)
    drm1t2_unaligned = get_drm_t(drm12, _t)
    drm2t1_unaligned = get_drm_t(drm21_unaligned, _t)

    drm0t1 = warp(drm0t1_unaligned, flow01 * drm0t1_unaligned, None, warp_method)
    drm1t0 = warp(drm1t0_unaligned, flow10 * drm1t0_unaligned, None, warp_method)
    drm1t2 = warp(drm1t2_unaligned, flow12 * drm1t2_unaligned, None, warp_method)
    drm2t1 = warp(drm2t1_unaligned, flow21 * drm2t1_unaligned, None, warp_method)

    mask0t1 = warp(ones_mask, flow01 * drm0t1_unaligned, None, warp_method)
    mask1t0 = warp(ones_mask, flow10 * drm1t0_unaligned, None, warp_method)
    mask1t2 = warp(ones_mask, flow12 * drm1t2_unaligned, None, warp_method)
    mask2t1 = warp(ones_mask, flow21 * drm2t1_unaligned, None, warp_method)

    holes0t1 = mask0t1 < 0.999
    holes1t0 = mask1t0 < 0.999
    holes1t2 = mask1t2 < 0.999
    holes2t1 = mask2t1 < 0.999

    drm0t1[holes0t1] = drm0t1_unaligned[holes0t1]
    drm1t0[holes1t0] = drm1t0_unaligned[holes1t0]
    drm1t2[holes1t2] = drm1t2_unaligned[holes1t2]
    drm2t1[holes2t1] = drm2t1_unaligned[holes2t1]

    return drm0t1, drm1t0, drm1t2, drm2t1


def calc_drm_rife_auxiliary(_t, drm10, drm12, flow10, flow12, metric10, metric12):
    # drm10 and drm12 can be directly calculated by nesting calc_drm_gmfss(flow10, flow12, metric10, metric12).
    # However, in order to reuse these two variables to improve efficiency,
    # drm10 and drm12 are directly passed as parameters here.
    warp_method = 'soft' if (metric10 is not None and metric12 is not None) else 'avg'

    # For RIFE, drm should be aligned with the time corresponding to the intermediate frame.
    drm1t0_unaligned = get_drm_t(drm10, _t)
    drm1t2_unaligned = get_drm_t(drm12, _t)

    drm1t0 = warp(drm1t0_unaligned, flow10 * drm1t0_unaligned, metric10, warp_method)
    drm1t2 = warp(drm1t2_unaligned, flow12 * drm1t2_unaligned, metric12, warp_method)

    ones_mask = torch.ones_like(drm1t0, device=drm1t0.device)

    warped_ones_mask1t0 = warp(ones_mask, flow10 * drm1t0_unaligned, metric10, warp_method)
    warped_ones_mask1t2 = warp(ones_mask, flow12 * drm1t2_unaligned, metric12, warp_method)

    holes1t0 = warped_ones_mask1t0 < 0.999
    holes1t2 = warped_ones_mask1t2 < 0.999

    drm1t0[holes1t0] = drm1t0_unaligned[holes1t0]
    drm1t2[holes1t2] = drm1t2_unaligned[holes1t2]

    return drm1t0, drm1t2

# Deprecated: Code below is no longer in use and may be removed in the future.

# def calc_drm_rife_auxiliary(_t, drm10, drm12, flow10, flow12, metric10, metric12):
#     # drm10 and drm12 can be directly calculated by nesting calc_drm_gmfss(flow10, flow12, metric10, metric12).
#     # However, in order to reuse these two variables to improve efficiency,
#     # drm10 and drm12 are directly passed as parameters here.
#     warp_method = 'soft' if (metric10 is not None and metric12 is not None) else 'avg'
#     # For RIFE, drm should be aligned with the time corresponding to the intermediate frame.
#     _drm01r = warp(1 - drm10, flow10 * ((1 - drm10) * 2) * _t, metric10, warp_method)
#     _drm21r = warp(1 - drm12, flow12 * ((1 - drm12) * 2) * _t, metric12, warp_method)
#
#     ones_mask = torch.ones_like(_drm01r, device=_drm01r.device)
#
#     warped_ones_mask01r = warp(ones_mask, flow10 * ((1 - drm10) * 2) * _t, metric10, warp_method)
#     warped_ones_mask21r = warp(ones_mask, flow12 * ((1 - drm12) * 2) * _t, metric12, warp_method)
#
#     holes01r = warped_ones_mask01r < 0.999
#     holes21r = warped_ones_mask21r < 0.999
#
#     _drm01r[holes01r] = (1 - drm10)[holes01r]
#     _drm21r[holes21r] = (1 - drm12)[holes21r]
#
#     return _drm01r, _drm21r
#
#
# def calc_drm_rife(_t, flow10_p, flow12_p, flow10_s, flow12_s):
#     # Compute the distance using the optical flow and distance calculator
#     d10_p = distance_calculator(flow10_p) + 1e-4
#     d12_p = distance_calculator(flow12_p) + 1e-4
#     d10_s = distance_calculator(flow10_s) + 1e-4
#     d12_s = distance_calculator(flow12_s) + 1e-4
#
#     # Calculate the distance ratio map
#     drm10_p = d10_p / (d10_p + d12_p)
#     drm12_p = d12_p / (d10_p + d12_p)
#     drm10_s = d10_s / (d10_s + d12_s)
#     drm12_s = d12_s / (d10_s + d12_s)
#
#     warp_method = 'avg'
#     # The distance ratio map (drm) is initially aligned with I1.
#     # To align it with I0 and I2, we need to warp the drm maps.
#     # Note: 1. To reverse the direction of the drm map, use 1 - drm and then warp it.
#     # 2. For RIFE, drm should be aligned with the time corresponding to the intermediate frame.
#     _drm01r_p = warp(1 - drm10_p, flow10_p * ((1 - drm10_p) * 2) * _t, None, warp_method)
#     _drm21r_p = warp(1 - drm12_p, flow12_p * ((1 - drm12_p) * 2) * _t, None, warp_method)
#     _drm01r_s = warp(1 - drm10_s, flow10_s * ((1 - drm10_s) * 2) * _t, None, warp_method)
#     _drm21r_s = warp(1 - drm12_s, flow12_s * ((1 - drm12_s) * 2) * _t, None, warp_method)
#
#     ones_mask = torch.ones_like(_drm01r_p, device=_drm01r_p.device)
#
#     warped_ones_mask01r_p = warp(ones_mask, flow10_p * ((1 - drm10_p) * 2) * _t, None, warp_method)
#     warped_ones_mask21r_p = warp(ones_mask, flow12_p * ((1 - drm12_p) * 2) * _t, None, warp_method)
#     warped_ones_mask01r_s = warp(ones_mask, flow10_s * ((1 - drm10_s) * 2) * _t, None, warp_method)
#     warped_ones_mask21r_s = warp(ones_mask, flow12_s * ((1 - drm12_s) * 2) * _t, None, warp_method)
#
#     holes01r_p = warped_ones_mask01r_p < 0.999
#     holes21r_p = warped_ones_mask21r_p < 0.999
#
#     _drm01r_p[holes01r_p] = _drm01r_s[holes01r_p]
#     _drm21r_p[holes21r_p] = _drm21r_s[holes21r_p]
#
#     holes01r_s = warped_ones_mask01r_s < 0.999
#     holes21r_s = warped_ones_mask21r_s < 0.999
#
#     holes01r = torch.logical_and(holes01r_p, holes01r_s)
#     holes21r = torch.logical_and(holes21r_p, holes21r_s)
#
#     _drm01r_p[holes01r] = (1 - _drm01r_p)[holes01r]
#     _drm21r_p[holes21r] = (1 - _drm21r_p)[holes21r]
#
#     return _drm01r_p, _drm21r_p
