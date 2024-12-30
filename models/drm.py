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

    # The distance ratio map (drm) is initially aligned with I1.
    # To align it with I0 and I2, we need to warp the drm maps.
    # Note: To reverse the direction of the drm map, use 1 - drm and then warp it.
    drm01 = warp(1 - drm10, flow10, metric10, warp_method)
    drm21 = warp(1 - drm12, flow12, metric12, warp_method)

    # Create a mask with all ones to identify the holes in the warped drm maps
    ones_mask = torch.ones_like(drm01, device=drm01.device)

    # Warp the ones mask
    warped_ones_mask01 = warp(ones_mask, flow10, metric10, warp_method)
    warped_ones_mask21 = warp(ones_mask, flow12, metric12, warp_method)

    # Identify holes in warped drm map
    holes01 = warped_ones_mask01 < 0.999
    holes21 = warped_ones_mask21 < 0.999

    # Fill the holes in the warped drm maps with the inverse of the original drm maps
    drm01[holes01] = (1 - drm10)[holes01]
    drm21[holes21] = (1 - drm12)[holes21]

    return drm01, drm10, drm12, drm21


def calc_drm_rife_auxiliary(_t, drm10, drm12, flow10, flow12, metric10, metric12):
    # drm10 and drm12 can be directly calculated by nesting calc_drm_gmfss(flow10, flow12, metric10, metric12).
    # However, in order to reuse these two variables to improve efficiency,
    # drm10 and drm12 are directly passed as parameters here.
    warp_method = 'soft' if (metric10 is not None and metric12 is not None) else 'avg'
    # For RIFE, drm should be aligned with the time corresponding to the intermediate frame.
    _drm01r = warp(1 - drm10, flow10 * ((1 - drm10) * 2) * _t, metric10, warp_method)
    _drm21r = warp(1 - drm12, flow12 * ((1 - drm12) * 2) * _t, metric12, warp_method)

    ones_mask = torch.ones_like(_drm01r, device=_drm01r.device)

    warped_ones_mask01r = warp(ones_mask, flow10 * ((1 - drm10) * 2) * _t, metric10, warp_method)
    warped_ones_mask21r = warp(ones_mask, flow12 * ((1 - drm12) * 2) * _t, metric12, warp_method)

    holes01r = warped_ones_mask01r < 0.999
    holes21r = warped_ones_mask21r < 0.999

    _drm01r[holes01r] = (1 - drm10)[holes01r]
    _drm21r[holes21r] = (1 - drm12)[holes21r]

    return _drm01r, _drm21r


def calc_drm_rife(_t, flow10_p, flow12_p, flow10_s, flow12_s):
    # Compute the distance using the optical flow and distance calculator
    d10_p = distance_calculator(flow10_p) + 1e-4
    d12_p = distance_calculator(flow12_p) + 1e-4
    d10_s = distance_calculator(flow10_s) + 1e-4
    d12_s = distance_calculator(flow12_s) + 1e-4

    # Calculate the distance ratio map
    drm10_p = d10_p / (d10_p + d12_p)
    drm12_p = d12_p / (d10_p + d12_p)
    drm10_s = d10_s / (d10_s + d12_s)
    drm12_s = d12_s / (d10_s + d12_s)

    warp_method = 'avg'
    # The distance ratio map (drm) is initially aligned with I1.
    # To align it with I0 and I2, we need to warp the drm maps.
    # Note: 1. To reverse the direction of the drm map, use 1 - drm and then warp it.
    # 2. For RIFE, drm should be aligned with the time corresponding to the intermediate frame.
    _drm01r_p = warp(1 - drm10_p, flow10_p * ((1 - drm10_p) * 2) * _t, None, warp_method)
    _drm21r_p = warp(1 - drm12_p, flow12_p * ((1 - drm12_p) * 2) * _t, None, warp_method)
    _drm01r_s = warp(1 - drm10_s, flow10_s * ((1 - drm10_s) * 2) * _t, None, warp_method)
    _drm21r_s = warp(1 - drm12_s, flow12_s * ((1 - drm12_s) * 2) * _t, None, warp_method)

    ones_mask = torch.ones_like(_drm01r_p, device=_drm01r_p.device)

    warped_ones_mask01r_p = warp(ones_mask, flow10_p * ((1 - drm10_p) * 2) * _t, None, warp_method)
    warped_ones_mask21r_p = warp(ones_mask, flow12_p * ((1 - drm12_p) * 2) * _t, None, warp_method)
    warped_ones_mask01r_s = warp(ones_mask, flow10_s * ((1 - drm10_s) * 2) * _t, None, warp_method)
    warped_ones_mask21r_s = warp(ones_mask, flow12_s * ((1 - drm12_s) * 2) * _t, None, warp_method)

    holes01r_p = warped_ones_mask01r_p < 0.999
    holes21r_p = warped_ones_mask21r_p < 0.999

    _drm01r_p[holes01r_p] = _drm01r_s[holes01r_p]
    _drm21r_p[holes21r_p] = _drm21r_s[holes21r_p]

    holes01r_s = warped_ones_mask01r_s < 0.999
    holes21r_s = warped_ones_mask21r_s < 0.999

    holes01r = torch.logical_and(holes01r_p, holes01r_s)
    holes21r = torch.logical_and(holes21r_p, holes21r_s)

    _drm01r_p[holes01r] = (1 - _drm01r_p)[holes01r]
    _drm21r_p[holes21r] = (1 - _drm21r_p)[holes21r]

    return _drm01r_p, _drm21r_p


def get_drm_t(drm, _t, precision=1e-3):
    _x, b = 0.5, 0.5
    l, r = 0, 1

    x_drm, b_drm = drm.clone(), drm.clone()
    l_drm, r_drm = torch.zeros_like(x_drm, device=x_drm.device), torch.ones_like(x_drm, device=x_drm.device)

    while abs(_x - _t) > precision:
        if _x > _t:
            r = _x
            _x = _x - (_x - l) * b

            r_drm = x_drm.clone()
            x_drm = x_drm - (x_drm - l_drm) * b_drm

        if _x < _t:
            l = _x
            _x = _x + (r - _x) * b

            l_drm = x_drm.clone()
            x_drm = x_drm + (r_drm - x_drm) * b_drm

    return x_drm


def calc_drm_rife_v2(_t, flow10_p, flow12_p, flow10_s, flow12_s):
    # Compute the distance using the optical flow and distance calculator
    d10_p = distance_calculator(flow10_p) + 1e-4
    d12_p = distance_calculator(flow12_p) + 1e-4
    d10_s = distance_calculator(flow10_s) + 1e-4
    d12_s = distance_calculator(flow12_s) + 1e-4

    # Calculate the distance ratio map
    drm10_p = d10_p / (d10_p + d12_p)
    drm12_p = d12_p / (d10_p + d12_p)
    drm10_s = d10_s / (d10_s + d12_s)
    drm12_s = d12_s / (d10_s + d12_s)

    warp_method = 'avg'
    # The distance ratio map (drm) is initially aligned with I1.
    # To align it with I0 and I2, we need to warp the drm maps.
    # Note: 1. To reverse the direction of the drm map, use 1 - drm and then warp it.
    # 2. For RIFE, drm should be aligned with the time corresponding to the intermediate frame.
    drm10r_p = get_drm_t(1 - drm10_p, _t)
    drm12r_p = get_drm_t(1 - drm12_p, _t)
    drm10r_s = get_drm_t(1 - drm10_s, _t)
    drm12r_s = get_drm_t(1 - drm12_s, _t)
    _drm01r_p = warp(drm10r_p, flow10_p * drm10r_p, None, warp_method)
    _drm21r_p = warp(drm12r_p, flow12_p * drm12r_p, None, warp_method)
    _drm01r_s = warp(drm10r_s, flow10_s * drm10r_s, None, warp_method)
    _drm21r_s = warp(drm12r_s, flow12_s * drm12r_s, None, warp_method)

    ones_mask = torch.ones_like(_drm01r_p, device=_drm01r_p.device)

    warped_ones_mask01r_p = warp(ones_mask, flow10_p * drm10r_p, None, warp_method)
    warped_ones_mask21r_p = warp(ones_mask, flow12_p * drm12r_p, None, warp_method)
    warped_ones_mask01r_s = warp(ones_mask, flow10_s * drm10r_s, None, warp_method)
    warped_ones_mask21r_s = warp(ones_mask, flow12_s * drm12r_s, None, warp_method)

    holes01r_p = warped_ones_mask01r_p < 0.999
    holes21r_p = warped_ones_mask21r_p < 0.999

    _drm01r_p[holes01r_p] = _drm01r_s[holes01r_p]
    _drm21r_p[holes21r_p] = _drm21r_s[holes21r_p]

    holes01r_s = warped_ones_mask01r_s < 0.999
    holes21r_s = warped_ones_mask21r_s < 0.999

    holes01r = torch.logical_and(holes01r_p, holes01r_s)
    holes21r = torch.logical_and(holes21r_p, holes21r_s)

    _drm01r_p[holes01r] = drm10r_p[holes01r]
    _drm21r_p[holes21r] = drm12r_p[holes21r]

    return _drm01r_p, _drm21r_p


def calc_drm_rife_auxiliary_v2(_t, drm10, drm12, flow10, flow12, metric10, metric12):
    # drm10 and drm12 can be directly calculated by nesting calc_drm_gmfss(flow10, flow12, metric10, metric12).
    # However, in order to reuse these two variables to improve efficiency,
    # drm10 and drm12 are directly passed as parameters here.
    warp_method = 'soft' if (metric10 is not None and metric12 is not None) else 'avg'
    # For RIFE, drm should be aligned with the time corresponding to the intermediate frame.
    drm10r = get_drm_t(1 - drm10, _t)
    drm12r = get_drm_t(1 - drm12, _t)
    _drm01r = warp(drm10r, flow10 * drm10r, metric10, warp_method)
    _drm21r = warp(drm12r, flow12 * drm12r, metric12, warp_method)

    ones_mask = torch.ones_like(_drm01r, device=_drm01r.device)

    warped_ones_mask01r = warp(ones_mask, flow10 * drm10r, metric10, warp_method)
    warped_ones_mask21r = warp(ones_mask, flow12 * drm12r, metric12, warp_method)

    holes01r = warped_ones_mask01r < 0.999
    holes21r = warped_ones_mask21r < 0.999

    _drm01r[holes01r] = drm10r[holes01r]
    _drm21r[holes21r] = drm12r[holes21r]

    return _drm01r, _drm21r
