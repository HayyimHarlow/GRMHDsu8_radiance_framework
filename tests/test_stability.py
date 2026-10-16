from su8_core.stability import StabilityParams, jacobian_matrix, is_linearly_stable, z_pinch_excluded

def test_jacobian_shape():
    params = StabilityParams(a_su8=1.0, a_ac=0.5, b_ac=0.3)
    J = jacobian_matrix(params)
    assert J.shape == (2, 2)

def test_stability_positive_damping():
    params = StabilityParams(a_su8=1.0, a_ac=0.5, b_ac=1.0)
    assert is_linearly_stable(params)

def test_z_pinch_excluded():
    assert z_pinch_excluded(1.0, 0.5, 2.0, gamma_ac=0.1)
