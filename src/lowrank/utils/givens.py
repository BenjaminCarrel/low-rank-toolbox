import numpy as np

def givens(x, y):
    """
    Computes the complex Givens rotation matrix for a 2-element complex vector [x, y].

    This function generalizes the real Givens rotation to the complex plane.
    The rotation is a unitary matrix G such that for a vector v = [x, y].T,
    the product G @ v is equal to [r, 0].T, where r is the real, non-negative
    value sqrt(|x|^2 + |y|^2).

    The resulting matrix G is an element of SU(2) and has the form:
    [[c,           s],
     [-conj(s), conj(c)]]
    where c and s are complex numbers satisfying |c|^2 + |s|^2 = 1.
    Specifically, the parameters are calculated as c = conj(x)/r and s = conj(y)/r.
    This implementation is numerically stable, avoiding overflow for large inputs.

    Args:
        x (complex or float): The first element of the vector.
        y (complex or float): The second element of the vector.

    Returns:
        np.ndarray: A 2x2 complex unitary Givens rotation matrix.
    """
    # Ensure inputs are complex for subsequent calculations
    x = complex(x)
    y = complex(y)
    
    if y == 0:
        if x == 0:
            c = 1.0 + 0j
            s = 0.0 + 0j
        else:
            # Rotate x to be real and non-negative, i.e., |x|
            c = np.conj(x) / np.abs(x)
            s = 0.0 + 0j
    else:
        abs_x = np.abs(x)
        abs_y = np.abs(y)
        
        # Use numerically stable calculations based on the larger component
        if abs_y >= abs_x:
            tau = x / y
            r_over_absy = np.hypot(np.abs(tau), 1.0)
            
            # Phase term from y contributes to s
            s_phase = np.conj(y) / abs_y
            
            s = s_phase / r_over_absy
            c = s * np.conj(tau)
        else:  # abs_x > abs_y
            tau = y / x
            r_over_absx = np.hypot(np.abs(tau), 1.0)
            
            # Phase term from x contributes to c
            c_phase = np.conj(x) / abs_x
            
            c = c_phase / r_over_absx
            s = c * np.conj(tau)

    # Construct the unitary Givens rotation matrix
    G = np.array([[c, s], [-np.conj(s), np.conj(c)]], dtype=np.complex128)
    
    return G

# Example of usage:
if __name__ == '__main__':
    # --- Real Vector Example ---
    x_real, y_real = 3.0, 4.0
    G_real = givens(x_real, y_real)
    v_real = np.array([x_real, y_real])
    result_real = G_real @ v_real
    
    print("--- Real Example ---")
    print(f"Input vector: {v_real}")
    print("Givens matrix:\n", np.round(G_real, 5))
    print(f"Result G @ v: {np.round(result_real, 5)}")
    print(f"Expected r: {np.hypot(x_real, y_real)}")
    print("-" * 20)

    # --- Complex Vector Example ---
    x_complex, y_complex = 1 + 2j, 3 - 4j
    G_complex = givens(x_complex, y_complex)
    v_complex = np.array([x_complex, y_complex])
    result_complex = G_complex @ v_complex
    
    print("\n--- Complex Example ---")
    print(f"Input vector: {v_complex}")
    print("Givens matrix:\n", np.round(G_complex, 5))
    print(f"Result G @ v: {np.round(result_complex, 5)}")
    print(f"Expected r: {np.hypot(np.abs(x_complex), np.abs(y_complex))}")
    print("-" * 20)
