import pytest
import numpy as np
import math
from easybfe.amber.prep_ligand_abfe import compute_bond, compute_angle, compute_dihedral


class TestComputeBond:
    """Test cases for compute_bond function."""
    
    def test_bond_along_x_axis(self):
        """Test bond length along x-axis."""
        pos0 = [0.0, 0.0, 0.0]
        pos1 = [3.0, 0.0, 0.0]
        assert abs(compute_bond(pos0, pos1) - 3.0) < 1e-10
    
    def test_bond_along_y_axis(self):
        """Test bond length along y-axis."""
        pos0 = [0.0, 0.0, 0.0]
        pos1 = [0.0, 5.0, 0.0]
        assert abs(compute_bond(pos0, pos1) - 5.0) < 1e-10
    
    def test_bond_along_z_axis(self):
        """Test bond length along z-axis."""
        pos0 = [0.0, 0.0, 0.0]
        pos1 = [0.0, 0.0, 7.0]
        assert abs(compute_bond(pos0, pos1) - 7.0) < 1e-10
    
    def test_bond_3d_diagonal(self):
        """Test bond length in 3D space."""
        pos0 = [0.0, 0.0, 0.0]
        pos1 = [3.0, 4.0, 0.0]
        assert abs(compute_bond(pos0, pos1) - 5.0) < 1e-10
    
    def test_bond_arbitrary_positions(self):
        """Test bond length with arbitrary positions."""
        pos0 = [1.0, 2.0, 3.0]
        pos1 = [4.0, 6.0, 9.0]
        expected = math.sqrt((4-1)**2 + (6-2)**2 + (9-3)**2)
        assert abs(compute_bond(pos0, pos1) - expected) < 1e-10
    
    def test_bond_zero_length(self):
        """Test bond length when positions are identical."""
        pos0 = [1.0, 2.0, 3.0]
        pos1 = [1.0, 2.0, 3.0]
        assert abs(compute_bond(pos0, pos1)) < 1e-10
    
    def test_bond_with_numpy_arrays(self):
        """Test bond computation with numpy arrays."""
        pos0 = np.array([0.0, 0.0, 0.0])
        pos1 = np.array([3.0, 4.0, 0.0])
        assert abs(compute_bond(pos0, pos1) - 5.0) < 1e-10


class TestComputeAngle:
    """Test cases for compute_angle function."""
    
    def test_angle_90_degrees(self):
        """Test 90-degree angle (right angle)."""
        # pos0-pos1-pos2 forms a right angle
        pos0 = [1.0, 0.0, 0.0]
        pos1 = [0.0, 0.0, 0.0]  # Central atom
        pos2 = [0.0, 1.0, 0.0]
        angle = compute_angle(pos0, pos1, pos2)
        assert abs(angle - 90.0) < 1e-10
    
    def test_angle_180_degrees(self):
        """Test 180-degree angle (straight line)."""
        pos0 = [1.0, 0.0, 0.0]
        pos1 = [0.0, 0.0, 0.0]
        pos2 = [-1.0, 0.0, 0.0]
        angle = compute_angle(pos0, pos1, pos2)
        assert abs(angle - 180.0) < 1e-10
    
    def test_angle_60_degrees(self):
        """Test 60-degree angle (equilateral triangle)."""
        # Equilateral triangle: all angles are 60 degrees
        pos1 = [0.0, 0.0, 0.0]
        pos0 = [1.0, 0.0, 0.0]
        pos2 = [0.5, math.sqrt(3)/2, 0.0]
        angle = compute_angle(pos0, pos1, pos2)
        assert abs(angle - 60.0) < 1e-9
    
    def test_angle_0_degrees(self):
        """Test 0-degree angle (collinear, same direction)."""
        pos0 = [0.0, 0.0, 0.0]
        pos1 = [1.0, 0.0, 0.0]
        pos2 = [2.0, 0.0, 0.0]
        angle = compute_angle(pos0, pos1, pos2)
        assert abs(angle - 0.0) < 1e-10
    
    def test_angle_45_degrees(self):
        """Test 45-degree angle."""
        pos1 = [0.0, 0.0, 0.0]
        pos0 = [1.0, 0.0, 0.0]
        pos2 = [1.0, 1.0, 0.0]
        angle = compute_angle(pos0, pos1, pos2)
        assert abs(angle - 45.0) < 1e-9
    
    def test_angle_with_numpy_arrays(self):
        """Test angle computation with numpy arrays."""
        pos0 = np.array([1.0, 0.0, 0.0])
        pos1 = np.array([0.0, 0.0, 0.0])
        pos2 = np.array([0.0, 1.0, 0.0])
        angle = compute_angle(pos0, pos1, pos2)
        assert abs(angle - 90.0) < 1e-10
    
    def test_angle_zero_length_vector(self):
        """Test angle computation raises error for zero-length vector."""
        pos0 = [0.0, 0.0, 0.0]
        pos1 = [0.0, 0.0, 0.0]
        pos2 = [1.0, 0.0, 0.0]
        with pytest.raises(ValueError, match="zero-length vector"):
            compute_angle(pos0, pos1, pos2)


class TestComputeDihedral:
    """Test cases for compute_dihedral function."""
    
    def test_dihedral_0_degrees(self):
        """Test 0-degree dihedral (cis conformation)."""
        # All atoms in the same plane, dihedral = 0
        pos0 = [0.0, 0.0, 0.0]
        pos1 = [1.0, 0.0, 0.0]
        pos2 = [2.0, 0.0, 0.0]
        pos3 = [2.0, 1.0, 0.0]
        dihedral = compute_dihedral(pos0, pos1, pos2, pos3)
        assert abs(dihedral - 0.0) < 1e-9
    
    def test_dihedral_180_degrees(self):
        """Test 180-degree dihedral (trans conformation)."""
        pos0 = [0.0, 0.0, 0.0]
        pos1 = [1.0, 0.0, 0.0]
        pos2 = [2.0, 0.0, 0.0]
        pos3 = [2.0, -1.0, 0.0]
        dihedral = compute_dihedral(pos0, pos1, pos2, pos3)
        assert abs(abs(dihedral) - 180.0) < 1e-9
    
    def test_dihedral_90_degrees(self):
        """Test 90-degree dihedral."""
        pos0 = [0.0, 0.0, 0.0]
        pos1 = [1.0, 0.0, 0.0]
        pos2 = [2.0, 0.0, 0.0]
        pos3 = [2.0, 0.0, 1.0]
        dihedral = compute_dihedral(pos0, pos1, pos2, pos3)
        assert abs(abs(dihedral) - 90.0) < 1e-9
    
    def test_dihedral_minus_90_degrees(self):
        """Test -90-degree dihedral."""
        pos0 = [0.0, 0.0, 0.0]
        pos1 = [1.0, 0.0, 0.0]
        pos2 = [2.0, 0.0, 0.0]
        pos3 = [2.0, 0.0, -1.0]
        dihedral = compute_dihedral(pos0, pos1, pos2, pos3)
        assert abs(dihedral + 90.0) < 1e-9
    
    def test_dihedral_45_degrees(self):
        """Test 45-degree dihedral."""
        pos0 = [0.0, 0.0, 0.0]
        pos1 = [1.0, 0.0, 0.0]
        pos2 = [2.0, 0.0, 0.0]
        # Create 45-degree dihedral by rotating in xy plane
        pos3 = [2.0, 1.0, 1.0]
        dihedral = compute_dihedral(pos0, pos1, pos2, pos3)
        # Should be approximately 45 degrees
        assert abs(abs(dihedral) - 45.0) < 1.0  # Allow some tolerance
    
    def test_dihedral_with_numpy_arrays(self):
        """Test dihedral computation with numpy arrays."""
        pos0 = np.array([0.0, 0.0, 0.0])
        pos1 = np.array([1.0, 0.0, 0.0])
        pos2 = np.array([2.0, 0.0, 0.0])
        pos3 = np.array([2.0, 0.0, 1.0])
        dihedral = compute_dihedral(pos0, pos1, pos2, pos3)
        assert abs(abs(dihedral) - 90.0) < 1e-9
    
    def test_dihedral_range(self):
        """Test that dihedral angles are in (-180, 180] range."""
        # Test various configurations
        test_cases = [
            ([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 1.0, 0.0]),
            ([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, -1.0, 0.0]),
            ([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 0.0, 1.0]),
            ([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 0.0, -1.0]),
        ]
        for pos0, pos1, pos2, pos3 in test_cases:
            dihedral = compute_dihedral(pos0, pos1, pos2, pos3)
            assert -180 < dihedral <= 180, f"Dihedral {dihedral} not in range (-180, 180]"
    
    def test_dihedral_collinear_atoms(self):
        """Test dihedral computation raises error for collinear atoms."""
        # All atoms collinear
        pos0 = [0.0, 0.0, 0.0]
        pos1 = [1.0, 0.0, 0.0]
        pos2 = [2.0, 0.0, 0.0]
        pos3 = [3.0, 0.0, 0.0]
        with pytest.raises(ValueError, match="degenerate geometry"):
            compute_dihedral(pos0, pos1, pos2, pos3)
    
    def test_dihedral_zero_central_bond(self):
        """Test dihedral computation raises error for zero-length central bond."""
        pos0 = [0.0, 0.0, 0.0]
        pos1 = [1.0, 0.0, 0.0]
        pos2 = [1.0, 0.0, 0.0]  # Same as pos1
        pos3 = [2.0, 0.0, 0.0]
        with pytest.raises(ValueError, match="zero-length central bond"):
            compute_dihedral(pos0, pos1, pos2, pos3)


class TestGeometryIntegration:
    """Integration tests for geometry functions."""
    
    def test_water_molecule_geometry(self):
        """Test geometry functions with a water molecule."""
        # Water molecule: O at origin, H atoms at ~1.0 Angstrom, H-O-H angle ~104.5°
        O = [0.0, 0.0, 0.0]
        H1 = [0.9572, 0.0, 0.0]  # ~1.0 Angstrom from O
        H2 = [-0.2400, 0.9266, 0.0]  # ~1.0 Angstrom from O, ~104.5° angle
        
        # Test bond lengths
        bond_OH1 = compute_bond(O, H1)
        bond_OH2 = compute_bond(O, H2)
        assert abs(bond_OH1 - 0.9572) < 1e-4
        assert abs(bond_OH2 - 0.9572) < 1e-4
        
        # Test angle
        angle = compute_angle(H1, O, H2)
        assert abs(angle - 104.5) < 1.0  # Allow some tolerance
    
    def test_ethane_dihedral(self):
        """Test dihedral angle in ethane-like molecule."""
        # Ethane: C-C bond with staggered conformation (dihedral ~60°)
        C1 = [0.0, 0.0, 0.0]
        C2 = [1.5, 0.0, 0.0]
        H1 = [0.0, 1.0, 0.0]  # H on C1
        H2 = [1.5, 0.0, 1.0]  # H on C2, staggered
        
        dihedral = compute_dihedral(H1, C1, C2, H2)
        # In staggered ethane, dihedral is ~60° (or -60°)
        assert abs(abs(dihedral) - 60.0) < 5.0  # Allow some tolerance
