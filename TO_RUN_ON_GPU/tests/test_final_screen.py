"""Static additions only: these checks have not been executed in this change."""
import numpy as np
from scripts.finalize_candidates import heavy_coordinates


def test_pose_reader_uses_only_first_model_and_heavy_atoms(tmp_path):
    def atom(i, x, kind):
        return f'ATOM  {i:5d}  C   LIG A   1    {x:8.3f}{0.:8.3f}{0.:8.3f}  1.00  0.00     0.000 {kind}'
    p = tmp_path / 'pose.pdbqt'
    p.write_text('MODEL 1\n' + atom(1, 1., 'C') + '\n' + atom(2, 2., 'HD') +
                 '\nENDMDL\nMODEL 2\n' + atom(3, 99., 'C') + '\nENDMDL\n')
    np.testing.assert_allclose(heavy_coordinates(p, first_pose=True), [[1., 0., 0.]])
