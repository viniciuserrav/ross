from pathlib import Path
from tempfile import tempdir

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal

from ross.bearing_seal_element import BearingElement
from ross.materials import steel

from ross.results import MeshConvergenceResults
from ross.rotor_assembly import Rotor, rotor_example
from ross.shaft_element import ShaftElement


@pytest.fixture
def rotor_thin_overhang():
    diameters = [0.15] * 6 + [0.03] * 2
    lengths = [0.2] * 6 + [0.3] * 2
    shaft_elements = [
        ShaftElement(L=L, idl=0, odl=d, material=steel)
        for d, L in zip(diameters, lengths, strict=True)
    ]
    bearing_elements = [
        BearingElement(n=0, kxx=1e7, cxx=1e3),
        BearingElement(n=6, kxx=1e7, cxx=1e3),
    ]
    return Rotor(shaft_elements, bearing_elements=bearing_elements)


def test_run_mesh_convergence():
    rotor = rotor_example()
    results = rotor.run_mesh_convergence(rtol=1e-4)

    assert len(rotor.shaft_elements) == 6
    assert_equal(results.subdivisions, [2, 2, 2, 2, 2, 2])
    assert len(results.rotor.shaft_elements) == 12
    assert results.reference_elements == 48
    assert results.error.max() <= 1e-4
    assert_allclose(results.rotor.m, rotor.m)

    fine = np.sort(rotor.refine(16).run_modal(0).wn)[:6]
    assert np.max(np.abs(results.wn / fine - 1)) <= 1e-4


def test_run_mesh_convergence_cheapest_grades_the_mesh(rotor_thin_overhang):
    rotor = rotor_thin_overhang
    uniform = rotor.run_mesh_convergence(rtol=1e-4, strategy="uniform")
    cheapest = rotor.run_mesh_convergence(rtol=1e-4, strategy="cheapest")

    assert len(cheapest.rotor.shaft_elements) < len(uniform.rotor.shaft_elements)
    assert cheapest.error.max() <= 1e-4
    overhang = cheapest.subdivisions[6:]
    assert np.all(overhang > cheapest.subdivisions[:6].max())
    assert "graded" in cheapest.families


def test_run_mesh_convergence_max_elements():
    with pytest.warns(UserWarning, match="max_elements=20"):
        results = rotor_example().run_mesh_convergence(rtol=1e-6, max_elements=20)
    assert results.reference_elements == 12


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (dict(rtol=0), "rtol"),
        (dict(frequencies=0), "frequencies"),
        (dict(strategy="optimal"), "strategy"),
    ],
)
def test_run_mesh_convergence_errors(kwargs, message):
    with pytest.raises(ValueError, match=message):
        rotor_example().run_mesh_convergence(**kwargs)


def test_mesh_convergence_results_plot_save_load():
    results = rotor_example().run_mesh_convergence(rtol=1e-4)
    fig = results.plot()
    assert {trace.name for trace in fig.data} == {"Uniform", "Chosen"}

    file = Path(tempdir) / "mesh_convergence.toml"
    results.save(file)
    loaded = MeshConvergenceResults.load(file)
    assert_equal(loaded.subdivisions, results.subdivisions)
    assert_allclose(loaded.wn, results.wn)
    assert_equal(loaded.families, results.families)
    assert len(loaded.rotor.shaft_elements) == len(results.rotor.shaft_elements)
