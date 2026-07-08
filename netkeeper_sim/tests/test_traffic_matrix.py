from __future__ import annotations

import pytest

from netkeeper_sim.traffic.matrix import TrafficMatrix


def test_traffic_matrix_from_csv(tmp_path):
    traffic_file = tmp_path / "traffic.csv"
    traffic_file.write_text(
        "source,destination,demand\nR1,R3,100\nR2,R2,10\n",
        encoding="utf-8",
    )

    matrix = TrafficMatrix.from_csv(traffic_file, ["R1", "R2", "R3"])

    assert len(matrix) == 1
    assert matrix.demands[0].source == "R1"
    assert matrix.demands[0].destination == "R3"
    assert matrix.demands[0].demand == 100.0


def test_traffic_matrix_rejects_negative_csv_demand(tmp_path):
    traffic_file = tmp_path / "traffic.csv"
    traffic_file.write_text(
        "source,destination,demand\nR1,R3,-1\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="non-negative"):
        TrafficMatrix.from_csv(traffic_file, ["R1", "R3"])


def test_traffic_matrix_rejects_unknown_nodes(tmp_path):
    traffic_file = tmp_path / "traffic.csv"
    traffic_file.write_text(
        "source,destination,demand\nR1,R9,1\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unknown traffic destination"):
        TrafficMatrix.from_csv(traffic_file, ["R1", "R3"])


def test_traffic_matrix_from_numpy_array():
    np = pytest.importorskip("numpy")
    matrix = np.array(
        [
            [0, 10, 0],
            [0, 0, 20],
            [5, 0, 0],
        ]
    )

    traffic = TrafficMatrix.from_numpy(matrix, ["R1", "R2", "R3"])

    assert len(traffic) == 3
    assert traffic.total_demand == 35


def test_traffic_matrix_from_array_like_list():
    traffic = TrafficMatrix.from_numpy(
        [
            [0, 10],
            [3, 0],
        ],
        ["R1", "R2"],
    )

    assert len(traffic) == 2
    assert traffic.total_demand == 13
