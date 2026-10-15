"""Grid anchors: prompt states kept at every 4,096-token line from position 0, besides the message markers."""

from tensorfold.cuda.markers import MIN_GAP, grid_env, with_grid


def test_grid_lines_join_the_marker_points():
    points = with_grid(lambda ids: [300], 4096)
    assert points(list(range(10000))) == [300, 4096, 8192]
    assert points(list(range(4096))) == [300]                      # a line at the prompt's end is no anchor


def test_lines_too_close_to_a_marker_give_way_to_it():
    points = with_grid(lambda ids: [4096 - MIN_GAP + 1, 9000], 4096)
    assert points(list(range(12000))) == [4096 - MIN_GAP + 1, 8192, 9000]      # 4,096: within MIN_GAP of a marker


def test_without_markers_the_grid_alone():
    assert with_grid(None, 4096)(list(range(9000))) == [4096, 8192]


def test_the_grid_is_off_unless_its_env_names_a_step(monkeypatch):
    monkeypatch.delenv("TENSORFOLD_GRID_ANCHORS", raising=False)
    assert grid_env() == 0
    monkeypatch.setenv("TENSORFOLD_GRID_ANCHORS", "4096")
    assert grid_env() == 4096
