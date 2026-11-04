"""The start-up budget counts the free pages the kernel parks on its per-CPU lists (upstream #1323)."""

from tensorfold.cuda import capacity

PAGE = 4096
HIGH_MIN = 2146
GIB = 1 << 30


def zone(node, name, counts=None, high_min=HIGH_MIN):
    """One zone's /proc/zoneinfo block as kernel 6.7+ prints it, with page lists when ``counts`` is given."""

    out = f"Node {node}, zone {name:>8}\n  pages free     10\n        managed  1000\n      nr_free_cma  0\n"
    if counts is not None:
        out += "  pagesets\n"
        for cpu, count in enumerate(counts):
            out += f"    cpu: {cpu}\n              count:    {count}\n              high:     5000\n"
            out += "              batch:    63\n"
            if high_min is not None:
                out += f"              high_min: {high_min}\n"
            out += "              high_max: 198283\n  vm stats threshold: 110\n"
    return out + "  node_unreclaimable:  0\n  start_pfn:           4096\n"


def test_every_cpu_zone_and_node_less_each_lists_high_min():
    a, b, c = [200000, 3143, 2332, 17], [100000] * 3, [400000]
    text = (zone(0, "DMA", a) + zone(0, "DMA32") + zone(0, "Normal", b) + zone(1, "Normal", c)
            + zone(1, "Movable"))
    assert capacity.parked_bytes(text, PAGE) == (sum(a + b + c) - 8 * HIGH_MIN) * PAGE


def test_lists_under_their_resting_level_park_nothing():
    assert capacity.parked_bytes(zone(0, "Normal", [100] * 20), PAGE) == 0


def test_kernels_without_high_min_rest_the_first_512_mib():
    half = (GIB + capacity.PARKED_RESTING) // PAGE // 2
    assert capacity.parked_bytes(zone(0, "Normal", [half, half], high_min=None), PAGE) == GIB
    assert capacity.parked_bytes(zone(0, "Normal", [1000] * 4, high_min=None), PAGE) == 0


def test_only_count_and_high_min_lines_under_pagesets():
    text = ("  count: 5\n      count: 6\n" + zone(0, "Normal", [HIGH_MIN + 10])
            + "      count:    99999\n      high_min: 7\n"
            + "Node 0, zone Other\n      count:    77777\n  pagesets\n    cpu: 0\n              high:     1\n")
    assert capacity.parked_bytes(text, PAGE) == 10 * PAGE


def test_a_bad_field_counts_nothing():
    for text in ("", "Node 0, zone Normal\n  pagesets\n    cpu: 0\n      count: x\n"):
        assert capacity.parked_bytes(text, PAGE) == 0


def test_meminfo_adds_parked_pages_to_available_at_most_total(monkeypatch, tmp_path):
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal: 134217728 kB\nMemAvailable: 100663296 kB\n")
    zoneinfo = tmp_path / "zoneinfo"
    files = {"/proc/meminfo": meminfo, "/proc/zoneinfo": zoneinfo}
    real_path = capacity.Path
    monkeypatch.setattr(capacity, "Path", lambda p: real_path(files.get(str(p), p)))
    monkeypatch.setattr(capacity.os, "sysconf", lambda name: PAGE)

    zoneinfo.write_text(zone(0, "Normal", [HIGH_MIN + 6 * GIB // PAGE]))
    assert capacity._meminfo()["MemAvailable"] == 102 * GIB
    zoneinfo.write_text(zone(0, "Normal", [HIGH_MIN + 40 * GIB // PAGE]))
    assert capacity._meminfo()["MemAvailable"] == 128 * GIB          # MemTotal
    zoneinfo.unlink()
    assert capacity._meminfo()["MemAvailable"] == 96 * GIB
