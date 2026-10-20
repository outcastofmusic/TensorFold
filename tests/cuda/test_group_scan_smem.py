"""group_scan_kernel at 679-683 experts: 72 dynamic bytes an expert plus 264 static bytes cross 48 KiB at 680, so the
launch needs the opt-in there (review 2026-10-20, patch 01). Before the patch 680-682 fail to launch."""

import pytest
import torch

from test_group_kernel_smem import _assert_members

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")


@pytest.mark.parametrize("experts", [679, 680, 681, 682, 683])
def test_scan_grouping_launches_across_the_48_kib_line(experts):
    from tensorfold.cuda.exl3 import experts as x3

    rows, slots = 64, 9
    g = torch.Generator().manual_seed(experts)
    pick = torch.randint(0, experts, (rows, slots), generator=g, dtype=torch.int32).cuda()
    n = min(rows * slots, experts)
    ids = torch.zeros(n, dtype=torch.int32, device="cuda")
    count = torch.zeros(1, dtype=torch.int32, device="cuda")
    members = torch.full((n, rows), -1, dtype=torch.int32, device="cuda")
    x3._ext().group(pick, ids, count, members, rows, slots, experts, False, scan=True)
    torch.cuda.synchronize()
    _assert_members(pick, ids, count, members, experts)
