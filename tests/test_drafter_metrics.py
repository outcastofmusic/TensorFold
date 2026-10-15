"""Per-drafter acceptance in /metrics: copy drafts and MTP drafts counted apart, only where an engine reports them."""

from types import SimpleNamespace

from tensorfold.cuda import health
from tensorfold.cuda.streams import Stream
from tensorfold.server import metrics
from tests.test_metrics import sample

DRAFTED = f"{metrics.PREFIX}draft_tokens_total"
ACCEPTED = f"{metrics.PREFIX}draft_accepted_tokens_total"


def test_drafter_families_count_each_drafter_apart():
    app = SimpleNamespace()
    with health.of(app).running(5, [1, 2, 3]) as request:
        request.stats = {"drafted": 9, "accepted": 6, "drafters": {"mtp": [6, 4], "copy": [3, 2]}}
    with health.of(app).running(5, [1]) as request:
        request.stats = {"drafted": 5, "accepted": 1, "drafters": {"mtp": [5, 1]}}
    body = metrics.render(app)
    assert sample(body, DRAFTED + '{drafter="mtp"}') == "11" and sample(body, ACCEPTED + '{drafter="mtp"}') == "5"
    assert sample(body, DRAFTED + '{drafter="copy"}') == "3" and sample(body, ACCEPTED + '{drafter="copy"}') == "2"
    assert sample(body, f"{metrics.PREFIX}mtp_drafted_total") == "14"          # the existing totals are unchanged


def test_drafter_families_wait_for_an_engine_that_reports_them():
    app = SimpleNamespace()
    with health.of(app).running(5, [1]) as request:
        request.stats = {"drafted": 4, "accepted": 2}
    assert DRAFTED not in metrics.render(app)            # (spec_decode_num_draft_tokens_total is another family)


def test_a_stream_reports_its_drafters_only_when_it_counts_them():
    s = Stream([1, 2], 8)
    assert "drafters" not in s.stats()
    s.drafters = {"mtp": [3, 1], "copy": [5, 5]}
    assert s.stats()["drafters"] == {"mtp": [3, 1], "copy": [5, 5]}
    again = s.continued()                                       # a replay adds to what the first run counted
    again.drafters = {"mtp": [2, 2]}
    assert again.stats()["drafters"] == {"mtp": [5, 3], "copy": [5, 5]}
