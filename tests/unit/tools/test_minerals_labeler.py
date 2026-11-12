"""Minerals labeler: manifest builder rules and the server's blind, append-only behavior.

All data here is SYNTHETIC (generated in tmp_path); no real spectra are read.
"""

import csv
from contextlib import contextmanager
import hashlib
import http.client
import json
import math
from pathlib import Path
import re
import sys
import threading

import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.minerals_labeler import manifest as mf  # noqa: E402
from tools.minerals_labeler import server  # noqa: E402

COLUMNS = ["scan", "point", "rule", "center", "amp", "fwhm", "snr", "pass_sharpness", "r2"]


def comp(scan, point, rule, center, snr=6.0, fwhm=30.0, sharp=True, r2=0.5, amp=100.0):
    return [scan, point, rule, center, amp, fwhm, snr, sharp, r2]


def write_components(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(COLUMNS)
        writer.writerows(rows)
    return path


def write_results(root, scan, points, despiked_offset=None):
    folder = Path(root) / scan
    folder.mkdir(parents=True, exist_ok=True)
    axis = [600.0 + 10.0 * i for i in range(71)]  # 600..1300

    def write(name, offset):
        with open(folder / name, "w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["raman_shift"] + [str(p) for p in points])
            for x in axis:
                writer.writerow([x] + [offset + p + 50 * math.exp(-((x - 1000) / 20) ** 2)
                                       for p in points])

    write("SYN_R1_normalized_baselined.csv", 0.0)
    if despiked_offset is not None:
        write("SYN_R1_normalized_baselined_despiked.csv", despiked_offset)
    return root


@pytest.fixture
def synthetic(tmp_path):
    scan = "SYNTHETIC_target/0001_detail"
    rows = [
        # point 0: aicc keeps 1000 and 850, ftest keeps 1005 (matches 1000 within 10) -> 850 aicc-only
        comp(scan, 0, "aicc", 1000.0), comp(scan, 0, "aicc", 850.0, snr=3.5),
        comp(scan, 0, "ftest", 1005.0),
        # point 1: ftest-only 900; an aicc 960 component below the SNR gate is fitted but not kept
        comp(scan, 1, "ftest", 900.0, snr=4.2), comp(scan, 1, "aicc", 960.0, snr=2.9),
        # point 2: nothing disputed because each candidate fails one persist bar
        comp(scan, 2, "aicc", 800.0, fwhm=24.9), comp(scan, 2, "aicc", 820.0, sharp=False),
        comp(scan, 2, "ftest", 1100.0, r2=0.0),
        # point 3: centers 10.0 apart still match; 10.5 apart do not
        comp(scan, 3, "aicc", 1000.0), comp(scan, 3, "ftest", 1010.0),
        comp(scan, 3, "aicc", 760.0), comp(scan, 3, "ftest", 770.5),
        # gain2 sheet rows
        comp(scan, 4, "aicc", 1000.0, snr=9.0), comp(scan, 4, "aicc", 1080.0, snr=3.2),
        comp(scan, 4, "aicc_gain2", 1001.0, snr=9.0),
        comp(scan, 5, "aicc", 830.0, snr=5.5), comp(scan, 5, "aicc_gain2", 1150.0, snr=7.0),
    ]
    components = write_components(tmp_path / "comps.csv", rows)
    results = write_results(tmp_path / "results", scan, range(6), despiked_offset=1e6)
    flags = tmp_path / "flags.csv"
    flags.write_text("scan,point,center_cm1,mask_px_in_fwhm\n" f"{scan},0,850,2\n", encoding="utf-8")
    return {"scan": scan, "components": components, "results": results, "flags": flags,
            "tmp": tmp_path}


def keyed(rows):
    return sorted((r["point"], r["only_in"], r["comp"]["center_cm1"]) for r in rows)


def test_disputed_rows_persist_rule_and_match_tolerance(synthetic):
    groups = mf.read_components(synthetic["components"])
    rows = mf.disputed_rows(groups, ("aicc", "ftest"))
    assert keyed(rows) == [(0, "aicc", 850.0), (1, "ftest", 900.0),
                           (3, "aicc", 760.0), (3, "ftest", 770.5),
                           (4, "aicc", 1000.0), (4, "aicc", 1080.0), (5, "aicc", 830.0)]


def test_disputed_manifest_is_reproducible_and_blind_assignment_is_seeded(synthetic):
    args = (synthetic["components"], synthetic["results"], "disputed", 20261112, synthetic["flags"])
    one, two = mf.build_manifest(*args), mf.build_manifest(*args)
    assert json.dumps(one) == json.dumps(two)
    other = mf.build_manifest(synthetic["components"], synthetic["results"], "disputed", 7)
    assert [i["unit_id"] for i in other["items"]] != [i["unit_id"] for i in one["items"]] or \
        [i["fit_rules"] for i in other["items"]] != [i["fit_rules"] for i in one["items"]]
    header = one["header"]
    assert header["n_items"] == len(one["items"]) == 7
    assert header["seed"] == 20261112 and header["rules"] == ["aicc", "ftest"]
    assert set(v["label"] for v in header["labels"].values()) == server.LABELS
    assert header["components_sha256"] == hashlib.sha256(
        synthetic["components"].read_bytes()).hexdigest()
    for item in one["items"]:
        assert sorted(item["fit_rules"].values()) == ["aicc", "ftest"]
        x = item["window"]["x"]
        assert x[0] >= 700 and x[-1] <= 1200 and len(x) == 51
        assert max(item["window"]["y"]) < 1e5  # the despiked file is never read
        rule_of = item["fit_rules"]
        mine = [c for c in item["fits"][next(k for k, v in rule_of.items() if v == item["only_in"])]
                if c["center_cm1"] == item["band"]["center_cm1"]]
        assert mine and mine[0]["kept"]
    flagged = {i["band"]["center_cm1"]: i["flags"]["mask_px_in_band"] for i in one["items"]}
    assert flagged[850.0] == 2 and flagged[900.0] == 0
    unkept = [c for i in one["items"] if i["point"] == 1 for f in i["fits"].values() for c in f
              if not c["kept"]]
    assert [c["center_cm1"] for c in unkept] == [960.0]


def test_gain2_sheet_lists_lost_components_by_snr(synthetic):
    doc = mf.build_manifest(synthetic["components"], synthetic["results"], "gain2")
    assert doc["header"]["rules"] == ["aicc", "aicc_gain2"]
    assert doc["header"]["order"] == "snr ascending"
    # A point with no aicc_gain2 rows has zero components under that rule (as a failed fit
    # has under the F-test in the comparison set), so its kept aicc components count as lost.
    assert [(i["point"], i["only_in"], i["band"]["center_cm1"]) for i in doc["items"]] == [
        (4, "aicc", 1080.0), (0, "aicc", 850.0), (5, "aicc", 830.0), (0, "aicc", 1000.0),
        (3, "aicc", 760.0), (3, "aicc", 1000.0), (5, "aicc_gain2", 1150.0)]
    assert [i["rank"] for i in doc["items"]] == list(range(7))


def test_unknown_sheet_and_ambiguous_spectra_are_refused(synthetic):
    with pytest.raises(ValueError):
        mf.build_manifest(synthetic["components"], synthetic["results"], "other")
    second = Path(synthetic["results"]) / synthetic["scan"] / "SYN2_R1_normalized_baselined.csv"
    second.write_text("raman_shift,0\n700,1\n", encoding="utf-8")
    with pytest.raises(ValueError):
        mf.build_manifest(synthetic["components"], synthetic["results"])


def test_cli_writes_manifest(synthetic):
    out = synthetic["tmp"] / "m.json"
    mf.main(["--components", str(synthetic["components"]), "--results",
             str(synthetic["results"]), "--out", str(out), "--sheet", "disputed"])
    assert json.loads(out.read_text())["header"]["n_items"] == 7


@pytest.fixture
def manifest_file(synthetic):
    doc = mf.build_manifest(synthetic["components"], synthetic["results"], "disputed", 3,
                            synthetic["flags"])
    path = synthetic["tmp"] / "manifest.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path, doc


@contextmanager
def running(manifest, out, **kwargs):
    app = server.create_server(str(manifest), str(out), "tester", port=0, **kwargs)
    thread = threading.Thread(target=app.serve_forever, daemon=True)
    thread.start()
    try:
        yield app
    finally:
        app.shutdown()
        app.server_close()


def request(app, path, payload=None):
    connection = http.client.HTTPConnection(*app.server_address[:2], timeout=5)
    body = None if payload is None else json.dumps(payload).encode()
    headers = {} if payload is None else {"Content-Type": "application/json"}
    connection.request("GET" if payload is None else "POST", path, body=body, headers=headers)
    response = connection.getresponse()
    data = response.read()
    connection.close()
    return response.status, data, response


def test_items_served_in_manifest_order_with_only_public_fields(manifest_file, tmp_path):
    path, doc = manifest_file
    with running(path, tmp_path / "out") as app:
        status, raw, response = request(app, "/api/state")
        assert status == 200 and json.loads(raw)["n_total"] == 7
        assert "default-src 'self'" in response.getheader("Content-Security-Policy")
        served = []
        for k in range(7):
            status, raw, _ = request(app, f"/api/item/{k}")
            assert status == 200
            item = json.loads(raw)
            assert set(item) == {"id", "k", "n_total", "window", "band", "fits", "flags"}
            assert set(item["fits"]) == {"A", "B"}
            for fit in item["fits"].values():
                for c in fit:
                    assert set(c) == {"center_cm1", "amplitude", "fwhm_cm1", "snr", "kept"}
            text = raw.decode()
            for private in ("SYNTHETIC_target", "aicc", "ftest", "unit_id", "only_in", "rank",
                            "fit_rules", "disputed"):
                assert private not in text
            served.append(item["band"]["center_cm1"])
        assert served == [i["band"]["center_cm1"] for i in doc["items"]]
        for asset in ("/", "/static/app.js", "/static/app.css"):
            status, raw, _ = request(app, asset)
            assert status == 200 and b"aicc" not in raw.lower() and b"ftest" not in raw.lower()
        assert request(app, "/api/item/7")[0] == 404


def test_label_finish_freeze_and_restart(manifest_file, tmp_path):
    path, doc = manifest_file
    out = tmp_path / "out"
    with running(path, out) as app:
        ids = [json.loads(request(app, f"/api/item/{k}")[1])["id"] for k in range(7)]
        assert request(app, "/api/label", {"id": ids[0], "label": "real"})[0] == 400
        for opaque, label in zip(ids, ["present", "absent", "spike", "unsure", "present"]):
            assert request(app, "/api/label", {"id": opaque, "label": label})[0] == 200
        assert request(app, "/api/label", {"id": ids[0], "label": "absent"})[0] == 200
        assert json.loads(request(app, "/api/state")[1])["next_index"] == 5
        assert request(app, "/api/finish", {})[0] == 200
        assert request(app, "/api/label", {"id": ids[5], "label": "present"})[0] == 409
    final = (out / "labels-final.json").read_bytes()
    assert (out / "labels-final.sha256").read_text().strip() == hashlib.sha256(final).hexdigest()
    labels = json.loads(final)["labels"]
    order = [i["unit_id"] for i in doc["items"]]
    assert labels == {order[0]: "absent", order[1]: "absent", order[2]: "spike",
                      order[3]: "unsure", order[4]: "present"}
    assert json.loads(final)["manifest_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    journal = [json.loads(line) for line in (out / "labels.jsonl").read_text().splitlines()]
    assert len(journal) == 6 and all(re.fullmatch(r"[0-9a-f]{16}", e["id"]) for e in journal)
    assert oct((out / "key").stat().st_mode & 0o777) == "0o600"
    with running(path, out) as app:
        assert json.loads(request(app, "/api/state")[1])["frozen"] is True


def test_resume_refuses_a_changed_manifest(manifest_file, tmp_path):
    path, doc = manifest_file
    out = tmp_path / "out"
    with running(path, out):
        pass
    doc["items"] = doc["items"][::-1]
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(ValueError):
        server.create_server(str(path), str(out), "tester", port=0)


@pytest.mark.parametrize("host", ["0.0.0.0", "192.0.2.1", "example.org"])
def test_nonloopback_refused_before_io(manifest_file, tmp_path, host):
    path, _ = manifest_file
    with pytest.raises(ValueError, match="Non-loopback"):
        server.create_server(str(path), str(tmp_path / "out"), "tester", host=host, port=0)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("defect", ["no_b", "bad_fwhm", "no_kept", "bad_band", "bad_flag",
                                    "no_check"])
def test_invalid_manifests_are_refused(manifest_file, tmp_path, defect):
    path, doc = manifest_file
    item = doc["items"][0]
    if defect == "no_b":
        del item["fits"]["B"]
    elif defect == "bad_fwhm":
        item["fits"]["A"][0]["fwhm_cm1"] = 0
    elif defect == "no_kept":
        del item["fits"]["A"][0]["kept"]
    elif defect == "bad_band":
        item["band"]["center_cm1"] = "1000"
    elif defect == "bad_flag":
        item["flags"]["mask_px_in_band"] = -1
    else:
        doc["header"]["public_check"] = " "
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(ValueError):
        server.Session(path, tmp_path / "out", "tester")


def test_static_ui_offers_exactly_the_four_classes():
    static = Path(server.STATIC_DIR)
    html = (static / "index.html").read_text(encoding="utf-8")
    script = (static / "app.js").read_text(encoding="utf-8")
    assert set(re.findall(r'data-label="([a-z]+)"', html)) == server.LABELS
    keys = dict(re.findall(r'"([1-4])": "([a-z]+)"', script))
    assert keys == {k: v[0] for k, v in mf.LABELS.items()}
