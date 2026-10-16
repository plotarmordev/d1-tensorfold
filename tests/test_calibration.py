"""ECE, bins and the row bootstrap on hand-computed inputs; a whole run against a stub System One server, with
its serving provenance read from stubbed `docker inspect` output and a stand-in checkpoint."""
import copy
import hashlib
import http.server
import json
import subprocess
import sys
import tempfile
import threading
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
import calibration as C
import calibration_chart


class Bins(unittest.TestCase):
    def test_edges_are_ten_equal_widths(self):
        self.assertEqual(C.edges(), [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    def test_a_bin_holds_its_upper_edge_and_the_first_holds_zero(self):
        for confidence, expected in {0.0: 0, 0.05: 0, 0.1: 0, 0.1000001: 1, 0.3: 2, 0.3000001: 3, 0.7: 6,
                                     0.95: 9, 1.0: 9}.items():
            self.assertEqual(C.bin_of(confidence), expected, confidence)


class Ece(unittest.TestCase):
    def test_a_calibrated_set_scores_zero(self):
        items = ([(0.55, i < 11) for i in range(20)] + [(0.75, i < 3) for i in range(4)]
                 + [(0.95, i < 19) for i in range(20)])
        self.assertAlmostEqual(C.ece(C.tally(items)), 0, places=12)

    def test_constant_over_confidence_scores_its_gap(self):
        t = C.tally([(0.9, i < 6) for i in range(10)])
        self.assertAlmostEqual(C.accuracy(t), 0.6)
        self.assertAlmostEqual(C.ece(t), 0.3)

    def test_bins_are_weighted_by_count(self):
        # 30/40 * |1 - 0.9| + 10/40 * |0 - 0.6| = 0.225; an unweighted mean of the two gaps would be 0.35
        self.assertAlmostEqual(C.ece(C.tally([(0.9, True)] * 30 + [(0.6, False)] * 10)), 0.225)


class Bootstrap(unittest.TestCase):
    def test_identical_rows_give_a_point_interval(self):
        lo, hi = C.interval([C.ece(t) for t in C.resample([[(0.9, True), (0.9, False)]] * 50, samples=200)])
        self.assertAlmostEqual(lo, 0.4)
        self.assertAlmostEqual(hi, 0.4)

    def test_rows_are_drawn_whole(self):
        # a row of one right answer and a row of three wrong ones: accuracy 1, 1/4 or 0, never 1/2
        boot = C.resample([[(0.9, True)], [(0.9, False)] * 3], samples=500)
        self.assertEqual({C.accuracy(t) for t in boot}, {0.0, 0.25, 1.0})

    def test_the_seed_fixes_the_draws(self):
        groups = [[(0.6 + i % 4 / 10, i % 3 > 0)] for i in range(40)]
        self.assertEqual(C.resample(groups), C.resample(groups))
        self.assertNotEqual(C.resample(groups, seed=1), C.resample(groups))

    def test_interval_is_the_central_95_percent(self):
        self.assertEqual(C.interval(list(range(1001))), [25, 975])


def head(task, gold, labels, multi=False):
    return {"task": task, "true_label": gold, "labels": labels, "multi_label": multi}


DATASET = {
    "a.jsonl": [
        {"input": "first", "output": {"classifications": [
            head("topic", ["billing"], ["billing", "shipping", "other"]), head("urgent", ["no"], ["yes", "no"]),
            head("areas", ["x", "y"], ["x", "y", "z"], multi=True)]}},
        {"input": "second", "output": {"classifications": [head("topic", ["shipping"], ["billing", "shipping"])]}},
    ],
    "b.jsonl": [{"input": "third", "output": {"classifications": [head("doc_type", ["memo"], ["memo", "invoice"])]}}],
}
MODEL_REVISION = "051bcc464b01b9f92942b364d9586b0ef5912432"
IMAGE = "sha256:" + "5e" * 32
SETTINGS = {"precision": "bf16", "token_budget": "65536", "cuda_graphs": "1", "tf32_override": "0", "context": "4096"}


class Stub(http.server.BaseHTTPRequestHandler):
    """Answers every choice with its first option at 0.7, the rest sharing 0.3."""

    def reply(self, body):
        raw = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        self.reply({"status": "ok"} if self.path == "/health" else {"object": "list", "data": [{"id": "d1-stub"}]})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.bodies.append(body)
        answers = {}
        for qid, q in body["questions"].items():
            options = list(q["criteria"])
            p = {o: 0.7 if i == 0 else 0.3 / (len(options) - 1) for i, o in enumerate(options)}
            answers[qid] = {"type": "choice", "choice": options[0], "confidence": 0.7, "probabilities": p}
        self.reply({"model": "d1-stub", "answers": answers, "usage": {"input_tokens": 1, "output_tokens": 0}})

    def log_message(self, *args):
        pass


class Run(unittest.TestCase):
    def setUp(self):
        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Stub)
        self.server.bodies = []
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.data, self.out, self.model = root / "data", root / "out", root / "model"
        self.data.mkdir()
        self.pinned = {}
        for name, records in DATASET.items():
            raw = "".join(json.dumps(r) + "\n" for r in records).encode()
            (self.data / name).write_bytes(raw)
            self.pinned[name] = hashlib.sha256(raw).hexdigest()
        # a stand-in checkpoint as `hf download --local-dir` leaves it, and its pins
        (self.model / ".cache/huggingface/download").mkdir(parents=True)
        lines = []
        for name, raw in (("config.json", b"{}"), ("model.safetensors", b"weights")):
            (self.model / name).write_bytes(raw)
            (self.model / ".cache/huggingface/download" / f"{name}.metadata").write_text(
                f"{MODEL_REVISION}\n\"etag\"\n1760000000.0\n")
            lines.append(f"{hashlib.sha256(raw).hexdigest()}  {name}\n")
        (root / "d1-3B.sha256").write_text("".join(lines))
        self.container = {
            "Id": "c0ffee", "Image": IMAGE, "State": {"Running": True, "StartedAt": "2026-10-16T00:00:00Z"},
            "Path": "tensorfold",
            "Args": ["serve", "/model", "--backend", "cuda", "--no-drafts", "--context", "4096", "--host", "0.0.0.0",
                     "--port", "8080", "--name", "d1-stub"],
            "Config": {"Image": "d1-tensorfold:feedface",
                       "Env": ["PATH=/usr/bin", "TENSORFOLD_LFM2_PRECISION=bf16", "TENSORFOLD_LFM2_TOKEN_BUDGET=65536",
                               "TENSORFOLD_LFM2_GRAPHS=1", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0"]},
            "HostConfig": {"PortBindings": {"8080/tcp": [{"HostIp": "127.0.0.1",
                                                          "HostPort": str(self.server.server_port)}]}},
            "Mounts": [{"Destination": "/cache", "Source": "/elsewhere"},
                       {"Destination": "/model", "Source": str(self.model)}],
        }
        self.daemon = "unix:///var/run/docker.sock"
        for target, value in (("MODEL_PINS", root / "d1-3B.sha256"), ("docker", self.docker)):
            patcher = mock.patch.object(C, target, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def docker(self, *args):
        if args == ("context", "inspect"):
            return {"Name": "default", "Endpoints": {"docker": {"Host": self.daemon}}}
        if args[:2] == ("container", "inspect"):
            return copy.deepcopy(self.container)
        return {"Id": IMAGE, "RepoTags": ["d1-tensorfold:feedface"], "RepoDigests": [],
                "Config": {"Labels": {"d1.image-hash": "feedface"}}}

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.tmp.cleanup()

    def test_run_score_and_chart(self):
        C.run(self.url, self.data, self.pinned, self.out)
        self.assertEqual([b["state"] for b in self.server.bodies], ["first", "second", "third"])
        self.assertEqual(self.server.bodies[0]["questions"], {
            "topic": {"type": "choice", "instructions": "Choose the topic label.",
                      "criteria": {"billing": None, "shipping": None, "other": None}},
            "urgent": {"type": "choice", "instructions": "Choose the urgent label.",
                       "criteria": {"yes": None, "no": None}}})
        answers = [json.loads(line) for line in (self.out / "answers.jsonl").read_text().splitlines()]
        self.assertEqual([(a["row"], a["task"], a["gold"], a["choice"]) for a in answers], [
            ("a.jsonl:0", "topic", "billing", "billing"), ("a.jsonl:0", "urgent", "no", "yes"),
            ("a.jsonl:1", "topic", "shipping", "billing"), ("b.jsonl:0", "doc_type", "memo", "memo")])
        manifest = json.loads((self.out / "manifest.json").read_text())
        self.assertEqual(manifest["dataset"]["revision"], C.REVISION)
        self.assertEqual(manifest["dataset"]["files_sha256"], self.pinned)
        self.assertEqual(manifest["server_reported"]["models"]["data"][0]["id"], "d1-stub")
        self.assertEqual(manifest["server_reported"]["health"], {"status": "ok"})
        served = manifest["serving"]
        self.assertEqual(served["source"], "docker inspect")
        self.assertEqual(served["container"], {"name": "d1-tensorfold", "id": "c0ffee",
                                               "started": "2026-10-16T00:00:00Z",
                                               "daemon": "unix:///var/run/docker.sock"})
        self.assertEqual(served["image"], {"id": IMAGE, "tag": "d1-tensorfold:feedface", "repo_digests": [],
                                           "recipe_hash": "feedface"})
        self.assertEqual(served["settings"], SETTINGS)
        self.assertEqual(served["command"][:2], ["tensorfold", "serve"])
        self.assertNotIn("PATH", served["env"])
        self.assertEqual(served["model"], {"dir": str(self.model), "revision": MODEL_REVISION,
                                           "files_sha256": C.pins(C.MODEL_PINS)})
        self.assertEqual((manifest["requests"], manifest["questions"]), (3, 4))
        self.assertRegex(manifest["recipe"]["commit"], "^[0-9a-f]{40}$")

        summary = C.score(self.out)
        self.assertEqual(summary["run"]["serving"], served)
        self.assertEqual(summary["accuracy"]["value"], 0.5)
        self.assertEqual(summary["ece"]["value"], 0.2)
        self.assertEqual([b["n"] for b in summary["bins"]], [0, 0, 0, 0, 0, 0, 4, 0, 0, 0])
        self.assertEqual(summary["bins"][6]["accuracy_minus_confidence"], -0.2)
        for render in (calibration_chart.reliability, calibration_chart.gaps):
            svg = render(summary)
            ET.fromstring(svg)
            self.assertIn("n = 4 questions in 3 rows", svg)
            self.assertIn("d1-stub", svg)
            self.assertIn(f"Image {'5e' * 6}, model revision 051bcc46, bf16, context 4096, token budget 65536", svg)

        with self.assertRaises(SystemExit):
            C.run(self.url, self.data, self.pinned, self.out)

    def test_a_changed_dataset_file_is_refused_before_any_request(self):
        with (self.data / "b.jsonl").open("a") as f:
            f.write("\n")
        with self.assertRaisesRegex(ValueError, "sha256"):
            C.run(self.url, self.data, self.pinned, self.out)
        self.assertEqual(self.server.bodies, [])

    def test_an_edited_answer_file_is_refused(self):
        C.run(self.url, self.data, self.pinned, self.out)
        with (self.out / "answers.jsonl").open("a") as f:
            f.write("\n")
        with self.assertRaisesRegex(ValueError, "manifest"):
            C.score(self.out)

    def test_missing_serving_evidence_stops_the_run_before_any_question(self):
        def unavailable(*args):
            raise subprocess.CalledProcessError(1, "docker", stderr="Error: No such container: d1-tensorfold\n")

        # each change, and the reason the run gives for refusing it
        cases = {
            "is not running": lambda c: c["State"].update(Running=False),
            "does not publish": lambda c: c["HostConfig"]["PortBindings"]["8080/tcp"][0].update(HostPort="1"),
            "mounts no checkpoint": lambda c: c.update(Mounts=[]),
            "settings.cuda_graphs": lambda c: c["Config"]["Env"].remove("TENSORFOLD_LFM2_GRAPHS=1"),
            "settings.context": lambda c: c.update(Args=["serve", "/model"]),
        }
        for reason, change in cases.items():
            with self.subTest(reason):
                original = copy.deepcopy(self.container)
                change(self.container)
                with self.assertRaisesRegex(SystemExit, reason):
                    C.run(self.url, self.data, self.pinned, self.out)
                self.container = original
        with (self.subTest("no docker"), mock.patch.object(C, "docker", unavailable),
              self.assertRaisesRegex(SystemExit, "--provenance")):
            C.run(self.url, self.data, self.pinned, self.out)
        metadata = self.model / ".cache/huggingface/download/model.safetensors.metadata"
        for path, raw, error, reason in ((self.model / "model.safetensors", b"other", ValueError, "sha256"),
                                         (metadata, b"0" * 40, SystemExit, "one `hf download` revision")):
            with self.subTest(reason):
                kept = path.read_bytes()
                path.write_bytes(raw)
                with self.assertRaisesRegex(error, reason):
                    C.run(self.url, self.data, self.pinned, self.out)
                path.write_bytes(kept)
        self.assertEqual(self.server.bodies, [])
        self.assertFalse(self.out.exists())

    def test_only_a_loopback_endpoint_the_container_publishes_is_vouched_for(self):
        port = str(self.server.server_port)
        binding = self.container["HostConfig"]["PortBindings"]["8080/tcp"][0]
        for host, bound in (("127.0.0.1", "0.0.0.0"), ("127.0.0.1", "::"), ("127.0.0.1", ""),
                            ("localhost", "127.0.0.1"), ("localhost", "::1"), ("[::1]", "::1")):
            with self.subTest(f"{host} on {bound or 'all'}"):
                binding["HostIp"] = bound
                self.assertEqual(C.container("d1-tensorfold", f"http://{host}:{port}")[0]["container"]["id"], "c0ffee")
        for host, bound, reason in (("unrelated.example", "0.0.0.0", "not a loopback address"),
                                    ("192.0.2.7", "0.0.0.0", "not a loopback address"),
                                    ("127.0.0.1", "192.0.2.7", "does not publish"),
                                    ("127.0.0.1", "127.0.0.2", "does not publish"),
                                    ("[::1]", "127.0.0.1", "does not publish")):
            with self.subTest(f"{host} on {bound}"):
                binding["HostIp"] = bound
                with self.assertRaisesRegex(SystemExit, reason):
                    C.container("d1-tensorfold", f"http://{host}:{port}")
        binding["HostIp"] = "127.0.0.1"
        for daemon in ("tcp://192.0.2.1:2375", "ssh://builder@192.0.2.1"):
            with self.subTest(daemon):
                self.daemon = daemon
                with self.assertRaisesRegex(SystemExit, "is not local"):
                    C.run(self.url, self.data, self.pinned, self.out)
        self.assertEqual(self.server.bodies, [])

    def test_supplied_provenance_stands_in_for_docker(self):
        def unavailable(*args):
            raise FileNotFoundError("docker")

        evidence = {"image": {"id": IMAGE}, "model": {"revision": MODEL_REVISION, "files_sha256": {"x": "0"}},
                    "settings": SETTINGS}
        path = Path(self.tmp.name) / "serving.json"
        path.write_text(json.dumps(evidence))
        with mock.patch.object(C, "docker", unavailable):
            C.run(self.url, self.data, self.pinned, self.out, provenance=path)
            served = json.loads((self.out / "manifest.json").read_text())["serving"]
            self.assertEqual(served, {"source": "file", "file": str(path), "file_sha256": C.sha256(path), **evidence})
            path.write_text(json.dumps({**evidence, "settings": {k: v for k, v in SETTINGS.items() if k != "context"}}))
            with self.assertRaisesRegex(SystemExit, "settings.context"):
                C.run(self.url, self.data, self.pinned, Path(self.tmp.name) / "again", provenance=path)

    def test_a_restarted_container_voids_the_run(self):
        inspections = []

        def restarting(*args):
            out = self.docker(*args)
            if args[:2] == ("container", "inspect"):
                inspections.append(args)
                if len(inspections) > 1:
                    out["State"]["StartedAt"] = "2026-10-16T01:00:00Z"
            return out

        with mock.patch.object(C, "docker", restarting), self.assertRaisesRegex(SystemExit, "restarted"):
            C.run(self.url, self.data, self.pinned, self.out)
        self.assertFalse((self.out / "manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
