#!/usr/bin/env python3
"""Unit and end-to-end tests for user/mark/Mdask.py."""

import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import unittest


def load_module(path):
    spec = importlib.util.spec_from_file_location("madagascar_sfdask", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


if len(sys.argv) > 1 and sys.argv[1].endswith(".py") and os.path.isfile(sys.argv[1]):
    os.environ["SFDASK_TEST_MODULE"] = os.path.abspath(sys.argv[1])
    del sys.argv[1]
MODULE = load_module(os.environ["SFDASK_TEST_MODULE"])


def command_exists(name):
    return shutil.which(name) is not None


class PureTests(unittest.TestCase):
    def test_balanced_partitions_matches_sf_split(self):
        parts = MODULE.balanced_partitions(11, 4)
        self.assertEqual(
            [(part.start, part.count) for part in parts],
            [(0, 3), (3, 3), (6, 3), (9, 2)],
        )
        self.assertEqual(len(MODULE.balanced_partitions(2, 8)), 2)

    def test_parse_quoted_pipeline_and_auxiliary_files(self):
        parsed = MODULE.parse_cli(
            [
                "filein=in.mdio",
                "informat=mdio",
                "_velocity=vel.rsf",
                "__tfile=head.rsf",
                "--",
                "sfscale dscale=2 | sfbandpass",
            ]
        )
        self.assertEqual(parsed.command, "sfscale dscale=2 | sfbandpass")
        self.assertEqual(parsed.aux_inputs, {"velocity": "vel.rsf"})
        self.assertEqual(parsed.aux_outputs, {"tfile": "head.rsf"})

    def test_unknown_wrapper_option_fails(self):
        with self.assertRaises(MODULE.DaskRsfError):
            MODULE.parse_cli(["bogus=y", "--", "sfscale"])


@unittest.skipUnless(
    all(command_exists(tool) for tool in ("sfspike", "sfscale", "sfadd", "sfattr")),
    "Madagascar command-line programs are unavailable",
)
class RsfIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from distributed import Client, LocalCluster
        except ImportError as exc:
            raise unittest.SkipTest("dask.distributed is unavailable") from exc
        cls.Client = Client
        cls.LocalCluster = LocalCluster

    def setUp(self):
        self.work = tempfile.mkdtemp(prefix="sfdask-test-")
        self.data = os.path.join(self.work, "data")
        os.makedirs(self.data)
        self.env = os.environ.copy()
        self.env["DATAPATH"] = self.data + os.sep
        self.input = os.path.join(self.work, "input.rsf")
        self._run(
            ["sfspike", "n1=32", "n2=11", "n3=3", "k1=7", "k2=4", "k3=2"],
            stdout=self.input,
            stdin_null=True,
        )

    def tearDown(self):
        shutil.rmtree(self.work, ignore_errors=True)

    def _run(self, command, stdin=None, stdout=None, stdin_null=False):
        source = (
            subprocess.DEVNULL
            if stdin_null
            else (open(stdin, "rb") if stdin is not None else None)
        )
        target = open(stdout, "wb") if stdout is not None else subprocess.PIPE
        try:
            proc = subprocess.run(
                command,
                stdin=source,
                stdout=target,
                stderr=subprocess.PIPE,
                env=self.env,
                check=False,
            )
        finally:
            if source not in (None, subprocess.DEVNULL):
                source.close()
            if stdout is not None:
                target.close()
        if proc.returncode:
            self.fail(
                "%s failed: %s"
                % (" ".join(command), proc.stderr.decode("utf-8", "replace"))
            )
        return proc

    def _assert_same_rsf(self, first, second):
        with open(first, "rb") as source:
            diff = subprocess.Popen(
                ["sfadd", "scale=1,-1", second],
                stdin=source,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=self.env,
            )
            attr = subprocess.run(
                ["sfattr", "want=nonzero"],
                stdin=diff.stdout,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=self.env,
                check=False,
            )
            diff.stdout.close()
            diff.wait()
        text = attr.stdout.decode("utf-8", "replace")
        diff_error = diff.stderr.read().decode()
        diff.stderr.close()
        self.assertEqual(diff.returncode, 0, diff_error)
        self.assertEqual(attr.returncode, 0, attr.stderr.decode())
        self.assertRegex(text, r"nonzero(?:\s+samples)?\s*=\s*0(?:\.0*)?\b")

    def test_supplied_client_remains_open_and_pipeline_matches_serial(self):
        serial = os.path.join(self.work, "serial.rsf")
        actual = os.path.join(self.work, "actual.rsf")
        self._run(
            ["bash", "-o", "pipefail", "-c", "sfscale dscale=2 | sfbandpass"],
            stdin=self.input,
            stdout=serial,
        )
        with self.LocalCluster(
            n_workers=2, threads_per_worker=1, processes=False
        ) as cluster:
            client = self.Client(cluster)
            MODULE.run(
                "sfscale dscale=2 | sfbandpass",
                filein=self.input,
                fileout=actual,
                split=2,
                client=client,
                tempdir=self.work,
            )
            self.assertEqual(client.status, "running")
            client.close()
        self._assert_same_rsf(serial, actual)

    def test_scheduler_address_and_add_reduction(self):
        serial = os.path.join(self.work, "stack-serial.rsf")
        actual = os.path.join(self.work, "stack-dask.rsf")
        self._run(["sfstack", "axis=2"], stdin=self.input, stdout=serial)
        with self.LocalCluster(
            n_workers=2, threads_per_worker=1, processes=False
        ) as cluster:
            MODULE.run(
                "sfstack axis=2",
                filein=self.input,
                fileout=actual,
                split=2,
                join=0,
                scheduler=cluster.scheduler_address,
                tasks=2,
                tempdir=self.work,
            )
        self._assert_same_rsf(serial, actual)

    def test_local_client_and_auxiliary_input(self):
        if not command_exists("sfmath"):
            self.skipTest("sfmath unavailable")
        serial = os.path.join(self.work, "aux-serial.rsf")
        actual = os.path.join(self.work, "aux-dask.rsf")
        self._run(["sfscale", "dscale=2"], stdin=self.input, stdout=serial)
        MODULE.run(
            "sfmath output='input+velocity'",
            filein=self.input,
            fileout=actual,
            split=2,
            workers=2,
            threads=1,
            tasks=2,
            tempdir=self.work,
            aux_inputs={"velocity": self.input},
        )
        self._assert_same_rsf(serial, actual)

    def test_worker_failure_cleans_temporary_directory(self):
        before = set(os.listdir(self.work))
        with self.assertRaises(Exception):
            MODULE.run(
                "bash -c 'exit 7'",
                filein=self.input,
                fileout=os.path.join(self.work, "never.rsf"),
                split=2,
                workers=1,
                threads=1,
                tempdir=self.work,
            )
        after = set(os.listdir(self.work))
        self.assertFalse(any(name.startswith("sfdask-") for name in after - before))

    def test_parallel_mdio_roundtrip_when_supported(self):
        if not all(command_exists(tool) for tool in ("sfmdiowrite", "sfmdioread")):
            self.skipTest("MDIO executables unavailable")
        source_mdio = os.path.join(self.work, "source.mdio")
        output_mdio = os.path.join(self.work, "output.mdio")
        serial = os.path.join(self.work, "mdio-serial.rsf")
        actual = os.path.join(self.work, "mdio-actual.rsf")

        with open(self.input, "rb") as source:
            create = subprocess.run(
                ["sfmdiowrite", "mdio=%s" % source_mdio],
                stdin=source,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=self.env,
                check=False,
            )
        if create.returncode:
            self.skipTest("MDIO support unavailable: " + create.stderr.decode())

        self._run(
            ["bash", "-o", "pipefail", "-c", "sfscale dscale=3 | sfbandpass"],
            stdin=self.input,
            stdout=serial,
        )
        MODULE.run(
            "sfscale dscale=3 | sfbandpass",
            filein=source_mdio,
            informat="mdio",
            fileout=output_mdio,
            outformat="mdio",
            split=3,
            workers=2,
            threads=1,
            tasks=2,
            tempdir=self.work,
        )
        self._run(
            ["sfmdioread", "read=d", "mdio=%s" % output_mdio],
            stdout=actual,
            stdin_null=True,
        )
        self._assert_same_rsf(serial, actual)


if __name__ == "__main__":
    unittest.main(verbosity=2)
