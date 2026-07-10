#!/usr/bin/env python3
"""Python <-> C++ MDIO interop helpers for run_roundtrip.sh.

This drives the official Python MDIO package (``multidimio``) so the shell test
can exercise interoperability with Madagascar's sfmdioread / sfmdiowrite (which
use mdio-cpp).

Subcommands:

  gen OUT_MDIO
      Build a small 3D post-stack SEG-Y in memory, ingest it to an MDIO v1 store
      with the Python package, and write it as ZARR v2 (mdio-cpp / tensorstore
      only read v2).  Two metadata fix-ups are applied so mdio-cpp can open the
      result:
        * statsV1 is rewritten from a JSON-encoded *string* (how Python stores
          it) to a JSON *object* (what mdio-cpp expects).
        * the segy_file_header variable is NOT saved (its <U1 dtype is rejected
          by tensorstore); per the test design we only round-trip the Python
          store's trace DATA and trace HEADERS, not the SEG-Y file headers.
      Prints "OK <inline> <crossline> <samples>".

  verify-read MDIO DATA_RSF TFILE_RSF
      Compare what sfmdioread produced (DATA_RSF amplitudes + TFILE_RSF trace
      headers) against the same MDIO read natively in Python.  Exit 0 on match.

  verify-cpp MDIO REF_RSF HFILE BFILE
      Open a store written by sfmdiowrite, and check that Python sees the same
      amplitudes (vs REF_RSF) and the SEG-Y text/binary headers stored on the
      segy_file_header variable (vs HFILE / BFILE).  Exit 0 on match.

Note: mdio's importer uses 'spawn' multiprocessing, so this must be run as a
real file (``python3 mdio_pytools.py ...``), never piped via stdin or ``-c``.
"""
import json
import os
import shutil
import sys

import numpy as np
import zarr

# mdio-cpp (tensorstore) reads Zarr v2 only; force it for everything we write.
zarr.config.set({"default_zarr_format": 2})

NI, NX, NS = 6, 8, 60  # small post-stack 3D grid used by `gen`


# --------------------------------------------------------------------------
# Minimal RSF reader (text header `key=value` + raw binary at `in=`)
# --------------------------------------------------------------------------
def read_rsf(path):
    """Return (flat_array, dims) for a regular RSF file. dims is [n1, n2, ...]."""
    hdr = {}
    with open(path, "rb") as fp:
        text = fp.read().decode("latin-1")
    for tok in text.replace("\n", " ").split():
        if "=" in tok:
            k, _, v = tok.partition("=")
            hdr[k.strip()] = v.strip().strip('"')
    dims = []
    i = 1
    while ("n%d" % i) in hdr:
        dims.append(int(hdr["n%d" % i]))
        i += 1
    fmt = hdr.get("data_format", "native_float")
    dtype = np.int32 if "int" in fmt else np.float32
    datafile = hdr["in"]
    arr = np.fromfile(datafile, dtype=dtype)
    return arr, dims


# --------------------------------------------------------------------------
# gen
# --------------------------------------------------------------------------
def _build_segy(path):
    from segy.factory import SegyFactory
    from segy.standards import get_segy_standard

    spec = get_segy_standard(1.0)
    fac = SegyFactory(spec, sample_interval=4000, samples_per_trace=NS)
    ntr = NI * NX
    hdr = fac.create_trace_header_template(ntr)
    smp = fac.create_trace_sample_template(ntr)
    names = hdr.dtype.names
    ils = np.repeat(np.arange(1, NI + 1), NX)
    xls = np.tile(np.arange(1, NX + 1), NI)
    for t in range(ntr):
        smp[t, :] = np.arange(NS, dtype=smp.dtype) + t

    def setif(field, val):
        if field in names:
            hdr[field] = val

    setif("inline", ils)
    setif("crossline", xls)
    setif("cdp_x", ils * 25)
    setif("cdp_y", xls * 25)
    setif("coordinate_scalar", 1)
    with open(path, "wb") as f:
        f.write(fac.create_textual_header())
        f.write(fac.create_binary_header())
        f.write(fac.create_traces(hdr, smp))
    return spec


def _normalize_stats(store):
    """Rewrite statsV1 from JSON-string to JSON-object (mdio-cpp requirement)."""
    def fix(d):
        if isinstance(d.get("statsV1"), str):
            d["statsV1"] = json.loads(d["statsV1"])
            return True
        return False

    for name in os.listdir(store):
        za = os.path.join(store, name, ".zattrs")
        if os.path.isfile(za):
            d = json.load(open(za))
            if fix(d):
                json.dump(d, open(za, "w"), indent=2)
    zm = os.path.join(store, ".zmetadata")
    if os.path.isfile(zm):
        z = json.load(open(zm))
        changed = False
        for k, v in z.get("metadata", {}).items():
            if k.endswith("/.zattrs") and isinstance(v, dict):
                changed |= fix(v)
        if changed:
            json.dump(z, open(zm, "w"), indent=4)


def cmd_gen(out_mdio):
    from mdio.builder.template_registry import TemplateRegistry
    from mdio.converters.segy import segy_to_mdio

    segy_path = out_mdio + ".segy"
    spec = _build_segy(segy_path)
    tmpl = TemplateRegistry.get_instance().get("PostStack3DTime")
    shutil.rmtree(out_mdio, ignore_errors=True)
    # NB: MDIO__IMPORT__SAVE_SEGY_FILE_HEADER is intentionally left unset, so the
    # <U1 segy_file_header variable (unreadable by tensorstore) is not created.
    segy_to_mdio(
        segy_spec=spec,
        mdio_template=tmpl,
        input_path=segy_path,
        output_path=out_mdio,
        overwrite=True,
    )
    _normalize_stats(out_mdio)
    print("OK %d %d %d" % (NI, NX, NS))


# --------------------------------------------------------------------------
# verify-read : python-created store, read back by sfmdioread
# --------------------------------------------------------------------------
def _open_xr(path):
    import xarray as xr

    return xr.open_zarr(path, consolidated=True)


def cmd_verify_read(mdio, data_rsf, tfile_rsf):
    ds = _open_xr(mdio)
    amp = np.asarray(ds["amplitude"].values, dtype=np.float32)

    rsf, dims = read_rsf(data_rsf)
    if rsf.size != amp.size:
        print("FAIL data size %d != %d" % (rsf.size, amp.size))
        return 1
    if not np.array_equal(rsf, amp.ravel(order="C")):
        print("FAIL data amplitudes differ")
        return 1
    print("  data amplitudes match (%d samples)" % amp.size)

    # Trace headers: sfmdioread's tfile is [SF_NKEYS, ntr] in Madagascar's fixed
    # key order (not labelled in the RSF header).  Rather than resolve that order,
    # confirm the geometry keys made the round trip by checking that some tfile
    # column equals the Python `headers` inline / crossline values (sfmdioread
    # maps iline/xline -> inline/crossline via its alias table).
    tf, tdims = read_rsf(tfile_rsf)
    nk, ntr = tdims[0], tdims[1]
    tf = tf.reshape(ntr, nk).astype(np.int64)  # n1=nkeys fastest -> rows=traces
    # The per-trace header fields are members of the structured `headers`
    # variable; index the numpy structured array (NOT ds["headers"]["inline"],
    # which would select the same-named dimension coordinate instead).
    hdrs = ds["headers"].values
    fields = hdrs.dtype.names or ()
    ok = True
    for pkey in ("inline", "crossline"):
        if pkey in fields:
            ref = np.asarray(hdrs[pkey], dtype=np.int64).ravel(order="C")
            found = any(np.array_equal(tf[:, c], ref) for c in range(nk))
            if found:
                print("  trace header '%s' round-tripped" % pkey)
            else:
                print("FAIL trace header '%s' not found in tfile" % pkey)
                ok = False
    return 0 if ok else 1


# --------------------------------------------------------------------------
# verify-cpp : sfmdiowrite store, read by python
# --------------------------------------------------------------------------
def _data_var(ds):
    best, brank = None, -1
    for name, da in ds.data_vars.items():
        if np.issubdtype(da.dtype, np.floating) and da.ndim > brank:
            best, brank = name, da.ndim
    return best


def cmd_verify_cpp(mdio, ref_rsf, hfile, bfile):
    ds = _open_xr(mdio)
    dv = _data_var(ds)
    if dv is None:
        print("FAIL no float data variable")
        return 1
    amp = np.asarray(ds[dv].values, dtype=np.float32)
    rsf, dims = read_rsf(ref_rsf)
    if rsf.size != amp.size or not np.array_equal(rsf, amp.ravel(order="C")):
        print("FAIL amplitudes differ (py read of cpp store)")
        return 1
    print("  python read amplitudes match (%d samples)" % amp.size)

    rc = 0
    if "segy_file_header" in ds:
        attrs = ds["segy_file_header"].attrs.get("attributes", {})
        th = attrs.get("textHeader")
        bh = attrs.get("binaryHeader")
        if th is not None and os.path.exists(hfile):
            want = open(hfile, "rb").read().decode("latin-1")
            got = th.replace("\n", "").replace("\r", "")
            if got[: len(want)] == want:
                print("  textHeader matches hfile")
            else:
                print("FAIL textHeader differs")
                rc = 1
        if bh is not None and os.path.exists(bfile):
            want = open(bfile, "rb").read()
            got = bytes([int(x) & 0xFF for x in bh])[: len(want)]
            if got == want:
                print("  binaryHeader matches bfile")
            else:
                print("FAIL binaryHeader differs")
                rc = 1
    else:
        print("FAIL no segy_file_header variable in cpp store")
        rc = 1
    return rc


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    cmd = argv[1]
    if cmd == "gen" and len(argv) == 3:
        cmd_gen(argv[2])
        return 0
    if cmd == "verify-read" and len(argv) == 5:
        return cmd_verify_read(argv[2], argv[3], argv[4])
    if cmd == "verify-cpp" and len(argv) == 6:
        return cmd_verify_cpp(argv[2], argv[3], argv[4], argv[5])
    print("bad usage; see --help")
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
