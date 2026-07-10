/* Convert an RSF dataset to MDIO.

Writes an RSF amplitude file (and, optionally, its SEG-Y trace headers tfile,
EBCDIC text header hfile, and 400-byte binary header bfile) to the MDIO
(Zarr) format through the mdio-cpp library.

The output may be a local path, or a gs:// or s3:// URL (when mdio-cpp was built
with the corresponding cloud drivers).

A header-copy option (headers= / like= pointing at another MDIO dataset)
copies the text header, binary header, and/or per-trace headers from that
dataset instead of taking them from tfile/hfile/bfile.  The hdrcopy= selector
(text, binary, trace, or all) controls what is copied.
*/

#include <stdio.h>
#include <stdlib.h>

#include <cctype>
#include <cstring>
#include <string>
#include <vector>

#include "mdio2segy.hh"

/* Sanitize an RSF axis label into a valid, unique MDIO dimension name. */
static std::string clean_name(const char* label, int rsfaxis,
                              std::vector<std::string>& used)
{
    std::string s;
    if (label) {
        for (const char* p = label; *p; p++)
            s += (isalnum((unsigned char) *p) || *p == '_') ? *p : '_';
    }
    if (s.empty()) { char b[16]; snprintf(b, sizeof(b), "dim%d", rsfaxis); s = b; }

    std::string base = s;
    int suffix = 1;
    bool clash = true;
    while (clash) {
        clash = false;
        for (size_t i = 0; i < used.size(); i++)
            if (used[i] == s) { clash = true; break; }
        if (clash) { char b[16]; snprintf(b, sizeof(b), "_%d", suffix++); s = base + b; }
    }
    used.push_back(s);
    return s;
}

static bool write_float_buf(mdio::Dataset& ds, const std::string& name,
                            const float* buf, long total)
{
    auto vr = ds.variables.get<mdio::dtypes::float32_t>(name);
    if (!vr.status().ok()) return false;
    auto var = vr.value();
    auto vdr = mdio::from_variable<mdio::dtypes::float32_t>(var);
    if (!vdr.status().ok()) return false;
    auto vd = vdr.value();
    long n = (long) vd.num_samples();
    auto off = vd.get_flattened_offset();
    float* p = static_cast<float*>(vd.get_data_accessor().data());
    for (long i = 0; i < n && i < total; i++) p[off + i] = buf[i];
    return var.Write(vd).status().ok();
}

static bool write_int_var(mdio::Dataset& ds, const std::string& name,
                          const std::vector<int>& col)
{
    auto vr = ds.variables.get<mdio::dtypes::int32_t>(name);
    if (!vr.status().ok()) return false;
    auto var = vr.value();
    auto vdr = mdio::from_variable<mdio::dtypes::int32_t>(var);
    if (!vdr.status().ok()) return false;
    auto vd = vdr.value();
    long n = (long) vd.num_samples();
    auto off = vd.get_flattened_offset();
    mdio::dtypes::int32_t* p =
        static_cast<mdio::dtypes::int32_t*>(vd.get_data_accessor().data());
    for (long i = 0; i < n && i < (long) col.size(); i++)
        p[off + i] = (mdio::dtypes::int32_t) col[(size_t) i];
    return var.Write(vd).status().ok();
}

template <typename T>
static bool chunk_aligned(const mdio::Variable<T>& var,
                          const std::vector<long>& first,
                          const std::vector<long>& count,
                          const std::vector<std::string>& labels,
                          std::string& why)
{
    auto cr = var.get_chunk_shape();
    if (!cr.status().ok()) {
        why = "cannot determine chunk shape";
        return false;
    }
    std::vector<mdio::DimensionIndex> chunks = cr.value();
    auto dom = var.dimensions();
    auto vlabels = dom.labels();
    auto shape = dom.shape();
    if (chunks.size() != vlabels.size()) {
        why = "chunk rank does not match variable rank";
        return false;
    }

    for (size_t v = 0; v < vlabels.size(); v++) {
        int a = -1;
        for (size_t i = 0; i < labels.size(); i++)
            if (std::string(vlabels[v]) == labels[i]) { a = (int) i; break; }
        if (a < 0) continue; /* e.g. a structured-header byte dimension */

        long c = (long) chunks[v];
        long begin = first[(size_t) a];
        long end = begin + count[(size_t) a];
        long extent = (long) shape[v];
        if (c <= 0 || begin % c != 0 || (end != extent && end % c != 0)) {
            char msg[256];
            snprintf(msg, sizeof(msg),
                     "axis %s region [%ld,%ld) is not aligned to chunk %ld"
                     " (extent %ld)",
                     labels[(size_t) a].c_str(), begin, end, c, extent);
            why = msg;
            return false;
        }
    }
    return true;
}

int main(int argc, char* argv[])
{
    sf_init(argc, argv);

    bool verb;
    if (!sf_getbool("verb", &verb)) verb = false;
    /* Verbosity flag */

    sf_file in = sf_input("in");
    if (SF_FLOAT != sf_gettype(in)) sf_error("Need float input");

    char* path = sf_getstring("mdio");
    /* output MDIO dataset (path or gs://, s3:// URL) */
    if (NULL == path) sf_error("Need mdio=");

    char* dataname = sf_getstring("data");
    /* name of the MDIO data variable (default "seismic") */
    std::string datavar = (NULL != dataname) ? dataname : "seismic";

    char* modearg = sf_getstring("mode");
    /* write mode: write (default/full replacement), init/create (create an
       empty full-shape dataset), or patch/update (write one existing region) */
    std::string mode = (NULL != modearg) ? modearg : "write";
    bool init_alias = false, patch_alias = false;
    sf_getbool("init", &init_alias);
    sf_getbool("patch", &patch_alias);
    if (init_alias) mode = "init";
    if (patch_alias) mode = "patch";
    bool initialize = (mode == "init" || mode == "create");
    bool patch = (mode == "patch" || mode == "update");
    if (!initialize && !patch && mode != "write")
        sf_error("Unknown mode=%s (use write, init/create, or patch/update)",
                 mode.c_str());
    if (initialize && patch) sf_error("init and patch modes are mutually exclusive");

    /* header-copy source dataset */
    char* srcpath = sf_getstring("headers");
    if (NULL == srcpath) srcpath = sf_getstring("like");
    /* another MDIO dataset to copy text/binary/trace headers from */

    const char* hdrcopy = sf_getstring("hdrcopy");
    /* what to copy from headers=/like=: text, binary, trace, or all */
    if (NULL == hdrcopy) hdrcopy = "all";
    std::string hc = hdrcopy;
    bool cp_text   = (hc.find("text")   != std::string::npos) ||
                     (hc.find("all")    != std::string::npos);
    bool cp_binary = (hc.find("binary") != std::string::npos) ||
                     (hc.find("all")    != std::string::npos);
    bool cp_trace  = (hc.find("trace")  != std::string::npos) ||
                     (hc.find("all")    != std::string::npos);

    /* ---- RSF geometry ---- */
    off_t n[SF_MAX_DIM];
    int dim = sf_largefiledims(in, n);
    long ns = (long) n[0];
    long total = 1;
    for (int i = 0; i < dim; i++) total *= (long) n[i];
    long ntr = total / ns;

    /* MDIO axes (slowest-first, sample axis last) = reverse of RSF axes. */
    std::vector<MdioAxis> axes(dim);
    std::vector<std::string> used;
    for (int a = 0; a < dim; a++) {
        int ri = dim - a;             /* RSF axis number (1-based) */
        char key[8];
        MdioAxis ax;
        ax.size = (long) n[ri - 1];
        float fo = 0., fd = 1.;
        snprintf(key, sizeof(key), "o%d", ri);
        ax.o = sf_histfloat(in, key, &fo) ? (double) fo : 0.;
        snprintf(key, sizeof(key), "d%d", ri);
        ax.d = sf_histfloat(in, key, &fd) ? (double) fd : 1.;
        snprintf(key, sizeof(key), "label%d", ri);
        char* lab = sf_histstring(in, key);
        ax.label = clean_name(lab, ri, used);
        if (lab) free(lab);
        snprintf(key, sizeof(key), "unit%d", ri);
        char* un = sf_histstring(in, key);
        if (un) { ax.unit = un; free(un); }
        ax.sample = (ri == 1);
        axes[a] = ax;
    }

    /* In init mode the input supplies axis metadata, while fulln# (or N#)
       supplies the global extent.  Omitting it preserves the input extent. */
    if (initialize) {
        for (int ri = 1; ri <= dim; ri++) {
            char key[16];
            off_t full = n[ri - 1];
            snprintf(key, sizeof(key), "fulln%d", ri);
            bool supplied = sf_getlargeint(key, &full);
            if (!supplied) {
                snprintf(key, sizeof(key), "N%d", ri);
                supplied = sf_getlargeint(key, &full);
            }
            if (full < 1) sf_error("Full extent for axis %d must be positive", ri);
            axes[dim - ri].size = (long) full;
        }
    }

    double o1 = axes[dim - 1].o;  /* sample-axis origin -> delrt */

    /* ---- trace headers from tfile ---- */
    std::vector<std::string> keynames;
    std::vector<std::vector<int> > columns;

    char* tname = sf_getstring("tfile");
    /* input trace header file (from sfsegyread or sfsegyheader) */
    if (NULL != tname) {
        sf_file hin = sf_input("tfile");
        int nk;
        if (!sf_histint(hin, "n1", &nk)) sf_error("No n1= in tfile");
        long thtr = sf_leftsize(hin, 1);
        if (thtr != ntr)
            sf_warning("tfile has %ld traces, data has %ld", thtr, ntr);
        long use = (thtr < ntr) ? thtr : ntr;

        segy_init(nk, hin);
        int* all = sf_intalloc(nk * (int) use);
        /* read only the traces we will use */
        sf_intread(all, (size_t) nk * use, hin);

        for (int k = 0; k < nk && k < SF_NKEYS; k++) {
            const char* nm = segykeyword(k);
            bool nonzero = false;
            std::vector<int> col((size_t) ntr, 0);
            for (long t = 0; t < use; t++) {
                int v = all[(size_t) t * nk + k];
                col[(size_t) t] = v;
                if (v) nonzero = true;
            }
            if (nonzero) { keynames.push_back(nm); columns.push_back(col); }
        }
        free(all);
        sf_fileclose(hin);
    } else {
        segy_init(SF_NKEYS, NULL);
    }

    /* ---- open header-copy source (if any) ---- */
    mdio::Dataset* srcds = NULL;
    std::string srcHeadersVar;
    if (NULL != srcpath) {
        auto sf = mdio::Dataset::Open(std::string(srcpath), mdio::constants::kOpen);
        if (!sf.status().ok()) sf_error("Cannot open headers source \"%s\"", srcpath);
        srcds = new mdio::Dataset(sf.value());
        srcHeadersVar = mdio_headers_variable(*srcds, NULL);
    }

    /* If copying trace headers, replace tfile columns with source values. */
    if (NULL != srcds && cp_trace) {
        std::vector<mdio::RangeDescriptor<mdio::Index> > nosl;
        keynames.clear();
        columns.clear();
        for (int k = 0; k < SF_NKEYS; k++) {
            const char* nm = segykeyword(k);
            std::vector<double> vals;
            if (!mdio_read_header_key(*srcds, std::string(srcpath), nosl,
                                      srcHeadersVar, nm, vals)) continue;
            if ((long) vals.size() != ntr) continue;
            std::vector<int> col((size_t) ntr, 0);
            for (long t = 0; t < ntr; t++) col[(size_t) t] = (int) vals[(size_t) t];
            keynames.push_back(nm);
            columns.push_back(col);
        }
    }

    /* Force delrt to reflect the sample-axis origin (mirrors sfsegywrite). */
    {
        int dk = segykey("delrt");
        bool found = false;
        for (size_t i = 0; i < keynames.size(); i++)
            if (segykey(keynames[i].c_str()) == dk) {
                for (long t = 0; t < ntr; t++)
                    columns[i][(size_t) t] = (int)(1000. * o1);
                found = true;
                break;
            }
        if (!found && o1 != 0.) {
            std::vector<int> col((size_t) ntr, (int)(1000. * o1));
            keynames.push_back("delrt");
            columns.push_back(col);
        }
    }

    /* ---- patch an existing dataset without changing its metadata ---- */
    if (patch) {
        auto of = mdio::Dataset::Open(std::string(path), mdio::constants::kOpen);
        if (!of.status().ok())
            sf_error("Cannot open MDIO \"%s\" for patching: %s", path,
                     of.status().ToString().c_str());
        mdio::Dataset ds = of.value();

        auto dvr = ds.variables.get<mdio::dtypes::float32_t>(datavar);
        if (!dvr.status().ok())
            sf_error("No float32 data variable \"%s\" in \"%s\"",
                     datavar.c_str(), path);
        auto data = dvr.value();
        auto dom = data.dimensions();
        int dataset_dim = (int) dom.rank();
        if (dim > dataset_dim)
            sf_error("Input rank %d exceeds MDIO rank %d",
                     dim, dataset_dim);

        auto dlabels = dom.labels();
        auto dshape = dom.shape();
        std::vector<std::string> labels((size_t) dataset_dim);
        std::vector<long> first((size_t) dataset_dim),
                          count((size_t) dataset_dim);
        for (int a = 0; a < dataset_dim; a++) {
            int ri = dataset_dim - a;
            char key[8];
            off_t lv;
            long ff = 0;
            /* Madagascar filters commonly omit trailing singleton axes from
               each partition's output history.  Treat those absent slow axes
               as length one so they can still patch a higher-rank dataset. */
            long input_extent = (ri <= dim) ? (long) n[ri - 1] : 1L;
            long nn = input_extent;
            snprintf(key, sizeof(key), "f%d", ri);
            if (sf_getlargeint(key, &lv)) ff = (long) lv;
            snprintf(key, sizeof(key), "n%d", ri);
            if (sf_getlargeint(key, &lv)) nn = (long) lv;
            if (nn != input_extent)
                sf_error("Patch n%d=%ld must equal input axis length %ld",
                         ri, nn, input_extent);
            if (ff < 0 || nn < 1 || ff + nn > (long) dshape[a])
                sf_error("Patch axis %d region [%ld,%ld) exceeds MDIO extent %ld",
                         ri, ff, ff + nn, (long) dshape[a]);
            labels[(size_t) a] = std::string(dlabels[a]);
            first[(size_t) a] = ff;
            count[(size_t) a] = nn;
        }

        std::string why;
        if (!chunk_aligned(data, first, count, labels, why))
            sf_error("Unsafe data patch: %s; patches must own complete chunks",
                     why.c_str());

        std::vector<mdio::RangeDescriptor<mdio::Index> > slices;
        for (int a = 0; a < dataset_dim; a++) {
            mdio::RangeDescriptor<mdio::Index> r = {
                labels[(size_t) a].c_str(),
                (mdio::Index) first[(size_t) a],
                (mdio::Index)(first[(size_t) a] + count[(size_t) a]), 1};
            slices.push_back(r);
        }
        auto sr = ds.isel(slices);
        if (!sr.status().ok())
            sf_error("Failed to select patch region: %s",
                     sr.status().ToString().c_str());
        mdio::Dataset region = sr.value();

        float* buf = sf_floatalloc(total);
        sf_floatread(buf, total, in);
        if (!write_float_buf(region, datavar, buf, total))
            sf_error("Failed to patch data variable \"%s\"", datavar.c_str());
        free(buf);

        /* Optional tfile patches are supported for the separate int32 header
           variables produced by this writer.  Each header region is checked
           against that variable's own chunk grid before it is touched. */
        for (size_t i = 0; i < keynames.size(); i++) {
            auto hvr = ds.variables.get<mdio::dtypes::int32_t>(keynames[i]);
            if (!hvr.status().ok()) {
                if (verb) sf_warning("Skipping absent header variable \"%s\"",
                                     keynames[i].c_str());
                continue;
            }
            std::string hwhy;
            if (!chunk_aligned(hvr.value(), first, count, labels, hwhy))
                sf_error("Unsafe tfile patch for \"%s\": %s",
                         keynames[i].c_str(), hwhy.c_str());
            if (!write_int_var(region, keynames[i], columns[i]))
                sf_error("Failed to patch header variable \"%s\"",
                         keynames[i].c_str());
        }

        if (verb) sf_warning("Patched MDIO \"%s\"", path);
        exit(0);
    }

    /* ---- chunking strategy / sizes ---- */
    /* axes[] is in MDIO order (slowest first, sample axis last); RSF axis ri
       maps to MDIO axis a = dim - ri. */
    char* chunkstr = sf_getstring("chunk");
    /* chunk size or named strategy for the data variable: an integer applied to
       every axis (0 or >= axis size means a single full-length chunk), or one of
       auto (default, min(size,128) per axis), full (whole array in one chunk), or
       trace (chunk only the sample axis, full-length trace axes).  Per-axis
       overrides chunk1=,chunk2=,... (RSF axis order; axis 1 = samples) take
       precedence. */
    std::string strat = (NULL != chunkstr) ? chunkstr : "auto";

    bool strat_named = (strat == "auto" || strat == "full" || strat == "trace");
    long strat_int = 0;
    if (!strat_named) strat_int = atol(strat.c_str());

    std::vector<long> chunk(dim);
    for (int a = 0; a < dim; a++) {
        long size = axes[a].size;
        long c;
        if (strat == "full") {
            c = size;
        } else if (strat == "trace") {
            c = axes[a].sample ? ((size < 128) ? size : 128) : size;
        } else if (!strat_named) {        /* integer default for all axes */
            c = (strat_int <= 0 || strat_int >= size) ? size : strat_int;
        } else {                          /* "auto" */
            c = (size < 128) ? size : 128;
        }
        chunk[a] = c;
    }

    /* per-axis overrides chunk1=, chunk2=, ... (RSF axis order) */
    for (int ri = 1; ri <= dim; ri++) {
        char key[16];
        int v;
        snprintf(key, sizeof(key), "chunk%d", ri);
        if (sf_getint(key, &v)) {
            int a = dim - ri;
            long size = axes[a].size;
            chunk[a] = (v <= 0 || v >= size) ? size : v;
        }
    }

    /* ---- build schema and create dataset ---- */

    char* dname = sf_getstring("name");
    std::string dsname = (NULL != dname) ? dname : "Madagascar MDIO";

    nlohmann::json schema =
        mdio_build_schema(dsname, axes, datavar, keynames, chunk);

    /* text / binary headers into the dataset metadata */
    char ahead[SF_EBCBYTES];
    bool have_text = false;
    if (NULL != srcds && cp_text && mdio_get_text_header(*srcds, ahead)) {
        have_text = true;
    } else {
        char* hname = sf_getstring("hfile");
        /* input SEG-Y EBCDIC text header file */
        if (NULL != hname) {
            FILE* fp = fopen(hname, "r");
            if (NULL != fp) {
                memset(ahead, ' ', SF_EBCBYTES);
                fread(ahead, 1, SF_EBCBYTES, fp);
                fclose(fp);
                have_text = true;
            }
        }
    }
    if (have_text) mdio_put_text_header(schema, ahead);

    char bhead[SF_BNYBYTES];
    bool have_bin = false;
    if (NULL != srcds && cp_binary && mdio_get_binary_header(*srcds, bhead)) {
        have_bin = true;
    } else {
        char* bname = sf_getstring("bfile");
        /* input SEG-Y binary header file */
        if (NULL != bname) {
            FILE* fp = fopen(bname, "rb");
            if (NULL != fp) {
                memset(bhead, 0, SF_BNYBYTES);
                fread(bhead, 1, SF_BNYBYTES, fp);
                fclose(fp);
                have_bin = true;
            }
        }
    }
    if (have_bin) mdio_put_binary_header(schema, bhead);

    if (verb)
        sf_warning("Creating MDIO \"%s\"%s",
                   path, initialize ? " (empty initialization)" : "");

    auto dsFut = mdio::Dataset::from_json(schema, std::string(path),
                                          mdio::constants::kCreate);
    if (!dsFut.status().ok())
        sf_error("Failed to create MDIO \"%s\": %s", path,
                 dsFut.status().ToString().c_str());
    mdio::Dataset ds = dsFut.value();

    /* ---- write dimension coordinates (uniform o + i*d) ---- */
    for (int a = 0; a < dim; a++) {
        auto vr = ds.variables.get<mdio::dtypes::float32_t>(axes[a].label);
        if (!vr.status().ok()) continue;
        auto var = vr.value();
        auto vdr = mdio::from_variable<mdio::dtypes::float32_t>(var);
        if (!vdr.status().ok()) continue;
        auto vd = vdr.value();
        long nc = (long) vd.num_samples();
        auto off = vd.get_flattened_offset();
        float* p = static_cast<float*>(vd.get_data_accessor().data());
        for (long i = 0; i < nc; i++) p[off + i] = (float)(axes[a].o + i * axes[a].d);
        var.Write(vd);
    }

    /* ---- write data (RSF order matches MDIO row-major order) ---- */
    if (!initialize) {
        float* buf = sf_floatalloc(total);
        sf_floatread(buf, total, in);
        if (!write_float_buf(ds, datavar, buf, total))
            sf_error("Failed to write data variable \"%s\"", datavar.c_str());
        free(buf);
    } else {
        /* An RSF input may be backed by a data-server pipe.  Init intentionally
           consumes no samples, so close it explicitly instead of leaving the
           producer waiting for a reader at process shutdown. */
        sf_fileclose(in);
    }

    /* ---- write trace-header variables ---- */
    if (!initialize) {
        for (size_t i = 0; i < keynames.size(); i++) {
            if (!write_int_var(ds, keynames[i], columns[i]) && verb)
                sf_warning("Could not write header variable \"%s\"",
                           keynames[i].c_str());
        }
    }

    if (verb) sf_warning("Done");

    exit(0);
}
