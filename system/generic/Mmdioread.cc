/* Convert an MDIO dataset to RSF.

Reads the MDIO (Zarr) format through the mdio-cpp library and writes the
amplitudes to an RSF file, the per-trace SEG-Y headers to a separate tfile
(compatible with sfsegyread/sfsegywrite), and optionally the SEG-Y EBCDIC text
header (hfile) and 400-byte binary header (bfile) when they are present in the
dataset metadata.

The input may be a local path, or a gs:// or s3:// URL (when mdio-cpp was built
with the corresponding cloud drivers).

The data can be windowed on read with the same parameters as sfwindow
(n#, f#, j#, min#, max#).  Axis 1 is the fast (sample) axis; the remaining
axes index the trace grid, in the order RSF stores them (n2 is the fastest
trace dimension).  Because mdio-cpp slices with unit stride, a contiguous
bounding box is read lazily and any j#>1 decimation is applied afterwards.

Trace headers are read either from one MDIO variable per SEG-Y key, or from a
single structured "headers" variable.
*/

#include <stdio.h>
#include <stdlib.h>

#include <string>
#include <vector>

#include "mdio2segy.hh"

/* Decode a row-major (last axis fastest) linear index into a multi-index. */
static void decode(long lin, int ndim, const long* shape, long* idx)
{
    for (int a = ndim - 1; a >= 0; a--) {
        idx[a] = lin % shape[a];
        lin /= shape[a];
    }
}

int main(int argc, char* argv[])
{
    sf_init(argc, argv);

    bool verb;
    if (!sf_getbool("verb", &verb)) verb = false;
    /* Verbosity flag */

    char* path = sf_getstring("mdio");
    /* input MDIO dataset (path or gs://, s3:// URL) */
    if (NULL == path) sf_error("Need mdio=");

    char* dataname = sf_getstring("data");
    /* name of the MDIO data variable (default: auto-detect) */
    char* hdrsname = sf_getstring("headers");
    /* name of the MDIO trace-headers variable (default: auto-detect) */

    const char* read = sf_getstring("read");
    /* what to read: h - header, d - data, b - both (default) */
    if (NULL == read) read = "b";

    /* ---- open dataset (lazy) ---- */
    auto dsFut = mdio::Dataset::Open(std::string(path), mdio::constants::kOpen);
    if (!dsFut.status().ok()) sf_error("Cannot open MDIO \"%s\"", path);
    mdio::Dataset ds = dsFut.value();

    std::string datavar = mdio_data_variable(ds, dataname);
    if (datavar.empty()) sf_error("Could not find a data variable in \"%s\"", path);
    std::string headersVar = mdio_headers_variable(ds, hdrsname);

    if (verb) sf_warning("data variable: %s", datavar.c_str());

    std::vector<MdioAxis> axes = mdio_axes(ds, datavar);
    int rank = (int) axes.size();
    if (rank < 1) sf_error("Data variable \"%s\" has no dimensions", datavar.c_str());

    /* ---- resolve windowing per axis (window.c convention) ---- */
    std::vector<long> f(rank), m(rank), jp(rank), mc(rank);
    for (int i = 1; i <= rank; i++) {     /* RSF axis i -> MDIO axis a */
        int a = rank - i;
        long n = axes[a].size;
        double o = axes[a].o, d0 = axes[a].d;
        char key[8];
        int    iv;
        float  av;
        off_t  lv;

        /* jump j# (or sampling d#) */
        long j = 1;
        snprintf(key, sizeof(key), "j%d", i);
        if (sf_getint(key, &iv)) {
            j = iv;
        } else {
            snprintf(key, sizeof(key), "d%d", i);
            if (sf_getfloat(key, &av) && d0 != 0.) j = (long)(0.5 + av / d0);
        }
        if (j < 1) j = 1;

        /* start f# (or minimum min#) */
        long ff = 0;
        snprintf(key, sizeof(key), "f%d", i);
        if (sf_getlargeint(key, &lv)) {
            ff = lv;
        } else {
            snprintf(key, sizeof(key), "min%d", i);
            if (sf_getfloat(key, &av) && d0 != 0.) ff = (long)(0.5 + (av - o) / d0);
        }
        if (ff < 0) ff = n + ff;
        if (ff < 0) sf_error("Negative f%d", i);

        double onew = o + ff * d0;
        double dnew = d0 * j;

        /* count n# (or maximum max#) */
        long mm;
        snprintf(key, sizeof(key), "n%d", i);
        if (sf_getlargeint(key, &lv)) {
            mm = lv;
        } else {
            snprintf(key, sizeof(key), "max%d", i);
            if (sf_getfloat(key, &av) && dnew != 0.)
                mm = (long)(1.5 + (av - onew) / dnew);
            else
                mm = 1 + (n - 1 - ff) / j;
        }
        if (mm < 1) mm = 1;
        if (ff + (mm - 1) * j > n - 1)
            sf_error("n%d=%ld is too big (axis %s, size %ld)",
                     i, (long) mm, axes[a].label.c_str(), n);

        f[a]  = ff;
        m[a]  = mm;
        jp[a] = j;
        mc[a] = (mm - 1) * j + 1;

        axes[a].o = onew;  /* carry windowed geometry to the output */
        axes[a].d = dnew;
    }

    /* ---- contiguous slice over the bounding box ---- */
    std::vector<mdio::RangeDescriptor<mdio::Index> > slices;
    for (int a = 0; a < rank; a++) {
        mdio::RangeDescriptor<mdio::Index> r = {axes[a].label.c_str(),
                                                (mdio::Index) f[a],
                                                (mdio::Index)(f[a] + mc[a]),
                                                1};
        slices.push_back(r);
    }

    mdio::Dataset sliced = ds;
    {
        auto r = ds.isel(slices);
        if (!r.status().ok()) sf_error("Failed to slice MDIO dataset");
        sliced = r.value();
    }

    /* sizes */
    int sa = rank - 1;             /* MDIO sample axis */
    long ns = m[sa];
    long ntr = 1;
    for (int a = 0; a < rank; a++) if (a != sa) ntr *= m[a];
    long total = ns * ntr;

    if (verb) sf_warning("Reading %ld samples x %ld traces", ns, ntr);

    /* ---- outputs ---- */
    sf_file out = NULL, hdr = NULL;

    if (read[0] != 'h') { /* data (and possibly headers) */
        out = sf_output("out");
        sf_settype(out, SF_FLOAT);
        /* RSF axis i corresponds to MDIO axis rank-i */
        for (int i = 1; i <= rank; i++) {
            int a = rank - i;
            char key[8];
            snprintf(key, sizeof(key), "n%d", i);
            sf_putint(out, key, (int) m[a]);
            snprintf(key, sizeof(key), "d%d", i);
            sf_putfloat(out, key, (float) axes[a].d);
            snprintf(key, sizeof(key), "o%d", i);
            sf_putfloat(out, key, (float) axes[a].o);
            snprintf(key, sizeof(key), "label%d", i);
            sf_putstring(out, key,
                         axes[a].label.empty() ?
                         (i == 1 ? "Time" : "Trace") : axes[a].label.c_str());
            if (!axes[a].unit.empty()) {
                snprintf(key, sizeof(key), "unit%d", i);
                sf_putstring(out, key, axes[a].unit.c_str());
            }
        }
    }

    if (read[0] != 'd') { /* headers */
        hdr = sf_output("tfile");
        sf_putint(hdr, "n1", SF_NKEYS);
        sf_putint(hdr, "n2", (int) ntr);
        sf_settype(hdr, SF_INT);
        segy_init(SF_NKEYS, NULL);
        segy2hist(hdr, SF_NKEYS);

        char* tname = sf_getstring("tfile");
        /* output trace header file */
        if (NULL != out) sf_putstring(out, "head", (NULL != tname) ? tname : "tfile");
    } else {
        segy_init(SF_NKEYS, NULL);
    }

    if (NULL != out) sf_fileflush(out, NULL);

    /* ---- text / binary SEG-Y headers ---- */
    char* hname = sf_getstring("hfile");
    /* output SEG-Y EBCDIC text header file */
    if (NULL != hname) {
        char ahead[SF_EBCBYTES];
        if (mdio_get_text_header(ds, ahead)) {
            FILE* fp = fopen(hname, "w");
            if (NULL == fp) sf_error("Cannot open hfile \"%s\"", hname);
            fwrite(ahead, 1, SF_EBCBYTES, fp);
            fclose(fp);
            if (verb) sf_warning("Text header written to \"%s\"", hname);
        } else if (verb) {
            sf_warning("No text header in dataset; hfile not written");
        }
    }

    char* bname = sf_getstring("bfile");
    /* output SEG-Y binary header file */
    if (NULL != bname) {
        char bhead[SF_BNYBYTES];
        if (mdio_get_binary_header(ds, bhead)) {
            FILE* fp = fopen(bname, "wb");
            if (NULL == fp) sf_error("Cannot open bfile \"%s\"", bname);
            fwrite(bhead, 1, SF_BNYBYTES, fp);
            fclose(fp);
            if (verb) sf_warning("Binary header written to \"%s\"", bname);
        } else if (verb) {
            sf_warning("No binary header in dataset; bfile not written");
        }
    }

    /* ---- data ---- */
    if (NULL != out) {
        std::vector<double> draw;
        if (!mdio_read_field(sliced, datavar, draw))
            sf_error("Failed to read data variable \"%s\"", datavar.c_str());

        float* obuf = sf_floatalloc(total);
        std::vector<long> idx(rank);
        for (long t = 0; t < total; t++) {
            decode(t, rank, &m[0], &idx[0]);
            long s = 0;
            for (int a = 0; a < rank; a++) s = s * mc[a] + idx[a] * jp[a];
            obuf[t] = (s < (long) draw.size()) ? (float) draw[s] : 0.0f;
        }
        sf_floatwrite(obuf, total, out);
        free(obuf);
    }

    /* ---- headers / tfile ---- */
    if (NULL != hdr) {
        /* trace-grid geometry (MDIO axes except the sample axis) */
        int tr = rank - 1;
        std::vector<long> mt(tr), mct(tr), jt(tr);
        for (int a = 0; a < tr; a++) { mt[a] = m[a]; mct[a] = mc[a]; jt[a] = jp[a]; }

        std::vector<std::vector<int> > keyvals(SF_NKEYS);
        for (int k = 0; k < SF_NKEYS; k++) {
            const char* nm = segykeyword(k);
            std::vector<double> vals;
            if (!mdio_read_header_key(sliced, std::string(path), slices,
                                      headersVar, nm, vals))
                continue;

            std::vector<int> dec((size_t) ntr, 0);
            std::vector<long> idx(tr > 0 ? tr : 1);
            for (long t = 0; t < ntr; t++) {
                long s = 0;
                if (tr > 0) {
                    decode(t, tr, &mt[0], &idx[0]);
                    for (int a = 0; a < tr; a++) s = s * mct[a] + idx[a] * jt[a];
                }
                dec[(size_t) t] = (s < (long) vals.size()) ? (int) vals[s] : 0;
            }
            keyvals[k] = dec;
        }

        int* itrace = sf_intalloc(SF_NKEYS);
        for (long t = 0; t < ntr; t++) {
            for (int k = 0; k < SF_NKEYS; k++)
                itrace[k] = keyvals[k].empty() ? 0 : keyvals[k][(size_t) t];
            sf_intwrite(itrace, SF_NKEYS, hdr);
        }
        free(itrace);
    }

    exit(0);
}
