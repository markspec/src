/* Shared helpers translating between the MDIO (mdio-cpp) data model and
   the Madagascar SEG-Y / RSF representation used by sfmdioread and
   sfmdiowrite. */

#ifndef _mdio2segy_hh
#define _mdio2segy_hh

#include <string>
#include <vector>

#include <mdio/mdio.h>
#include <nlohmann/json.hpp>

extern "C" {
#include <rsf.h>
#include <rsfsegy.h>
}

/* Geometry of one MDIO dimension, kept in the data variable's native order
   (slowest first, the sample axis last). */
struct MdioAxis {
    std::string label;
    long        size;       /* full extent in the file                       */
    double      o;          /* origin (first dimension-coordinate value)     */
    double      d;          /* sampling (coordinate spacing)                 */
    std::string unit;
    bool        sample;     /* true for the fastest axis -> RSF n1           */
};

/* Resolve the principal (sample-bearing) data variable.  When "given" is
   non-empty it is returned verbatim; otherwise the first floating-point,
   non-coordinate variable is chosen.  Returns "" when none is found. */
std::string mdio_data_variable(mdio::Dataset& ds, const char* given);

/* Resolve the per-trace headers variable.  When "given" is non-empty it is
   returned verbatim; otherwise a variable named headers/trace_headers, or any
   structured variable (one carrying an unlabeled "byte" dimension), is used.
   Returns "" when none is found. */
std::string mdio_headers_variable(mdio::Dataset& ds, const char* given);

/* Read the values of SEG-Y key "key" over the (already sliced) trace grid.
   Supports two MDIO header layouts:
     1. one int/float variable per SEG-Y field (read directly from "ds"); and
     2. a single structured headers variable, whose field is extracted with
        Dataset::SelectField after re-opening and re-slicing "path"/"slices".
   Returns false when no matching field is present. */
bool mdio_read_header_key(mdio::Dataset& ds, const std::string& path,
                          const std::vector<mdio::RangeDescriptor<mdio::Index> >& slices,
                          const std::string& headersVar, const char* key,
                          std::vector<double>& out);

/* Geometry of every dimension of the data variable, in MDIO order. */
std::vector<MdioAxis> mdio_axes(mdio::Dataset& ds, const std::string& datavar);

/* Read a full variable (coordinate or trace-header field) into a double
   buffer, trying the common integer/float MDIO data types.  The flattened
   slice offset is applied, so the returned values are in logical order.
   Returns false when the variable is absent or unreadable. */
bool mdio_read_field(mdio::Dataset& ds, const std::string& name,
                     std::vector<double>& out);

/* Origin/sampling of one dimension, derived from its coordinate variable.
   Returns false when no usable coordinate variable exists. */
bool mdio_coord(mdio::Dataset& ds, const std::string& label,
                double* o, double* d);

/* SEG-Y text (3200-byte EBCDIC) and binary (400-byte) headers stored in the
   dataset-level metadata attributes.  Each returns false when absent. */
bool mdio_get_text_header(mdio::Dataset& ds, char ahead[SF_EBCBYTES]);
bool mdio_get_binary_header(mdio::Dataset& ds, char bhead[SF_BNYBYTES]);

/* Embed SEG-Y text/binary headers into a schema metadata object so they are
   persisted with a newly created MDIO dataset. */
void mdio_put_text_header(nlohmann::json& metadata,
                          const char ahead[SF_EBCBYTES]);
void mdio_put_binary_header(nlohmann::json& metadata,
                            const char bhead[SF_BNYBYTES]);

/* Resolve the MDIO variable name that stores values for SEG-Y key "key",
   honoring a small alias table for common MDIO field names.  Returns "" when
   the dataset exposes no matching trace-header variable. */
std::string mdio_header_field(mdio::Dataset& ds, const char* key);

/* Build an MDIO v1 schema (nlohmann::json) for a new dataset:
   - axes: dimension geometry in MDIO order (sample axis last);
   - datavar: name/dtype of the float data variable;
   - keys: SEG-Y key names to materialize as int32 trace-header variables
     (those sharing the non-sample trace dimensions);
   - chunk: chunk shape (same rank/order as axes);
   - name: dataset name. */
nlohmann::json mdio_build_schema(const std::string& name,
                                 const std::vector<MdioAxis>& axes,
                                 const std::string& datavar,
                                 const std::vector<std::string>& keys,
                                 const std::vector<long>& chunk);

#endif
