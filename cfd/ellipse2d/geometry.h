/*  ellipse2d -- geometry + boundary scaffold (OpenLB v1.9.0, 2D CPU)
 *
 *  Flow around an elliptical (arbitrary closed-polygon) cylinder in an
 *  open free-stream channel. This header provides the geometry-owning
 *  portions of a Case<>-style case.h:
 *
 *      - parameter declarations (namespace olb::parameters)
 *      - the polygon body loader + signed-distance builder
 *      - createMesh()
 *      - prepareGeometry()
 *
 *  Modelled on examples/laminar/cylinder2d/case.h but with an OPEN domain
 *  (free-stream top/bottom, velocity inlet, pressure outlet) instead of the
 *  cylinder2d no-slip Poiseuille channel.
 *
 *  INTEGRATION
 *  -----------
 *  This is a self-contained snippet header. To use it in a full case.h you
 *  have two options:
 *    (a) Keep it as-is and include it from your case.h *after* you have
 *        defined `using MyCase = Case<...>;`. If you do this, DELETE the
 *        `using MyCase = ...` block below and the duplicate
 *        `using namespace` lines, since they will already be in scope.
 *    (b) Paste the parameter structs, the three free functions
 *        (loadBodyPolygon / createMesh / prepareGeometry) and the global
 *        `bodyPointFilePath` directly into your case.h.
 *
 *  prepareLattice / getResults / simulate are NOT in this file -- those are
 *  owned by the numerics side. The material numbers defined here are the
 *  contract they consume:
 *
 *      MATERIAL SCHEME
 *      ---------------
 *      0  unused / outside domain
 *      1  bulk fluid
 *      2  top AND bottom strips  -> FREE-STREAM (slip / free-slip / equilibrium)
 *      3  left strip             -> velocity INLET
 *      4  right strip            -> pressure OUTLET
 *      5  body (the polygon)     -> wall (Bouzidi -- needs true signed distance)
 *      6  sponge zone            -> SpongeLayerDynamics absorbing layer just
 *                                   UPSTREAM of the outlet (graded density+velocity
 *                                   nudge toward the freestream, ramped 0->sigma_max
 *                                   over the last SPONGE_CHORDS chords) to dissipate
 *                                   acoustic waves + vortices before they reflect off
 *                                   the fixed-pressure outlet (material 4, KEPT).
 */
#pragma once

#include <olb.h>

#include <cmath>
#include <fstream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace olb;
using namespace olb::descriptors;
using namespace olb::graphics;
using namespace olb::names;

#define BOUZIDI

// === Step 1: Declarations ===
// (Delete this block if MyCase is already defined in your including case.h.)
//
// SPONGE FIELDS. SpongeLayerDynamics (src/dynamics/spongeLayerDynamics.h) reads
// three per-cell fields in its collide(): descriptors::DENSITY (reference rho),
// descriptors::VELOCITY (reference u, a d-vector) and descriptors::DAMPING (the
// per-cell relaxation strength sigma). These must be carried by the descriptor
// so the GPU (CUDA) build provisions them at COMPILE time -- a bare D2Q9<> would
// only auto-provision on CPU and silently lack them on device. They add a few
// scalars of storage per cell and are inert for the BGK / boundary materials
// that never touch them, so adding them globally is safe and does not change the
// validated physics on materials 1-5.
using MyCase = Case<
  NavierStokes,
  Lattice<FLOATING_POINT_TYPE,
          descriptors::D2Q9<descriptors::DENSITY,
                            descriptors::VELOCITY,
                            descriptors::DAMPING>>
>;

// ---------------------------------------------------------------------------
// String body-point file path.
//
//   The polygon-file path is a std::string and therefore CANNOT live in a
//   ParametersD FIELD_BASE (those store fixed-width floating point / int
//   Vectors only). It is instead carried in this translation-unit-global
//   string, which main() must populate before calling createMesh /
//   prepareGeometry, e.g.:
//
//       // from CLI:   ./ellipse2d shapes/ellipse_0001.dat
//       if (argc > 1) { bodyPointFilePath = argv[1]; }
//       // or from XML: bodyPointFilePath = config["Application"]["BodyFile"].get<std::string>();
//
//   Everything numeric (RESOLUTION, REYNOLDS, CHANNEL_HEIGHT, SEMI_MINOR_B,
//   UPSTREAM_CHORDS, DOWNSTREAM_CHORDS, ...) goes through ParametersD as
//   FIELD_BASE values, exactly as cylinder2d does.
// ---------------------------------------------------------------------------
inline std::string bodyPointFilePath = "";

namespace olb::parameters {

// Channel (domain) height in physical units. Passed in so the blockage
// ratio B/H matches the training dataset rather than being derived.
struct CHANNEL_HEIGHT     : public descriptors::FIELD_BASE<1> { };

// Ellipse semi-minor axis, physical units. Logging / diagnostics only;
// the actual body shape comes from the polygon file.
struct SEMI_MINOR_B       : public descriptors::FIELD_BASE<1> { };

// Inlet placement: how many chord lengths upstream of the body leading
// edge the inlet plane sits (spec: ~5-8, recommend 6).
struct UPSTREAM_CHORDS    : public descriptors::FIELD_BASE<1> { };

// Outlet placement: how many chord lengths downstream of the body trailing
// edge the outlet plane sits (spec: ~15-20).
struct DOWNSTREAM_CHORDS  : public descriptors::FIELD_BASE<1> { };

// Reference body length (chord = x-extent of the polygon). Filled in by
// loadBodyPolygon so prepareLattice can read charPhysLength from here.
struct CHORD_LENGTH       : public descriptors::FIELD_BASE<1> { };

// Body centroid (x,y), filled in by loadBodyPolygon for reuse downstream
// (e.g. force-integration probe points in getResults).
struct BODY_CENTROID      : public descriptors::FIELD_BASE<0, 1> { };

// Streamwise extent of the absorbing SPONGE zone (material 6), in chord
// lengths, measured from the inner edge of the outlet strip backwards into the
// bulk. The sponge sits in the last SPONGE_CHORDS chords of the domain just
// UPSTREAM of the pressure outlet. Default ~4 chords (set in main / CLI). Must
// be < DOWNSTREAM_CHORDS so it stays downstream of the body and leaves a clear
// near-wake. Carved from bulk material 1 before nothing else (the inlet/outlet/
// body do not overlap it).
struct SPONGE_CHORDS      : public descriptors::FIELD_BASE<1> { };

// Peak damping strength sigma (LATTICE units, dimensionless) reached at the
// outlet end of the sponge. The DAMPING field ramps 0 -> SPONGE_SIGMA_MAX
// quadratically across the zone. ~0.5-1.0 is a typical absorbing strength for
// the Xu & Sagaut (2013) layer; too large reflects, too small under-damps.
struct SPONGE_SIGMA_MAX   : public descriptors::FIELD_BASE<1> { };

}

// ---------------------------------------------------------------------------
// Loaded-polygon container.
//
//   Holds the ordered closed boundary points plus derived geometry. A single
//   instance is built once in prepareGeometry and the points vector is
//   captured BY VALUE into the SDF lambda so the indicator owns its own copy.
// ---------------------------------------------------------------------------
template <typename T>
struct BodyPolygon {
  std::vector<Vector<T,2>> points;  // ordered, closed boundary (no repeat of first pt required)
  T minX = T{};
  T maxX = T{};
  T minY = T{};
  T maxY = T{};
  Vector<T,2> centroid{};

  T chord() const { return maxX - minX; }  // charL convention
};

// Reads "x y" per line from `path`. Blank lines and lines starting with '#'
// are ignored. Computes bounds + centroid. Throws on failure / degenerate
// input so the run aborts loudly rather than meshing garbage.
template <typename T>
BodyPolygon<T> loadBodyPolygon(const std::string& path)
{
  OstreamManager clout(std::cout, "loadBodyPolygon");

  if (path.empty()) {
    throw std::runtime_error(
      "Body point-file path is empty. Set the global `bodyPointFilePath` "
      "from argv/XML before calling prepareGeometry.");
  }

  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("Could not open body point file: " + path);
  }

  BodyPolygon<T> body;
  body.minX = body.minY =  std::numeric_limits<T>::max();
  body.maxX = body.maxY = -std::numeric_limits<T>::max();

  Vector<T,2> sum{T{}, T{}};
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') {
      continue;
    }
    std::istringstream ss(line);
    T x, y;
    if (!(ss >> x >> y)) {
      continue;  // skip malformed line
    }
    Vector<T,2> p{x, y};
    body.points.push_back(p);
    body.minX = std::min(body.minX, x);
    body.maxX = std::max(body.maxX, x);
    body.minY = std::min(body.minY, y);
    body.maxY = std::max(body.maxY, y);
    sum += p;
  }

  if (body.points.size() < 3) {
    throw std::runtime_error(
      "Body polygon needs >= 3 points, got "
      + std::to_string(body.points.size()) + " from " + path);
  }

  body.centroid = sum / T(body.points.size());

  clout << "loaded " << body.points.size() << " points from " << path << std::endl;
  clout << "  chord (x-extent) = " << body.chord()
        << " ; y-extent = " << (body.maxY - body.minY) << std::endl;
  clout << "  bounds x[" << body.minX << ", " << body.maxX
        << "] y[" << body.minY << ", " << body.maxY << "]" << std::endl;
  clout << "  centroid = (" << body.centroid[0] << ", " << body.centroid[1] << ")" << std::endl;

  return body;
}

// Builds an IndicatorF2D wrapping a hand-written polygon signed-distance
// field. Convention: NEGATIVE inside, POSITIVE outside (true signed distance),
// which is what setBouzidiBoundary expects.
//
//   sign      : winding-number point-in-polygon test (robust for non-convex)
//   magnitude : min over edges of pure segment distance via sdf::line(.,.,.,0)
//   value     : sign * magnitude
//
// The polygon points are captured BY VALUE so the indicator is self-contained.
// `offset` is added to every point so the body can be placed in the domain
// without a separate translate indicator (no IndicatorTranslate2D exists in
// v1.9.0 -- only a 3D one).
template <typename T>
std::shared_ptr<IndicatorF2D<T>> makePolygonIndicator(const BodyPolygon<T>& body,
                                                      Vector<T,2> offset = {T{0}, T{0}})
{
  std::vector<Vector<T,2>> poly = body.points;  // copy, captured by value
  for (auto& q : poly) {
    q += offset;
  }

  std::function<T(Vector<T,2>)> sdf = [poly](Vector<T,2> p) -> T {
    const std::size_t n = poly.size();

    // --- magnitude: nearest segment distance (r = 0 -> pure segment dist) ---
    T minDist = std::numeric_limits<T>::max();
    // --- sign: winding number (crossing count) -> inside if odd ---
    int wn = 0;

    for (std::size_t i = 0, j = n - 1; i < n; j = i++) {
      const Vector<T,2>& a = poly[j];
      const Vector<T,2>& b = poly[i];

      const T d = sdf::line<T,2>(p, a, b, T{0});
      if (d < minDist) {
        minDist = d;
      }

      // ray-cast crossing test along +x for the edge (a -> b)
      const bool aboveA = (a[1] > p[1]);
      const bool aboveB = (b[1] > p[1]);
      if (aboveA != aboveB) {
        // x-coordinate of the edge at height p[1]
        const T t  = (p[1] - a[1]) / (b[1] - a[1]);
        const T xc = a[0] + t * (b[0] - a[0]);
        if (p[0] < xc) {
          ++wn;
        }
      }
    }

    const T sign = (wn % 2 == 1) ? T{-1} : T{1};  // odd crossings -> inside
    return sign * minDist;
  };

  auto ind = std::make_shared<IndicatorSDF2D<T>>(sdf);

  // IndicatorSDF2D does NOT derive a bounding box from the SDF: its constructor
  // takes only the lambda and leaves the base IndicatorF2D _myMin/_myMax at the
  // default (0,0). There is no ctor arg or setter for the AABB in v1.9.0, so the
  // only mechanism is to assign through the mutable lvalue-reference accessors
  // IndicatorF2D::getMin()/getMax() (return Vector<T,2>&; see
  // src/functors/analytical/indicator/indicatorBaseF2D.hh).
  //
  // This AABB matters because SuperGeometry::rename(from,to,indicator) ->
  // BlockGeometry::rename(...) only scans voxels between condition.getMin() and
  // getMax() (blockGeometry.hh:411-435). With the default (0,0)/(0,0) the placed
  // body region is never scanned and body count stays 0.
  //
  // PAD SIZING. The same AABB is consumed by TWO mechanisms, and BOTH break if
  // it is too tight:
  //
  //   (1) rename(1,5,indicator): maps physical min/max to lattice nodes via
  //       Cuboid::getLatticeR = floor((physR-origin)/delta + 0.5) -- round to the
  //       NEAREST node (cuboid.h:94). A tight AABB rounds the outermost body
  //       cells out of the scan window on whichever side the corner falls past a
  //       node centre, leaving fewer/zero material-5 cells.
  //
  //   (2) setBouzidiBoundary distance(): for every material-5 cell it ray-marches
  //       from each FLUID neighbour cell toward the solid (indicatorBase.h:36-92)
  //       and terminates the march on isInsideBox(pos) == point >= _myMin &&
  //       point <= _myMax (indicatorBaseF2D.hh:195). The march START points are
  //       the fluid cells ~1 cell OUTSIDE the body surface (up to ~sqrt(2) cells
  //       out along D2Q9 diagonals). If those start points or the marched path
  //       fall outside the AABB, distance() returns false and the link silently
  //       degrades to half-way bounce-back (qIpop = 0.5) -- you lose the curved
  //       Bouzidi wall with NO error unless verbose.
  //
  // So the AABB must enclose the body PLUS at least ~2 cell widths of margin on
  // every side. makePolygonIndicator does not know physDeltaX, so we pad by a
  // fraction of the body extent: 0.1*extent is ~3 cells at RESOLUTION=30 and ~4
  // at RESOLUTION=40 -- enough to enclose the diagonal Bouzidi start points
  // across the resolutions used here. An over-large pad is harmless: rename's
  // surplus cells are rejected by the SDF (and the scan is clamped to the block),
  // and the ray-march terminates the instant it crosses the surface. Do NOT
  // shrink this below ~2 cell widths or Bouzidi will quietly fall back to
  // bounce-back on the body. If RESOLUTION is ever pushed far higher and you
  // want a tighter box, switch this to an absolute pad of a few * physDeltaX
  // (which would require passing physDeltaX into makePolygonIndicator).
  T loX =  std::numeric_limits<T>::max();
  T loY =  std::numeric_limits<T>::max();
  T hiX = -std::numeric_limits<T>::max();
  T hiY = -std::numeric_limits<T>::max();
  for (const auto& q : poly) {
    loX = std::min(loX, q[0]); hiX = std::max(hiX, q[0]);
    loY = std::min(loY, q[1]); hiY = std::max(hiY, q[1]);
  }
  const T pad = T{0.1} * std::max(hiX - loX, hiY - loY);
  ind->getMin() = Vector<T,2>{loX - pad, loY - pad};
  ind->getMax() = Vector<T,2>{hiX + pad, hiY + pad};

  return ind;
}

// ===========================================================================
// createMesh
//
//   Domain layout (origin at (0,0), body shifted so its leading edge is
//   `upstreamChords` chords from the inlet):
//
//        x = 0                                                  x = Lx
//        |<-- upstream -->|<-- body -->|<------ downstream ------>|
//        +----------------------------------------------------------+  y = H
//        |                                                          |
//        |                  ( body centred at y = H/2 )             |
//        |                                                          |
//        +----------------------------------------------------------+  y = 0
//
//   physDeltaX is derived from the chord and RESOLUTION:
//       physDeltaX = chord / RESOLUTION
//   (RESOLUTION = voxels per characteristic length = voxels per chord).
//   This MUST be confirmed with openlb-physics: cylinder2d uses
//   2*radius/RESOLUTION as charL; here charL = chord, so RESOLUTION counts
//   cells across the chord, NOT across the semi-minor axis.
// ===========================================================================
Mesh<MyCase::value_t,MyCase::d> createMesh(MyCase::ParametersD& parameters)
{
  using T = MyCase::value_t;

  // Load the body once to size the domain around it.
  const BodyPolygon<T> body = loadBodyPolygon<T>(bodyPointFilePath);
  const T chord = body.chord();

  const T resolution      = parameters.get<parameters::RESOLUTION>();
  const T upstreamChords  = parameters.get<parameters::UPSTREAM_CHORDS>();
  const T downstreamChords= parameters.get<parameters::DOWNSTREAM_CHORDS>();
  const T H               = parameters.get<parameters::CHANNEL_HEIGHT>();

  const T physDeltaX = chord / resolution;

  // Domain length = upstream margin + chord + downstream margin.
  const T physLengthX = (upstreamChords + T{1} + downstreamChords) * chord;
  // +physDeltaX on Y matches the cylinder2d trick (keeps the body off the
  // node line so top/bottom strips are symmetric); confirm with numerics.
  const T physLengthY = H + physDeltaX;

  const Vector<T,2> extent{physLengthX, physLengthY};
  const Vector<T,2> origin{T{0}, T{0}};

  IndicatorCuboid2D<T> cuboid(extent, origin);
  Mesh<T,MyCase::d> mesh(cuboid, physDeltaX, singleton::mpi().getSize());
  mesh.setOverlap(parameters.get<parameters::OVERLAP>());
  return mesh;
}

// ===========================================================================
// prepareGeometry
// ===========================================================================
void prepareGeometry(MyCase& myCase)
{
  OstreamManager clout(std::cout, "prepareGeometry");
  clout << "Prepare Geometry ..." << std::endl;

  using T          = MyCase::value_t;
  using DESCRIPTOR = MyCase::descriptor_t;
  auto& parameters = myCase.getParameters();
  auto& geometry   = myCase.getGeometry();

  // ----- reload body + derived sizes (cheap; keeps createMesh stateless) ---
  const BodyPolygon<T> body = loadBodyPolygon<T>(bodyPointFilePath);
  const T chord = body.chord();

  const T resolution       = parameters.get<parameters::RESOLUTION>();
  const T upstreamChords   = parameters.get<parameters::UPSTREAM_CHORDS>();
  const T downstreamChords = parameters.get<parameters::DOWNSTREAM_CHORDS>();
  const T H                = parameters.get<parameters::CHANNEL_HEIGHT>();

  const T physDeltaX  = chord / resolution;
  const T physLengthX = (upstreamChords + T{1} + downstreamChords) * chord;
  const T physLengthY = H + physDeltaX;

  // Publish charL (chord) and centroid for downstream consumers.
  parameters.set<parameters::CHORD_LENGTH>(chord);
  parameters.set<parameters::BODY_CENTROID>(
    Vector<T,2>{body.centroid[0], body.centroid[1]});

  // ----- base fill: everything boundary -> 2, interior bulk -> 1 -----------
  //   rename(0,2)          : all domain voxels become 2 (placeholder boundary)
  //   rename(2,1,{1,1})    : voxels with a full 1-cell neighbourhood become 1
  //                          (bulk fluid); leaves a 1-cell shell as material 2.
  geometry.rename(0, 2);
  geometry.rename(2, 1, {1, 1});

  // ----- inlet (material 3): left strip ------------------------------------
  {
    Vector<T,2> extent{T{2} * physDeltaX, physLengthY};
    Vector<T,2> origin{-physDeltaX, T{0}};
    IndicatorCuboid2D<T> inflow(extent, origin);
    geometry.rename(2, 3, 1, inflow);
  }

  // ----- outlet (material 4): right strip ----------------------------------
  {
    Vector<T,2> extent{T{2} * physDeltaX, physLengthY};
    Vector<T,2> origin{physLengthX - physDeltaX, T{0}};
    IndicatorCuboid2D<T> outflow(extent, origin);
    geometry.rename(2, 4, 1, outflow);
  }

  // ----- free-stream (material 2): top AND bottom strips -------------------
  //   Whatever remains as material 2 after carving out inlet/outlet IS the
  //   top + bottom shell, so it already carries the free-stream number.
  //   Nothing further to do; left explicit here for the reader.

  // ----- sponge zone (material 6): bulk band just upstream of the outlet ---
  //   Absorbing layer to kill the trapped longitudinal acoustic mode. Carved
  //   from BULK material 1 (rename 1->6 inside the band), so it only claims
  //   interior fluid cells -- the top/bottom free-stream shell (2) and the
  //   outlet strip (4) are left untouched (rename only acts on cells currently
  //   == 1). The band spans the last SPONGE_CHORDS chords of the domain,
  //   ending at the inner edge of the outlet strip. Dynamics + the graded
  //   DAMPING ramp are assigned in lattice.h::prepareLattice; here we only tag
  //   the cells. Done BEFORE the body stamp so material 5 always wins on the
  //   (non-overlapping, far-upstream) body cells.
  {
    const T spongeChords = parameters.get<parameters::SPONGE_CHORDS>();
    if (spongeChords > T{0}) {
      const T spongeLen = spongeChords * chord;
      // The InterpolatedPressure outlet (material 4) needs a BULK (material 1)
      // fluid neighbour for its finite-difference stencil + discrete-normal
      // classification. If the sponge (6) sits directly against the outlet, the
      // boundary set fails ("Could not set Boundary"). So leave a small buffer of
      // bulk material 1 between the sponge and the outlet: the band ends
      // BUFFER_CELLS cells upstream of the outlet inner edge. The outlet origin
      // is at x = physLengthX - physDeltaX (a 2-cell-wide strip). We end the
      // sponge 2 cells before that.
      const T bufferCells  = T{2};
      const T xOutletInner = physLengthX - physDeltaX;
      const T xSpongeEnd   = xOutletInner - bufferCells * physDeltaX;
      const T xSpongeStart = xSpongeEnd - spongeLen;

      // The top/bottom free-stream strips (material 2) carry an InterpolatedVelocity
      // boundary whose finite-difference stencil + discrete-normal classification
      // need a BULK material-1 neighbour. If the sponge band runs the full channel
      // height it sits directly under those strips in the sponge x-range, leaving
      // them with material-6 neighbours -> "Could not set Boundary". So inset the
      // band vertically too: leave a bufferCells-thick ribbon of bulk material 1
      // between the band and the top/bottom shells.
      const T yInset = bufferCells * physDeltaX;
      Vector<T,2> extent{spongeLen, physLengthY - T{2} * yInset};
      Vector<T,2> origin{xSpongeStart, yInset};
      IndicatorCuboid2D<T> spongeBand(extent, origin);
      geometry.rename(1, 6, spongeBand);
    }
  }

  // ----- body (material 5): the polygon ------------------------------------
  //   Translate the body so its leading edge (minX) sits `upstreamChords`
  //   chords from the inlet, and centre it vertically in H.
  const T shiftX = upstreamChords * chord - body.minX;
  const T shiftY = (physLengthY / T{2}) - body.centroid[1];
  std::shared_ptr<IndicatorF2D<T>> placedShape =
    makePolygonIndicator<T>(body, Vector<T,2>{shiftX, shiftY});

  geometry.rename(1, 5, placedShape);

  // ----- finalise + verify -------------------------------------------------
  //   clean() removes boundary voxels (3/4) that have NO bulk-fluid neighbour.
  //   The sponge band (material 6) is fluid carved from the bulk and sits
  //   directly against the outlet strip (material 4); unless 6 is declared bulk
  //   here, clean() sees the outlet's only neighbour as "non-bulk" and deletes
  //   the entire outlet (observed: outlet(4) -> 0, geometry destroyed). So treat
  //   BOTH 1 and 6 as bulk for the cleanliness check.
  geometry.clean(true, {1, 6});

  // Dump material map as .vti for a visual sanity check (ParaView).
  {
    SuperVTMwriter2D<T> vtmWriter("ellipse2d_geometry");
    SuperGeometryF2D<T> materials(geometry);
    SuperLatticeCuboid2D<T,DESCRIPTOR> cuboidNo(myCase.getLattice(NavierStokes{}));
    SuperLatticeRank2D<T,DESCRIPTOR>   rankNo(myCase.getLattice(NavierStokes{}));
    vtmWriter.write(materials);
    vtmWriter.write(cuboidNo);
    vtmWriter.write(rankNo);
    vtmWriter.createMasterFile();
  }

  geometry.checkForErrors();
  geometry.print();

  // ----- assert non-empty inlet / outlet / body ----------------------------
  const auto& stats = geometry.getStatistics();
  const int nInlet  = stats.getNvoxel(3);
  const int nOutlet = stats.getNvoxel(4);
  const int nBody   = stats.getNvoxel(5);
  const int nSponge = stats.getNvoxel(6);
  clout << "voxel counts: inlet(3)=" << nInlet
        << " outlet(4)=" << nOutlet
        << " body(5)="   << nBody
        << " sponge(6)=" << nSponge << std::endl;

  if (nInlet == 0 || nOutlet == 0 || nBody == 0) {
    throw std::runtime_error(
      "Empty material region after voxelisation -- "
      "inlet(3)=" + std::to_string(nInlet) +
      " outlet(4)=" + std::to_string(nOutlet) +
      " body(5)="  + std::to_string(nBody) +
      ". Check RESOLUTION (cells across chord), CHANNEL_HEIGHT, and the body "
      "polygon file. A zero body count usually means the SDF sign is flipped "
      "or the body fell outside the domain.");
  }

  clout << "Prepare Geometry ... OK" << std::endl;
}
