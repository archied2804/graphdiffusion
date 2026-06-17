/*  ellipse2d -- driver (OpenLB v1.9.0, 2D CPU)
 *
 *  Flow around an elliptical (arbitrary closed-polygon) cylinder in an open
 *  free-stream channel. This translation unit owns:
 *    - main(): parameter defaults, CLI overrides, body-point-file resolution,
 *      and the createMesh -> Case -> prepareGeometry -> prepareLattice ->
 *      setInitialValues -> simulate ordering (mirrors cylinder2d.cpp).
 *    - simulate(): the time-stepping loop (modelled on cylinder2d case.h
 *      simulate(), lines 341-365).
 *
 *  Geometry, lattice/BC, and results layers live in the three headers, which
 *  MUST be included in this order: geometry.h defines MyCase + most params and
 *  must come first; lattice.h and results.h add their own params and consume
 *  the material-number contract published by geometry.h.
 */

#include <olb.h>

#include "geometry.h"   // MyCase, parameters, createMesh, prepareGeometry, bodyPointFilePath
#include "lattice.h"     // prepareLattice, setInitialValues, setTemporalValues
#include "results.h"     // getResults

using namespace olb;
using namespace olb::descriptors;
using namespace olb::graphics;
using namespace olb::names;

// ===========================================================================
// simulate
//
//   Time-stepping loop modelled on examples/laminar/cylinder2d/case.h
//   simulate() (lines 341-365). getResults (results.h) owns all output.
// ===========================================================================
void simulate(MyCase& myCase)
{
  OstreamManager clout(std::cout, "simulate");
  using T = MyCase::value_t;

  auto& parameters = myCase.getParameters();
  const T physMaxT = parameters.get<parameters::MAX_PHYS_T>();
  const std::size_t iTmax =
    myCase.getLattice(NavierStokes{}).getUnitConverter().getLatticeTime(physMaxT);

  clout << "starting simulation..." << std::endl;
  util::Timer<T> timer(iTmax, myCase.getGeometry().getStatistics().getNvoxel());
  timer.start();

  for (std::size_t iT = 0; iT < iTmax; ++iT) {
    /// === Step 8.1: Update the Boundary Values and Fields at Times ===
    setTemporalValues(myCase, iT);

    /// === Step 8.2: Collide and Stream Execution ===
    myCase.getLattice(NavierStokes{}).collideAndStream();

    /// === Step 8.3: Computation and Output of the Results ===
    getResults(myCase, timer, iT);
  }

  timer.stop();
  timer.printSummary();
}

int main(int argc, char* argv[])
{
  initialize(&argc, &argv);
  singleton::directories().setOutputDir("./tmp/");

  // --- resolve the body point-file path from the CLI ----------------------
  //   OpenLB v1.9.0 ParametersD::fromCLI parses numeric overrides in the form
  //   `--NAME VALUE` (a `--`-prefixed flag token followed by its value as a
  //   SEPARATE token; see src/case/parametersD.h:83 and io/cliReader.h). It
  //   does NOT use the `NAME=VALUE` form.
  //
  //   The body point-file path is the lone POSITIONAL argument: the first
  //   token that is neither a `--`-prefixed flag nor the value belonging to
  //   one. We therefore skip any `--flag` token AND the token immediately
  //   following it. A token containing '=' is also skipped for robustness in
  //   case the `NAME=VALUE` form is ever used.
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg.empty()) {
      continue;
    }
    if (arg.rfind("--", 0) == 0) {   // a `--NAME` flag: skip it and its value
      ++i;                           // consume the following value token
      continue;
    }
    if (arg[0] == '-') {             // any other dash-led token
      continue;
    }
    if (arg.find('=') != std::string::npos) {
      continue;
    }
    bodyPointFilePath = arg;         // first true positional token => path
    break;
  }
  // If none given, bodyPointFilePath stays empty and loadBodyPolygon throws a
  // clear error.

  /// === Step 2: Set Parameters (production defaults; CLI may override) ===
  MyCase::ParametersD myCaseParameters;
  {
    using namespace olb::parameters;

    // Overlap must be set before createMesh (it calls get<OVERLAP>() and
    // mesh.setOverlap()). OVERLAP defaults to 3 in src/case/parameters.h; we
    // set it explicitly for clarity. It is a TYPED_FIELD_BASE<unsigned,1>.
    myCaseParameters.set<OVERLAP>(3u);

    // Discretisation / flow regime.
    myCaseParameters.set<RESOLUTION>(300);
    myCaseParameters.set<REYNOLDS>(1000.);
    myCaseParameters.set<MAX_PHYS_T>(200.);
    myCaseParameters.set<PHYS_CHAR_VELOCITY>(1.0);
    myCaseParameters.set<PHYS_CHAR_DENSITY>(1.0);
    myCaseParameters.set<LATTICE_RELAXATION_TIME>(0.545);

    // Geometry (createMesh derives the domain extent from body + chords, so
    // DOMAIN_EXTENT is intentionally NOT set for this case).
    myCaseParameters.set<CHANNEL_HEIGHT>(8.0);
    myCaseParameters.set<SEMI_MINOR_B>(0.3);
    myCaseParameters.set<UPSTREAM_CHORDS>(6.);
    myCaseParameters.set<DOWNSTREAM_CHORDS>(17.);

    // Absorbing sponge zone (material 6) just upstream of the pressure outlet,
    // to damp the trapped longitudinal acoustic mode (the +/-15% avgRho
    // "breathing"). Graded SpongeLayerDynamics relaxing toward the freestream.
    // SPONGE_CHORDS must be < DOWNSTREAM_CHORDS so it stays clear of the wake.
    myCaseParameters.set<SPONGE_CHORDS>(4.);
    myCaseParameters.set<SPONGE_SIGMA_MAX>(0.7);
    // CHORD_LENGTH defaults to 0 (prepareGeometry/prepareLattice fill it from
    // the body file). BODY_CENTROID default {0,0}.
    myCaseParameters.set<CHORD_LENGTH>(0.);
    myCaseParameters.set<BODY_CENTROID>({0., 0.});

    // Launch ramp.
    myCaseParameters.set<RAMP_UP_END_FRACTION>(0.2);
    myCaseParameters.set<RAMP_UP_UPDATE>(0.05);

    // Output cadence.
    myCaseParameters.set<PHYS_VTK_ITER_T>(2.);
    myCaseParameters.set<PHYS_STAT_ITER_T>(0.5);

    // Time-averaging / washout.
    myCaseParameters.set<WASHOUT_WINDOW_CHORDS>(10.);
    myCaseParameters.set<WASHOUT_EPSILON>(5e-2);
    myCaseParameters.set<AVG_WINDOW_CHORDS>(50.);
  }
  myCaseParameters.fromCLI(argc, argv);

  /// === Step 3: Create Mesh ===
  Mesh mesh = createMesh(myCaseParameters);

  /// === Step 4: Create Case ===
  MyCase myCase(myCaseParameters, mesh);

  /// === Step 5: Prepare Geometry ===
  prepareGeometry(myCase);

  /// === Step 6: Prepare Lattice ===
  prepareLattice(myCase);

  /// === Step 7: Definition of Initial, Boundary Values, and Fields ===
  setInitialValues(myCase);

  /// === Step 8: Simulate ===
  simulate(myCase);
}
