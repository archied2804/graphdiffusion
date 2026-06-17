/*  ellipse2d -- results / output layer (OpenLB v1.9.0, 2D CPU)
 *
 *  Owns "how results are written so they open correctly in ParaView" and how
 *  the time-averaged validation metrics (Cd, Cl, surface Cp) are accumulated
 *  and dumped to CSV. Consumes the material-number contract from geometry.h:
 *
 *      0 unused, 1 bulk, 2 free-stream top/bottom, 3 inlet, 4 outlet, 5 body.
 *
 *  charL = chord (parameters::CHORD_LENGTH); body centroid in
 *  parameters::BODY_CENTROID; the body polygon is reloaded + re-placed with the
 *  SAME shiftX/shiftY as geometry.h:365-366 / lattice.h to recover the surface
 *  nodes for Cp probing.
 *
 *  WHY TIME-AVERAGING IS MANDATORY
 *  -------------------------------
 *  Target Re 500-1000 is in the unsteady, periodic vortex-shedding regime: the
 *  instantaneous Cd/Cl oscillate at the shedding frequency and a single-step
 *  readout is meaningless. We therefore (1) wait for the initial launch
 *  transient to wash out, then (2) accumulate running mean +/- std-dev of
 *  Cd/Cl and an ensemble-averaged pressure field over a window spanning many
 *  shedding periods.
 *
 *  Modelled on:
 *    - examples/laminar/cylinder2d/case.h getResults (VTK series + PhysDrag),
 *    - examples/thermal/squareCavity2d/turbulent/case.h (ValueTracer washout +
 *      SuperLatticeTimeAveragedF2D::addEnsemble averaging window).
 *
 *  Requires geometry.h and lattice.h to be included first (MyCase,
 *  BodyPolygon/loadBodyPolygon/makePolygonIndicator, bodyPointFilePath,
 *  parameters::*).
 *
 *  PARAMETERS
 *  ----------
 *  Reused (built-in src/case/parameters.h): PHYS_VTK_ITER_T, PHYS_STAT_ITER_T,
 *    MAX_PHYS_T, PHYS_CHAR_VELOCITY, PHYS_CHAR_DENSITY, RESOLUTION.
 *  Reused (geometry.h): CHORD_LENGTH, BODY_CENTROID, UPSTREAM_CHORDS,
 *    CHANNEL_HEIGHT.
 *  Added below: WASHOUT_WINDOW_CHORDS, WASHOUT_EPSILON, AVG_WINDOW_CHORDS.
 */
#pragma once

#include <olb.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <string>
#include <vector>

using namespace olb;
using namespace olb::descriptors;
using namespace olb::graphics;
using namespace olb::names;

namespace olb::parameters {

// Length of the transient-washout detection window, in chord lengths of phys
// time (U = charPhysVelocity). The ValueTracer that watches instantaneous Cd
// uses a sliding window of getLatticeTime(WASHOUT_WINDOW_CHORDS*chord/U)
// samples. With St ~ 0.2 the shedding period is T_shed ~ 5*chord/U, so a
// default of ~10 chords covers ~2 shedding periods -- long enough that the
// relative spread of Cd over the window stops shrinking once the launch
// transient is gone (see WASHOUT_EPSILON note below).
struct WASHOUT_WINDOW_CHORDS : public descriptors::FIELD_BASE<1> { };

// Relative-spread threshold (stdDev/average over the washout window) below
// which the INITIAL TRANSIENT is declared washed out. NOTE: because shedding
// is PERIODIC not steady, the instantaneous Cd never goes flat -- its window
// stdDev/average plateaus at a finite value set by the oscillation amplitude.
// We therefore set WASHOUT_EPSILON GENEROUSLY (e.g. 5e-2 = 5%), so
// hasConverged() fires when the transient decay has settled into a steady
// periodic oscillation, NOT when Cd is constant. This is a transient detector,
// not a convergence criterion.
struct WASHOUT_EPSILON       : public descriptors::FIELD_BASE<1> { };

// Length of the statistics-accumulation window AFTER washout, in chord lengths
// of phys time. Spec: >= 10 shedding periods; with T_shed ~ 5*chord/U that is
// >= 50 chords. Default 50. Mean/std of Cd,Cl and the pressure ensemble are
// accumulated every step over getLatticeTime(AVG_WINDOW_CHORDS*chord/U) iters.
struct AVG_WINDOW_CHORDS     : public descriptors::FIELD_BASE<1> { };

}

// ===========================================================================
// getResults
//
//   Called once per time step from simulate(). All persistent state lives in
//   function-local statics so the writer / tracer / averagers survive between
//   calls (matching the cylinder2d `static Gnuplot` and squareCavity2d
//   averaging pattern).
//
//   Phases:
//     A. VTK instantaneous-field series (every PHYS_VTK_ITER_T).
//     B. Console stats + Cd/Cl readout (every PHYS_STAT_ITER_T).
//     C. Washout detection (ValueTracer on instantaneous Cd).
//     D. Post-washout accumulation: Welford running mean/std of Cd,Cl +
//        addEnsemble() on time-averaged pressure, over AVG_WINDOW_CHORDS.
//     E. At end of window: dump ellipse2d_force.csv + ellipse2d_cp.csv.
// ===========================================================================
void getResults(MyCase& myCase,
                util::Timer<MyCase::value_t>& timer,
                std::size_t iT)
{
  OstreamManager clout(std::cout, "getResults");
  using T          = MyCase::value_t;
  using DESCRIPTOR = MyCase::descriptor_t;

  auto& parameters = myCase.getParameters();
  auto& lattice    = myCase.getLattice(NavierStokes{});
  auto& geometry   = myCase.getGeometry();

  const UnitConverter<T,DESCRIPTOR>& converter = lattice.getUnitConverter();

  // ----- physical scales / iter cadences -----------------------------------
  const T chord   = parameters.get<parameters::CHORD_LENGTH>();
  const T U       = parameters.get<parameters::PHYS_CHAR_VELOCITY>();
  const T rho     = parameters.get<parameters::PHYS_CHAR_DENSITY>();
  const T maxPhysT = parameters.get<parameters::MAX_PHYS_T>();

  const std::size_t vtkIter  =
    converter.getLatticeTime(parameters.get<parameters::PHYS_VTK_ITER_T>());
  const std::size_t statIter =
    converter.getLatticeTime(parameters.get<parameters::PHYS_STAT_ITER_T>());

  // Washout window (samples) and accumulation window (iters), in phys time
  // expressed as multiples of chord/U (U is the characteristic velocity).
  const T   tConvect      = chord / U;                       // one convective time
  const int washoutWindow = util::max<int>(1, (int) converter.getLatticeTime(
      parameters.get<parameters::WASHOUT_WINDOW_CHORDS>() * tConvect));
  const T   washoutEps    = parameters.get<parameters::WASHOUT_EPSILON>();
  const std::size_t avgWindow = util::max<std::size_t>(std::size_t{1},
      converter.getLatticeTime(
        parameters.get<parameters::AVG_WINDOW_CHORDS>() * tConvect));

  // =========================================================================
  // Persistent state (statics). Constructed on the first call only.
  // =========================================================================

  // (A) Instantaneous-field VTK series. Same base name => ParaView "open as
  //     series": writes tmp/vtkData/ellipse2d.pvd (master) + per-iT
  //     tmp/vtkData/data/ellipse2diT<zero-padded>.vti/.vtm. write(iT) appends a
  //     <DataSet timestep="iT"> entry to the .pvd so the series auto-loads.
  static SuperVTMwriter2D<T> vtmWriter("ellipse2d");
  static SuperLatticePhysVelocity2D<T,DESCRIPTOR> velocity(lattice, converter);
  static SuperLatticePhysPressure2D<T,DESCRIPTOR> pressure(lattice, converter);
  static bool vtkInit = [&]() {
    vtmWriter.addFunctor(velocity);   // PHYSICAL velocity [m/s]
    vtmWriter.addFunctor(pressure);   // PHYSICAL pressure [Pa]
    return true;
  }();
  (void) vtkInit;

  // (C) Washout detector: ValueTracer<T>(deltaT, epsilon) keeps a sliding
  //     window of `deltaT` samples and hasConverged() == relative spread
  //     (stdDev/average) over that window < epsilon. We feed it the
  //     INSTANTANEOUS Cd; see WASHOUT_EPSILON note -- it detects transient
  //     decay settling into steady periodic shedding, not true convergence.
  static util::ValueTracer<T> cdWashout(washoutWindow, washoutEps, "Cd_washout");
  static bool   washedOut    = false;
  static std::size_t washoutIT = 0;   // iT at which accumulation started

  // (D) Hand-rolled Welford accumulators for Cd, Cl mean+std (gives std
  //     directly, no second ValueTracer needed).
  static std::size_t nAcc = 0;
  static T cdMean = T{}, cdM2 = T{};
  static T clMean = T{}, clM2 = T{};

  // (C/D) Time-averaged PHYSICAL pressure field. Composition is legal:
  //     SuperLatticeTimeAveragedF2D<T>(SuperF2D&) wraps the pressure functor;
  //     its target dim is 2*sourceDim = 2 (index 0 = TA mean, index 1 = RMS).
  static SuperLatticeTimeAveragedF2D<T> taPressure(pressure);

  static bool dumped = false;         // CSVs written once

  // =========================================================================
  // (A) Geometry meta-data on iT == 0 (cuboid + rank + master .pvd file)
  // =========================================================================
  if (iT == 0) {
    SuperLatticeCuboid2D<T,DESCRIPTOR> cuboid(lattice);
    SuperLatticeRank2D<T,DESCRIPTOR>   rank(lattice);
    vtmWriter.write(cuboid);
    vtmWriter.write(rank);
    vtmWriter.createMasterFile();     // writes ellipse2d.pvd master for the series
  }

  // =========================================================================
  // (A) Instantaneous-field VTK series
  // =========================================================================
  if (iT % vtkIter == 0 && iT > 0) {
    lattice.setProcessingContext(ProcessingContext::Evaluation);
    vtmWriter.write(iT);              // zero-padded iT; appended to .pvd
  }

  // =========================================================================
  // Drag/lift functor (returns COEFFICIENTS already, see normalisation note).
  // =========================================================================
  SuperLatticePhysDrag2D<T,DESCRIPTOR> drag(lattice, geometry, 5, converter);

  // Helper: read instantaneous (Cd, Cl). Requires Evaluation context.
  auto readCdCl = [&](T& cd, T& cl) {
    lattice.setProcessingContext(ProcessingContext::Evaluation);
    T out[2] = {};
    int in[3] = {};                   // super-integral reduces over all blocks; input ignored
    drag(out, in);
    cd = out[0];                      // Cd = 2/(rho U^2 A_proj) * F_x
    cl = out[1];                      // Cl = 2/(rho U^2 A_proj) * F_y
  };

  // =========================================================================
  // (B) Console stats + washout feeding (every statIter)
  // =========================================================================
  if (iT % statIter == 0) {
    lattice.setProcessingContext(ProcessingContext::Evaluation);

    timer.update(iT);
    timer.printStep();
    lattice.getStatistics().print(iT, converter.getPhysTime(iT));

    T cd{}, cl{};
    readCdCl(cd, cl);
    clout << "t=" << converter.getPhysTime(iT)
          << "; Cd=" << cd << "; Cl=" << cl
          << (washedOut ? " [averaging]" : " [transient]") << std::endl;
  }

  // =========================================================================
  // (C) Averaging start -- begin averaging the FINAL `avgWindow` iters of the
  //     run. A stdDev/average convergence test on instantaneous Cd is
  //     UNRELIABLE for vortex shedding: the relative spread of an oscillating
  //     signal never falls below epsilon, so hasConverged() can never fire and
  //     averaging would never begin. We therefore discard a fixed transient
  //     (everything before the tail window) and average a deterministic tail.
  //     The ValueTracer is retained ONLY as an optional EARLY start for flows
  //     that genuinely settle to a steady state.
  // =========================================================================
  if (!washedOut) {
    // Sample instantaneous Cd only at the stat cadence. On GPU every functor
    // readout forces a device->host sync; per-step sampling serialises the
    // device (observed: ~33 MLUPs vs >>that when sampled at cadence). ~50
    // samples over the window is ample for a time-mean.
    if (statIter > 0 && iT % statIter == 0) {
      T cd{}, cl{};
      readCdCl(cd, cl);
      cdWashout.takeValue(cd);        // no per-step print (noisy)
    }

    const std::size_t iTmax = converter.getLatticeTime(maxPhysT);
    const std::size_t tailStart =
      (iTmax > avgWindow) ? (iTmax - avgWindow)
                          : static_cast<std::size_t>(T{0.2} * iTmax);
    // Never declare washout before the inlet ramp has fully finished.
    const bool rampDone = converter.getPhysTime(iT) > T{0.2} * maxPhysT;
    if ((rampDone && cdWashout.hasConverged()) || iT >= tailStart) {
      washedOut = true;
      washoutIT = iT;
      clout << "*** averaging started at t=" << converter.getPhysTime(iT)
            << " (final " << parameters.get<parameters::AVG_WINDOW_CHORDS>()
            << " chords, " << avgWindow << " iters) ***" << std::endl;
    }
  }

  // =========================================================================
  // (D) Post-washout accumulation, at the stat cadence over the averaging
  //     window. Sampling at `statIter` (not every step) avoids a per-step
  //     GPU->host sync; ~50 samples spanning many shedding periods is enough
  //     for the time-mean Cd/Cl and pressure field.
  // =========================================================================
  if (washedOut && !dumped && (iT - washoutIT) < avgWindow
      && statIter > 0 && iT % statIter == 0) {
    lattice.setProcessingContext(ProcessingContext::Evaluation);

    // Cd, Cl Welford update.
    T cd{}, cl{};
    readCdCl(cd, cl);
    ++nAcc;
    {
      const T d1 = cd - cdMean; cdMean += d1 / T(nAcc);
      const T d2 = cd - cdMean; cdM2 += d1 * d2;
    }
    {
      const T d1 = cl - clMean; clMean += d1 / T(nAcc);
      const T d2 = cl - clMean; clM2 += d1 * d2;
    }

    // Pressure ensemble: one snapshot per step (same window as Cd/Cl).
    taPressure.addEnsemble();
  }

  // =========================================================================
  // (E) End of averaging window: emit CSVs + an averaged-field VTK snapshot.
  // =========================================================================
  const std::size_t iTmaxDump = converter.getLatticeTime(maxPhysT);
  if (washedOut && !dumped &&
      ((iT - washoutIT) >= avgWindow || iT + 1 >= iTmaxDump)) {
    dumped = true;
    lattice.setProcessingContext(ProcessingContext::Evaluation);

    const T cdStd = (nAcc > 1) ? util::sqrt(cdM2 / T(nAcc - 1)) : T{0};
    const T clStd = (nAcc > 1) ? util::sqrt(clM2 / T(nAcc - 1)) : T{0};

    clout << "=== time-averaged force coefficients over " << nAcc
          << " samples ===" << std::endl;
    clout << "Cd = " << cdMean << " +/- " << cdStd << std::endl;
    clout << "Cl = " << clMean << " +/- " << clStd << std::endl;

    const std::string outDir = singleton::directories().getLogOutDir();

    // --- ellipse2d_force.csv : one row of mean/std Cd,Cl -------------------
    if (singleton::mpi().isMainProcessor()) {
      std::ofstream f(outDir + "ellipse2d_force.csv");
      f << "Cd_mean,Cd_std,Cl_mean,Cl_std,n_samples,t_start,t_end\n";
      f << std::setprecision(10)
        << cdMean << "," << cdStd << ","
        << clMean << "," << clStd << ","
        << nAcc   << ","
        << converter.getPhysTime(washoutIT) << ","
        << converter.getPhysTime(iT) << "\n";
    }

    // --- write the averaged pressure field as VTK for ParaView ------------
    {
      SuperVTMwriter2D<T> avgWriter("ellipse2d_timeAveraged");
      avgWriter.write(taPressure);    // dim 0 = mean p [Pa], dim 1 = RMS p [Pa]
      avgWriter.createMasterFile();
    }

    // --- surface Cp(x/c) from the TIME-AVERAGED pressure field ------------
    //
    //   Recover the placed body surface nodes by reloading + re-placing the
    //   polygon with the IDENTICAL shiftX/shiftY as geometry.h:365-366 /
    //   lattice.h. Probe the TA pressure ~0.5*dx OUTWARD along each node's
    //   outward normal (into the fluid; the surface node itself is solid /
    //   masked). Cp = (p - p_inf)/(0.5*rho*U^2), p_inf = 0.
    //
    //   ROOT-CAUSE NOTE (why we do NOT use AnalyticalFfromSuperF2D here):
    //   SuperLatticeTimeAveragedF2D is a derived SuperF2D that stores its
    //   averages in SuperData and exposes them ONLY through its super-level
    //   operator() -- it never populates the per-block functor vector
    //   (_blockF is empty; getBlockFSize()==0, no getBlockF override). The
    //   ctor of AnalyticalFfromSuperF2D loops iC=0..load.size()-1 calling
    //   _f.getBlockF(iC) (interpolationF2D.hh:209-214). For the TA functor
    //   that falls through to the base SuperF2D::getBlockF, which dereferences
    //   the EMPTY _blockF[iC] -> out-of-bounds (OLB_ASSERT is a no-op in
    //   release) -> SIGSEGV, before a single probe is evaluated. So analytic
    //   interpolation over the TA functor cannot be made safe.
    //
    //   FALLBACK (documented in spec): evaluate the TA functor on the NEAREST
    //   lattice node. We map each probe point to its (globalCuboid, iX, iY)
    //   via CuboidDecomposition::getLatticeR (returns std::optional -- empty
    //   when the point lies outside every cuboid), guard ownership with the
    //   load balancer's isLocal(), then call taPressure({glob,iX,iY}) which
    //   is the functor's own, supported access path. Points that miss the
    //   domain are SKIPPED, not dereferenced. Output dim is taken from
    //   getTargetDim() (==2: index 0 mean p, index 1 RMS p).
    {
      const BodyPolygon<T> body = loadBodyPolygon<T>(bodyPointFilePath);
      const T resolution     = T(parameters.get<parameters::RESOLUTION>());
      const T upstreamChords = parameters.get<parameters::UPSTREAM_CHORDS>();
      const T H              = parameters.get<parameters::CHANNEL_HEIGHT>();
      const T physDeltaX     = chord / resolution;
      const T physLengthY    = H + physDeltaX;

      // EXACT geometry.h:365-366 placement.
      const T shiftX = upstreamChords * chord - body.minX;
      const T shiftY = (physLengthY / T{2}) - body.centroid[1];
      const Vector<T,2> shift{shiftX, shiftY};

      // Placed surface points + placed centroid (for outward-sign check).
      const std::size_t n = body.points.size();
      std::vector<Vector<T,2>> P(n);
      for (std::size_t i = 0; i < n; ++i) {
        P[i] = body.points[i] + shift;
      }
      const Vector<T,2> centroidPlaced = body.centroid + shift;
      const T leadingEdgeX = body.minX + shiftX;  // x of the placed leading edge

      const T probeOffset = T{0.5} * physDeltaX;
      const T qInf        = T{0.5} * rho * U * U;   // dynamic pressure

      // TA functor access: super structure (cuboid decomposition + balancer)
      // and the true output dimension. Size buffers to getTargetDim() so a
      // dim-2 write never overflows a dim-1 buffer.
      auto& superStruct = taPressure.getSuperStructure();
      auto& cuboidDecomp = superStruct.getCuboidDecomposition();
      auto& loadBalancer = superStruct.getLoadBalancer();
      const int taDim = taPressure.getTargetDim();   // == 2 (mean, RMS)
      const int ensembles = taPressure.getEnsembles();

      // Ensure overlap halo is up to date before reading neighbouring cells.
      superStruct.communicate();

      std::size_t nWritten = 0;

      // Header-only fast path: with zero ensembles the TA operator() divides by
      // _ensembles (==0) -> NaN/inf, so do not read the field.
      if (ensembles <= 0) {
        if (singleton::mpi().isMainProcessor()) {
          std::ofstream f(outDir + "ellipse2d_cp.csv");
          f << "x_over_c,Cp,Cp_rms\n";
          clout << "WARNING: 0 ensembles accumulated; ellipse2d_cp.csv left "
                   "header-only (no averaging samples)." << std::endl;
        }
      }
      else {
        // MPI-CORRECTNESS: the TA pressure field is distributed across ranks,
        // so each surface node can only be read by the rank that OWNS its cell.
        // Every rank probes its locally-owned nodes into per-node arrays; we
        // then MPI_SUM-reduce to the main processor (each node is owned by at
        // most one rank, so the sum recovers the single owning value) and the
        // main processor writes the CSV. (The previous version probed only on
        // the main processor and skipped every non-local node -> empty CSV
        // under domain decomposition, since the body rarely sits on rank 0.)
        std::vector<T> cpLoc(n, T{}), cpRmsLoc(n, T{}), xocLoc(n, T{}),
                       foundLoc(n, T{});
        std::vector<T> out(static_cast<std::size_t>(taDim), T{});

        // Freestream reference pressure p_inf: Cp must reference the UNDISTURBED
        // static pressure at the body's streamwise station, not the outlet
        // gauge (p=0). The channel carries a streamwise pressure level, so
        // referencing the outlet offsets every Cp (stagnation reads >> 1).
        // Probe a point well above the body (deep in the u=1 free-stream) at the
        // body's mid-x; subtract its time-averaged pressure so stagnation -> ~1.
        T pInfLoc = T{}, pInfFoundLoc = T{};
        {
          const Vector<T,2> refPt{ centroidPlaced[0], centroidPlaced[1] + T{0.4} * H };
          const auto rR = cuboidDecomp.getLatticeR(refPt);
          if (rR && loadBalancer.isLocal((*rR)[0])) {
            int in[3] = { (*rR)[0], (*rR)[1], (*rR)[2] };
            std::fill(out.begin(), out.end(), T{});
            if (taPressure(out.data(), in)) {
              pInfLoc = out[0]; pInfFoundLoc = T{1};
            }
          }
        }

        for (std::size_t i = 0; i < n; ++i) {
          // Per-node outward normal: average of the two incident edge normals.
          // The right-hand normal of edge (a->b) is (dy,-dx); we average the
          // two edges sharing node i then fix the sign so the normal points
          // AWAY from the (placed) centroid.
          const Vector<T,2>& prev = P[(i + n - 1) % n];
          const Vector<T,2>& cur  = P[i];
          const Vector<T,2>& next = P[(i + 1) % n];

          Vector<T,2> e0 = cur  - prev;   // incoming edge
          Vector<T,2> e1 = next - cur;    // outgoing edge
          Vector<T,2> nm{ (e0[1] + e1[1]), -(e0[0] + e1[0]) };

          const T len = util::sqrt(nm[0]*nm[0] + nm[1]*nm[1]);
          if (len > std::numeric_limits<T>::epsilon()) {
            nm[0] /= len; nm[1] /= len;
          }
          // Ensure OUTWARD: flip if it points toward the centroid.
          const Vector<T,2> radial = cur - centroidPlaced;
          if (nm[0]*radial[0] + nm[1]*radial[1] < T{0}) {
            nm[0] = -nm[0]; nm[1] = -nm[1];
          }

          // Probe point just outside the surface, in the fluid.
          const Vector<T,2> probe{ cur[0] + probeOffset * nm[0],
                                   cur[1] + probeOffset * nm[1] };

          // Nearest lattice node, or skip if the probe misses the domain.
          const auto latticeR = cuboidDecomp.getLatticeR(probe);
          if (!latticeR) {
            continue;                       // outside every cuboid -> skip
          }
          const int glob = (*latticeR)[0];
          if (!loadBalancer.isLocal(glob)) {
            continue;                       // owned by another rank -> skip
          }

          // Evaluate the TA functor on its OWN supported access path:
          // input = {globalCuboid, iX, iY}.
          int in[3] = { glob, (*latticeR)[1], (*latticeR)[2] };
          std::fill(out.begin(), out.end(), T{});
          if (!taPressure(out.data(), in)) {
            continue;
          }

          xocLoc[i]   = (cur[0] - leadingEdgeX) / chord;
          cpLoc[i]    = (out[0] - T{0}) / qInf;             // p_inf = 0
          cpRmsLoc[i] = (taDim > 1 ? out[1] : T{0}) / qInf;
          foundLoc[i] = T{1};
        }

        // Gather every owning rank's contribution onto the main processor.
        std::vector<T> cpG(n, T{}), cpRmsG(n, T{}), xocG(n, T{}), foundG(n, T{});
        T pInfG = T{}, pInfFoundG = T{};
#ifdef PARALLEL_MODE_MPI
        // MPI build: the field is domain-decomposed, so each node is owned by at
        // most one rank. MPI_SUM-reduce the per-node locals onto the main
        // processor (the sum recovers the single owning value; non-owners hold 0).
        const int cnt = static_cast<int>(n);
        singleton::mpi().reduce(xocLoc.data(),   xocG.data(),   cnt, MPI_SUM);
        singleton::mpi().reduce(cpLoc.data(),    cpG.data(),    cnt, MPI_SUM);
        singleton::mpi().reduce(cpRmsLoc.data(), cpRmsG.data(), cnt, MPI_SUM);
        singleton::mpi().reduce(foundLoc.data(), foundG.data(), cnt, MPI_SUM);
        singleton::mpi().reduce(&pInfLoc,      &pInfG,      1, MPI_SUM);
        singleton::mpi().reduce(&pInfFoundLoc, &pInfFoundG, 1, MPI_SUM);
#else
        // Serial build (PARALLEL_MODE=NONE, e.g. the single-GPU GPU_CUDA build):
        // MpiManager exposes no reduce()/MPI_SUM. There is exactly one domain,
        // isLocal()/isMainProcessor() are always true, so the locals already
        // hold every found value -- the "reduction" is a straight copy.
        xocG    = xocLoc;
        cpG     = cpLoc;
        cpRmsG  = cpRmsLoc;
        foundG  = foundLoc;
        pInfG      = pInfLoc;
        pInfFoundG = pInfFoundLoc;
#endif

        if (singleton::mpi().isMainProcessor()) {
          // Cp offset from the freestream reference (0 if the ref point missed
          // the domain -- then Cp falls back to the outlet gauge).
          const T cpInf = (pInfFoundG > T{0.5}) ? (pInfG / qInf) : T{0};
          clout << "freestream-reference Cp offset (subtracted) = " << cpInf
                << std::endl;

          std::ofstream f(outDir + "ellipse2d_cp.csv");
          f << "x_over_c,Cp,Cp_rms\n";
          f << std::setprecision(8);
          for (std::size_t i = 0; i < n; ++i) {
            if (foundG[i] > T{0.5}) {        // node was owned by exactly one rank
              f << xocG[i] << "," << (cpG[i] - cpInf) << "," << cpRmsG[i] << "\n";
              ++nWritten;
            }
          }
        }
      }
      clout << "wrote " << nWritten << "/" << n
            << " surface Cp samples to " << outDir << "ellipse2d_cp.csv"
            << std::endl;
    }
  }
}
