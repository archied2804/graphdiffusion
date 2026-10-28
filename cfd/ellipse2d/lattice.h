/*  ellipse2d -- lattice + boundary-condition setup (OpenLB v1.9.0, 2D CPU)
 *
 *  Numerics-owning half of the case. Consumes the material-number contract
 *  published by geometry.h:
 *
 *      0  unused
 *      1  bulk fluid                -> BGKdynamics (plain incompressible BGK)
 *      2  top AND bottom strips     -> FREE-STREAM velocity Dirichlet (U,0)
 *      3  left strip                -> velocity INLET, uniform (U,0)
 *      4  right strip               -> pressure OUTLET, p = 0
 *      5  body polygon              -> Bouzidi curved no-slip wall
 *
 *  Modelled on examples/laminar/cylinder2d/case.h (prepareLattice /
 *  setInitialValues / setTemporalValues) but with an OPEN free-stream domain
 *  rather than the cylinder2d no-slip Poiseuille channel:
 *
 *    - material 2 is a velocity Dirichlet free-stream (NOT BounceBack),
 *    - the inlet (material 3) imposes a spatially UNIFORM (U,0) (NOT Poiseuille),
 *    - the whole field is launched from a uniform stream with a time ramp on
 *      the inlet + free-stream Dirichlet values for launch stability.
 *
 *  D2Q9, plain BGK, NO turbulence model (target Re 500-1000 is laminar).
 *
 *  Requires geometry.h (loadBodyPolygon / makePolygonIndicator / BodyPolygon /
 *  bodyPointFilePath / parameters::*) to be included first.
 */
#pragma once

#include <olb.h>

// SpongeLayerDynamics (absorbing layer, Xu & Sagaut JCP 2013) -- pulled in
// explicitly because it is not part of the default dynamics aggregation header
// in v1.9.0. Provides olb::SpongeLayerDynamics<T,DESCRIPTOR>.
#include <dynamics/spongeLayerDynamics.h>

#include <memory>

using namespace olb;
using namespace olb::descriptors;
using namespace olb::graphics;
using namespace olb::names;

namespace olb::parameters {

// Inlet velocity profile. 0 = spatially UNIFORM (U,0) -- the default, and what
// every ellipse experiment uses. 1 = PARABOLIC (Poiseuille), with the
// Schaefer-Turek convention maxVelocity = 1.5*U_mean, matching the stock
// OpenLB cylinder2d example.
//
// The uniform inlet is an open-domain simplification: it models a body in a
// free stream, not in a duct. P1-F06 exists to measure what that choice costs,
// which is why this is a flag and not a replacement.
struct INLET_PROFILE      : public descriptors::FIELD_BASE<1> { };

// Top/bottom boundary treatment. 0 = FREE-STREAM velocity Dirichlet (U,0) --
// the default open-domain behaviour. 1 = NO-SLIP walls (bounce-back), required
// for a duct benchmark such as Schaefer-Turek.
//
// A parabolic inlet without no-slip walls is physically inconsistent, so
// INLET_PROFILE=1 with WALL_NOSLIP=0 is rejected in prepareLattice rather than
// silently producing a meaningless run.
struct WALL_NOSLIP        : public descriptors::FIELD_BASE<1> { };

// Fraction of MAX_PHYS_T over which the inlet + free-stream velocity is ramped
// from 0 to the full free-stream value (smooth polynomial start-up).
struct RAMP_UP_END_FRACTION : public descriptors::FIELD_BASE<1> { };

// How often (in physical seconds) the ramped velocity field is re-imposed
// during the start-up window.
struct RAMP_UP_UPDATE        : public descriptors::FIELD_BASE<1> { };

}

// ===========================================================================
// prepareLattice
// ===========================================================================
void prepareLattice(MyCase& myCase)
{
  OstreamManager clout(std::cout, "prepareLattice");
  clout << "Prepare Lattice ..." << std::endl;

  using T          = MyCase::value_t;
  using DESCRIPTOR = MyCase::descriptor_t;
  auto& parameters = myCase.getParameters();
  auto& lattice    = myCase.getLattice(NavierStokes{});
  auto& geometry   = myCase.getGeometry();

  // ----- unit converter (from openlb-physics contract) ---------------------
  const int resolution           = parameters.get<parameters::RESOLUTION>();
  const T   latticeRelaxationTime= parameters.get<parameters::LATTICE_RELAXATION_TIME>();
  const T   charPhysVelocity     = parameters.get<parameters::PHYS_CHAR_VELOCITY>();
  const T   Re                   = parameters.get<parameters::REYNOLDS>();
  const T   physDensity          = parameters.get<parameters::PHYS_CHAR_DENSITY>();

  // charPhysLength = chord. prepareGeometry publishes CHORD_LENGTH; if the case
  // is wired so prepareLattice can run before that (or it is 0), reload the
  // body and use its chord directly so the converter is never seeded with 0.
  T charPhysLength = parameters.get<parameters::CHORD_LENGTH>();
  if (!(charPhysLength > T{0})) {
    const BodyPolygon<T> body = loadBodyPolygon<T>(bodyPointFilePath);
    charPhysLength = body.chord();
    parameters.set<parameters::CHORD_LENGTH>(charPhysLength);
  }

  myCase.getLattice(NavierStokes{})
        .setUnitConverter<UnitConverterFromResolutionAndRelaxationTime<T,DESCRIPTOR>>(
    resolution,                            // resolution: voxels per charPhysLength (= per chord)
    latticeRelaxationTime,                 // latticeRelaxationTime: must be > 0.5
    charPhysLength,                        // charPhysLength: chord
    charPhysVelocity,                      // charPhysVelocity: free-stream speed [m/s]
    charPhysVelocity*charPhysLength/Re,    // physViscosity = U*L/Re [m^2/s]
    physDensity                            // physDensity [kg/m^3]
  );
  const UnitConverter<T,DESCRIPTOR>& converter = lattice.getUnitConverter();
  converter.print();

  // ----- material 1: bulk BGK dynamics -------------------------------------
  auto bulkIndicator = geometry.getMaterialIndicator({1});
  dynamics::set<BGKdynamics>(lattice, bulkIndicator);

  // ----- material 3: velocity INLET (uniform (U,0)) ------------------------
  // ----- material 2: top+bottom FREE-STREAM velocity Dirichlet (U,0) -------
  //   Both use the same interpolated (finite-difference) velocity Dirichlet
  //   set. Material 2 here is the flat top/bottom strips only (the inlet/outlet
  //   already claimed the corner cells in prepareGeometry), so the Flat
  //   discrete-normal case of InterpolatedVelocity applies on both edges.
  const bool wallNoSlip     = parameters.get<parameters::WALL_NOSLIP>()   > T{0.5};
  const bool parabolicInlet = parameters.get<parameters::INLET_PROFILE>() > T{0.5};

  if (parabolicInlet && !wallNoSlip) {
    throw std::runtime_error(
      "INLET_PROFILE=1 (parabolic) requires WALL_NOSLIP=1: a Poiseuille profile "
      "is only defined between walls. Refusing to run an inconsistent case.");
  }

  boundary::set<boundary::InterpolatedVelocity>(lattice, geometry, 3);
  if (wallNoSlip) {
    // Duct walls, as in the stock cylinder2d example.
    boundary::set<boundary::BounceBack>(lattice, geometry, 2);
  }
  else {
    boundary::set<boundary::InterpolatedVelocity>(lattice, geometry, 2);
  }

  // ----- material 4: pressure OUTLET (p = 0) -------------------------------
  boundary::set<boundary::InterpolatedPressure>(lattice, geometry, 4);

  // ----- material 5: body, Bouzidi curved no-slip wall ---------------------
  //   Reconstruct the IDENTICAL placed indicator that prepareGeometry built and
  //   discarded: reload the body, recompute shiftX/shiftY exactly as
  //   geometry.h lines 365-366, re-wrap with makePolygonIndicator. Bouzidi
  //   needs the TRUE signed-distance indicator (negative inside), which
  //   makePolygonIndicator provides.
  {
    const BodyPolygon<T> body = loadBodyPolygon<T>(bodyPointFilePath);
    const T chord            = body.chord();
    const T upstreamChords   = parameters.get<parameters::UPSTREAM_CHORDS>();
    const T H                = parameters.get<parameters::CHANNEL_HEIGHT>();
    const T physDeltaX       = chord / T(resolution);
    const T physLengthY      = H + physDeltaX;

    const T shiftX = upstreamChords * chord - body.minX;          // geometry.h:365
    const T shiftY = (physLengthY / T{2}) - body.centroid[1]      // geometry.h:366
                   + parameters.get<parameters::BODY_Y_OFFSET>();
    std::shared_ptr<IndicatorF2D<T>> placedShape =
      makePolygonIndicator<T>(body, Vector<T,2>{shiftX, shiftY});

#ifdef BOUZIDI
    // Default OPERATOR = BouzidiPostProcessor (zero wall velocity); correct for
    // a stationary body. Signature confirmed: setBouzidiBoundary.h:214.
    setBouzidiBoundary(lattice, geometry, 5, *placedShape);
#else
    boundary::set<boundary::BounceBack>(lattice, geometry, 5);
#endif
  }

  // ----- material 6: absorbing SPONGE layer (just upstream of the outlet) --
  //   SpongeLayerDynamics (Xu & Sagaut, JCP 2013, eq. 24): a normal BGK collide
  //   PLUS an extra relaxation, of strength sigma=DAMPING, of the post-collision
  //   populations toward the equilibrium of a FIXED reference state
  //   (DENSITY=rhoRef, VELOCITY=uRef). That drags the local state back toward the
  //   undisturbed freestream, dissipating acoustic waves and vortices before they
  //   reach the fixed-pressure outlet (material 4) and reflect. We KEEP material 4
  //   as the InterpolatedPressure density anchor (do NOT remove it) -- the sponge
  //   only damps the band ahead of it.
  //
  //   Reference state = freestream: rhoRef = 1 (p=0), uRef = (uLat, 0) where
  //   uLat = converter.getCharLatticeVelocity() (the post-ramp freestream speed
  //   in lattice units; SpongeLayerDynamics reads VELOCITY/DENSITY as LATTICE
  //   quantities, exactly like the gausspulse3d reference).
  //
  //   DAMPING ramps 0 -> SPONGE_SIGMA_MAX quadratically from the upstream edge of
  //   the band to the outlet edge, so the layer is a GRADED ramp (no step), which
  //   itself minimises reflection off the sponge entry.
  {
    auto spongeIndicator = geometry.getMaterialIndicator({6});
    if (geometry.getStatistics().getNvoxel(6) > 0) {
      lattice.defineDynamics<SpongeLayerDynamics<T,DESCRIPTOR>>(spongeIndicator);

      const T uLat        = converter.getCharLatticeVelocity();
      const T spongeChords= parameters.get<parameters::SPONGE_CHORDS>();
      const T sigmaMax    = parameters.get<parameters::SPONGE_SIGMA_MAX>();

      // Reconstruct the band geometry exactly as geometry.h carved it so the
      // ramp coordinate matches the tagged cells.
      const T physLengthX  = (parameters.get<parameters::UPSTREAM_CHORDS>()
                              + T{1}
                              + parameters.get<parameters::DOWNSTREAM_CHORDS>())
                             * charPhysLength;
      const T physDeltaX   = charPhysLength / T(resolution);
      // MUST match geometry.h's band carve exactly: 2-cell bulk buffer before
      // the outlet, band of length spongeLen ending at xSpongeEnd.
      const T bufferCells  = T{2};
      const T xOutletInner = physLengthX - physDeltaX;
      const T xSpongeEnd   = xOutletInner - bufferCells * physDeltaX;
      const T spongeLen    = spongeChords * charPhysLength;
      const T xSpongeStart = xSpongeEnd - spongeLen;

      // Reference freestream fields the dynamics relaxes toward.
      AnalyticalConst2D<T,T> rhoRef(T{1});
      AnalyticalConst2D<T,T> uRef(uLat, T{0});
      lattice.defineField<descriptors::DENSITY>(spongeIndicator, rhoRef);
      lattice.defineField<descriptors::VELOCITY>(spongeIndicator, uRef);

      // Graded DAMPING sigma(x): 0 at xSpongeStart -> sigmaMax at xOutletInner,
      // quadratic so it eases in. Outside the band it returns 0 (harmless; only
      // material-6 cells carry SpongeLayerDynamics).
      struct SpongeSigma2D final : public AnalyticalF2D<T,T> {
        T x0, x1, smax;
        SpongeSigma2D(T x0_, T x1_, T smax_)
          : AnalyticalF2D<T,T>(1), x0(x0_), x1(x1_), smax(smax_) {}
        bool operator()(T out[], const T in[]) override {
          T xi = (in[0] - x0) / (x1 - x0);          // 0 at start, 1 at outlet edge
          if (xi < T{0}) xi = T{0};
          if (xi > T{1}) xi = T{1};
          out[0] = smax * xi * xi;                  // quadratic graded ramp
          return true;
        }
      };
      SpongeSigma2D sigmaF(xSpongeStart, xSpongeEnd, sigmaMax);
      lattice.defineField<descriptors::DAMPING>(spongeIndicator, sigmaF);

      clout << "sponge zone (mat 6): " << spongeChords << " chords, sigmaMax="
            << sigmaMax << ", uRef(lat)=" << uLat << std::endl;
    }
  }

  // ----- relaxation frequency ----------------------------------------------
  lattice.setParameter<descriptors::OMEGA>(converter.getLatticeRelaxationFrequency());

  clout << "Prepare Lattice ... OK" << std::endl;
}


// ===========================================================================
// setInitialValues
//
//   Launch the ENTIRE field from REST (u=0), CONSISTENT with the velocity ramp
//   in setTemporalValues (inlet + top/bottom free-stream ramp 0 -> U over the
//   first RAMP_UP_END_FRACTION of the run). The interior then accelerates with
//   the ramped inflow.
//
//   WHY rest, not full free-stream: initialising the interior at full (U,0)
//   while the inlet/free-stream BCs start the ramp at 0 is INCONSISTENT --
//   at t=0 the bulk moves at U but the boundaries impose 0, which fires an
//   impulsive pressure wave that excites the domain's fundamental longitudinal
//   acoustic mode (quarter-wave, period ~4L/cs). That mode was observed as a
//   sustained +/-15% avgRho "breathing" in the prior runs. Rest-init removes
//   the t=0 forcing; the absorbing outflow (below) damps the residual mode.
//
//   defineRhoU / iniEquilibrium take LATTICE units directly (they do NOT
//   rescale); rho = 1 (p = 0).
// ===========================================================================
void setInitialValues(MyCase& myCase)
{
  using T          = MyCase::value_t;
  using DESCRIPTOR = MyCase::descriptor_t;
  auto& lattice    = myCase.getLattice(NavierStokes{});
  auto& geometry   = myCase.getGeometry();

  AnalyticalConst2D<T,T> uF(T{0}, T{0});  // LATTICE units; start at REST
  AnalyticalConst2D<T,T> rhoF(T{1});

  // Material 6 (sponge) is fluid carved from bulk material 1, so it must be
  // initialised too (at REST, like the rest of the field). Its own reference
  // DENSITY/VELOCITY fields (set in prepareLattice) drive the sponge relaxation;
  // those are independent of this REST population init.
  lattice.defineRhoU   (geometry.getMaterialIndicator({1, 2, 3, 4, 6}), rhoF, uF);
  lattice.iniEquilibrium(geometry.getMaterialIndicator({1, 2, 3, 4, 6}), rhoF, uF);

  lattice.initialize();
}

// ===========================================================================
// setTemporalValues
//
//   Smooth polynomial ramp of the inlet (material 3) AND the top/bottom
//   free-stream (material 2) velocity from 0 to (U,0) over the first
//   RAMP_UP_END_FRACTION * MAX_PHYS_T seconds. Constant in space, ramped in
//   time. After the window the Dirichlet value stays at full (U,0).
//
//   WHY a ramp at tau = 0.545: launching a body into a step-change uniform
//   stream injects an impulsive pressure wave at t = 0. At Re = 1000 with
//   uLat ~ 0.05 (Ma ~ 0.087) and tau only 0.045 above the 0.5 stability floor,
//   that transient can overshoot the BGK stability envelope on the windward
//   stagnation cells before the wake organises. cylinder2d ramps for exactly
//   this reason. The ramp is cheap insurance and removes the launch transient
//   as a failure mode; it is NOT a viscosity band-aid.
// ===========================================================================
void setTemporalValues(MyCase& myCase, std::size_t iT)
{
  using T          = MyCase::value_t;
  using DESCRIPTOR = MyCase::descriptor_t;
  auto& parameters = myCase.getParameters();
  auto& lattice    = myCase.getLattice(NavierStokes{});
  auto& geometry   = myCase.getGeometry();

  const UnitConverter<T,DESCRIPTOR>& converter = lattice.getUnitConverter();

  const T maxPhysT = parameters.get<parameters::MAX_PHYS_T>();
  const T U        = parameters.get<parameters::PHYS_CHAR_VELOCITY>();

  const std::size_t iTmaxStart = converter.getLatticeTime(
    maxPhysT * parameters.get<parameters::RAMP_UP_END_FRACTION>());
  const std::size_t iTupdate = util::max(
    std::size_t{1},
    converter.getLatticeTime(parameters.get<parameters::RAMP_UP_UPDATE>()));

  if (iT % iTupdate == 0 && iT <= iTmaxStart) {
    T frac{};
    PolynomialStartScale<T,std::size_t>(iTmaxStart, T(1))(&frac, &iT);

    const T uRamped = U * frac;  // PHYSICAL units; setVelocity rescales

    if (parameters.get<parameters::INLET_PROFILE>() > T{0.5}) {
      // Parabolic (Poiseuille) inlet. The Schaefer-Turek convention is
      // U_max = 1.5*U_mean in 2D, and `distance2Wall = 0.5*physDeltaX` places
      // the no-slip plane half a cell outside the first fluid node -- both
      // taken from the stock cylinder2d example so L0 and L1 differ only in
      // the case, not in the profile definition.
      const T charLength = parameters.get<parameters::CHORD_LENGTH>();
      const T physDeltaX = charLength / parameters.get<parameters::RESOLUTION>();
      Poiseuille2D<T> poiseuilleU(
        geometry, 3, T{1.5} * uRamped, T{0.5} * physDeltaX);
      momenta::setVelocity(lattice, geometry.getMaterialIndicator(3), poiseuilleU);
      // Walls are bounce-back under this option; nothing to impose on 2.
    }
    else {
      AnalyticalConst2D<T,T> uF(uRamped, T{0});
      // Inlet (3) AND free-stream top/bottom (2) ramped together.
      momenta::setVelocity(lattice, geometry.getMaterialIndicator(3), uF);
      momenta::setVelocity(lattice, geometry.getMaterialIndicator(2), uF);
    }

    lattice.setProcessingContext<Array<momenta::FixedVelocityMomentumGeneric::VELOCITY>>(
      ProcessingContext::Simulation);
  }
}
