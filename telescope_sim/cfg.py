"""Telescope simulation configuration dataclasses."""

from dataclasses import dataclass, field
from typing import List, Optional

import logging
import pyrallis

logger = logging.getLogger(__name__)


@dataclass()
class DetectorConfig:
    """Detector noise configuration"""

    # (NOT IMPLEMENTED) Total number of photons/m^2 from FOV
    integrated_photon_flux: float = 1e5
    # RMS read noise for each point in detector grid (counts) (HCIPy, NoisyDetector)
    read_noise: int = 2
    # Turns the photon noise on or off (HCIPy, NoisyDetector)
    include_photon_noise: bool = True


@dataclass()
class SpiderConfig:
    """Secondary spiders configuration"""

    # Width of spider in pupil plane (meters)
    width: float = 0.03175
    # Angle of spider orientation (degrees)
    angle: Optional[float] = None
    # Randomize spide angle each initialization (if angle not set)
    random_angle: bool = True


@dataclass()
class FilterConfig:
    """Focal-plane configuration"""

    # Central wavelength of focal-plane observation (meters)
    central_lambda: float = 7.5e-7
    # Angular extent of simulated PSF (arcsec)
    # Default: Plate scale of .0207 arcsec/pix * resolution
    focal_extent: float = 0.0207 * 128
    # Resolution of simulated PSF (this and extent set pixel scale for extended image convolution)
    focal_resolution: int = 128
    # Fractional bandwidth of filter. I.e., (1 +/- frac_bandwidth / 2) * central_lam
    frac_bandwidth: float = 0.172
    # Number of monochromatic PSFs across the bandwidth range
    bandwidth_samples: int = 7


@dataclass()
class DMConfig:
    """
    Deformable mirror configuration.
    See hcipy.deformable_mirror.make_xinetics_influence_functions
    """

    # When HCIPy DM args are specified, one can either approximate PTT or actuate the DM directly
    # when sampling (but no simultaneous PTT + DM actuation for now)
    directly_actuate: bool = False
    # The number of actuators across the pupil. The total number of actuators will be this number
    # squared. (HCIPy)
    num_actuators_across_pupil: int = 35
    # Pupil-plane spacing of actuators (meters) before tilting the deformable mirror. (HCIPy)
    actuator_spacing: float = 0.1125
    # The tilt of the deformable mirror around the x-axis in radians. (HCIPy)
    x_tilt: float = 0.0
    # The tilt of the deformable mirror around the y-axis in radians. (HCIPy)
    y_tilt: float = 0.0
    # The tilt of the deformable mirror around the z-axis in radians. (HCIPy)
    z_tilt: float = 0.0


@dataclass()
class MirrorConfig:
    """Mirror geometry configuration"""

    # Mirror layout for building geometry: (keck, monolithic, elf)
    layout: str = "elf"
    # Distance from telescope center to aperature centers (ELF) or mirror edge (meters)
    telescope_radius: float = 1.25
    # Number of apertures in annulus (ELF)
    num_apertures: Optional[int] = 15
    # Radius of each sub-aperature (meters). Default (None) is maximal filling
    subaperture_radius: Optional[float] = None
    # Ratio of central obscuration diameter to pupil diameter (monolithic only)
    central_obscuration_ratio: Optional[float] = 0.25


@dataclass
class PupilPlaneConfig:
    """Pupil plane configuration"""

    # Resolution of pupil plane simulation
    resolution: int = int(2**8)
    # Sub-aperture piston actuation scale (meters)
    piston_actuate_scale: float = 1e-6
    # Sub-aperture tip and tilt actuation scale (microns/meters ~= radians)
    tip_tilt_actuate_scale: float = 1e-6


@dataclass
class AtmosphereConfig:
    """Atmosphere configuration"""

    # Atmosphere layer type: (single, multi)
    layer_type: Optional[str] = None
    # Fried parameter, r0 @ 550nm (meters)
    fried_parameter: float = 0.20
    # Outer scale (meters)
    outer_scale: float = 200.0
    # Velocity (meters/second) (FAILS with multi-layer atmosphere above 10 m/s)
    velocity: float = 10.0
    # Telescope slew velocity (deg/sec)
    slew_deg_per_sec: float = 1.0
    # Direction w.r.t. focal-plane of slewing (degrees, [0, 360])
    slew_focal_plane_angle: float = 0.0
    # Simulate atmospheric scintillation in multi-layer atmosphere (FAILS TO RENDER)
    scintillation: bool = False


@dataclass()
class SimulateMultiAperatureConfig:
    """Multi-aperture telescope simulation configuration"""

    # Pupil plane configuration
    pupil: PupilPlaneConfig = PupilPlaneConfig
    # Telescope mirror geometry configuration
    mirror: MirrorConfig = MirrorConfig
    # List of filter configurations serving to define focal plane parameters
    filters: List = field(default_factory=lambda: [FilterConfig()])
    # Noisy detector configuration
    detector: Optional[DetectorConfig] = DetectorConfig
    # Deformable mirror configuration (HCIPy)
    dm: Optional[DMConfig] = None
    # Spider configuration (HCIPy)
    spiders: Optional[SpiderConfig] = None
    # Atmosphere configruation
    atmosphere: Optional[AtmosphereConfig] = AtmosphereConfig
    # Bundle FFT (real, imag) with PSF samples for machine learning
    include_fft: bool = False
    # Normalize PSFs by max acheivable intensity
    max_intensity_norm: bool = True
    # Normalize each PSF individually from [0-1]
    per_sample_norm: bool = False
    # Scale sampler output by this power
    power_scale: Optional[float] = 0.05
    # Sigma of gaussian noise, added after int norm but before power scaling
    gauss_noise: Optional[float] = None
    # Path of image to convolve PSF with (if none, PSF returned).
    # Must match focal-plane resolution if noise is provided.
    extended_object_image: Optional[str] = None
    # Sigma of the normal distribution of errors to add to piston, tip, and tilt each step.
    # Modulated by PTT scales set in config.
    add_ptt_perturbations_sigma: Optional[float] = None
    # Apply the best fit actuator corrections (PTT or DM)
    apply_optimal_actuator_corrections: bool = False
    # Toggle environment render function
    render: bool = False
    # Number of steps to run simulation
    num_steps: int = 20
    # Time granularity of each simulation step (seconds)
    step_dt: float = 0.01

# deserialize yaml/cli string to class, give Wrap object
pyrallis.decode.register(List, lambda x: [FilterConfig(**x)])

if __name__ == "__main__":
    """Create the simulator config, export to yaml"""
    sim_cfg = pyrallis.parse(config_class=SimulateMultiAperatureConfig)
    pyrallis.dump(sim_cfg, open("./configs/default.yaml", "w"))
