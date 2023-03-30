"""
Helper class for building and running multi-aperture telescope simulations with MultiAperturePSFSampler

Author: Ian Cunnyngham (Institute for Astronomy, University of Hawai'i) 2021
"""


from pprint import pp
from typing import List, Optional, Tuple

import hcipy
import numpy as np
import pyrallis
from matplotlib import pyplot as plt
from telescope_sim.cfg import SimulateMultiAperatureConfig
from telescope_sim.multi_aperture_psf import MultiAperturePSFSampler


class SimulateMultiApertureTelescope:
    def __init__(self, cfg: SimulateMultiAperatureConfig):

        self.cfg = cfg

        # Create parameters for HCIPy pupil plane, deriving from mirror dataclass
        (
            self.num_apertures,
            self.pupil_plane_diameter,
            self.aperture_centers,
            self.aperture_config,
        ) = self._set_telescope()

        # Create spiders from spider dataclass
        self.spider_config = self._set_spiders()

        # Create filters configs from filter dataclass
        self.filters = self._set_filters()

        # Create detectors based on number of filters
        self.int_phot_flux = self.cfg.detector.integrated_photon_flux
        self.detector_configs = self._set_detector()
        for f, d in zip(self.filters, self.detector_configs):
            if d is not None:
                f["detector_config"] = d

        # Create deformable mirror configuration
        self.dm_config = self._set_dm()
        # (Make an easily accessible variable to show when the sampler is expecting direct DM actuation
        # For now, simulator.get_observation(..., dm_actuate=...) how you use this.)
        if self.cfg.dm is not None:
            self.direct_dm_actuation = (
                self.cfg.dm.directly_actuate
                and self.cfg.dm.num_actuators_across_pupil > 0
            )
        else:
            self.direct_dm_actuation = False

        # Build MultiAperturePSFSampler configuration dictionary
        self.sampler_config = {
            "mirror_config": {
                "positions": self.aperture_centers,
                "aperture_config": self.aperture_config,
                "pupil_extent": self.pupil_plane_diameter,
                "pupil_res": self.cfg.pupil.resolution,
                "piston_scale": self.cfg.pupil.piston_actuate_scale,
                "tip_tilt_scale": self.cfg.pupil.tip_tilt_actuate_scale,
                "spider_config": self.spider_config,
                "dm_confg": self.dm_config,
                #"aprox_ptt_wih_dm": not (self.direct_dm_actuation),
                "aprox_ptt_wih_dm": False,
            },
            "filter_configs": self.filters,
        }

        pp(self.sampler_config)

        # Construct MultiAperturePSFSampler from sampler_config
        self.mas_psf_sampler = MultiAperturePSFSampler(**self.sampler_config)

        # Setup atmosphere (optional)
        if (
            self.cfg.atmosphere.layer_type != "single"
            and self.cfg.atmosphere.layer_type != "multi"
        ):
            self.atmos = None
        else:
            self.generate_atmosphere(
                r0=self.cfg.atmosphere.fried_parameter,
                outer_scale=self.cfg.atmosphere.outer_scale,
                velocity=self.cfg.atmosphere.velocity,
                multi=(self.cfg.atmosphere.layer_type == "multi"),
                scintillation=self.cfg.atmosphere.scintillation,
            )

        # Setup slewing simulation by imposing wind-velocities on atmospheres (optional)
        self.slew_deg_per_sec = self.cfg.atmosphere.slew_deg_per_sec
        if self.slew_deg_per_sec is not None:
            self.slew_focal_plane_angle = self.cfg.atmosphere.slew_focal_plane_angle
            slew_th = np.pi * self.slew_focal_plane_angle / 180
            slew_deg_sec = [
                self.slew_deg_per_sec * np.cos(slew_th),
                self.slew_deg_per_sec * np.sin(slew_th),
            ]

        # Load and set extended object if specified
        self.extended_object_image_file = self.cfg.extended_object_image
        if self.extended_object_image_file is not None:
            self.set_extended_object(plt.imread(self.extended_object_image_file))
        else:
            self.set_extended_object(None)

        # Set focal plane extent (arcsecs)
        f_ext = self.sampler_config["filter_configs"][0]["focal_extent"]
        self.plt_focal_extent = [-0.5 * f_ext, 0.5 * f_ext, -0.5 * f_ext, 0.5 * f_ext]

        # Set a reasonable minimum value for focal plane (minimum of perfectly phased)
        x, _ = self.mas_psf_sampler.sample()
        self.plt_focal_logmin = np.log10(x.min())

    def _set_telescope(self) -> Tuple[int, float, hcipy.Grid, list]:
        """
        Create telescope geometry based on mirror dataclass.

        Returns:
            num_apertures: number of telescope apertures
            pupil_plane_diameter: pupil plane diameter (meters)
            aperture_centers: HCIPy grid of aperture center coordinates
            aperture_config: list of args to be sent to HCIPy aperature constructor
        """

        # Build a single monolithic primary
        if self.cfg.mirror.layout == "monolithic":
            num_apertures = 1
            aperture_diameter = 2 * self.cfg.mirror.telescope_radius
            # One mirror at center of pupil plane
            aperture_centers = hcipy.CartesianGrid(np.array([[0], [0]]))
            # Pupil-plane extent should be a bit larger than the mirror
            pupil_plane_diameter = 1.05 * aperture_diameter
            # Specify the aperture config
            aperture_config = [
                "circular_central_obstruction",
                aperture_diameter,
                self.cfg.mirror.central_obscuration_ratio,
            ]

        # Build ELF-like telescope geometry annulus of sub-apertures with centers at telescope_radius
        elif self.cfg.mirror.layout == "elf":
            num_apertures = self.cfg.mirror.num_apertures
            # Linear space of angular coordinates for mirror centers
            thetas = np.linspace(0, 2 * np.pi, num_apertures + 1)[:-1]
            # Use HCIPy coordinate generation to quickly generate mirror centers
            aper_coords = hcipy.SeparatedCoords(
                (np.array([self.cfg.mirror.telescope_radius]), thetas)
            )
            # Create an HCIPy "CartesianGrid" by creating PolarGrid and converting
            aperture_centers = hcipy.PolarGrid(aper_coords).as_("cartesian")
            # Calculate subaperture diameter
            if self.cfg.mirror.subaperture_radius is not None:
                aperture_diameter = 2 * self.cfg.mirror.subaperture_radius
            else:
                # Calculate sub-aperture diameter from the distance between centers
                # (Assuming dense packing of apertures for now, could simulate gaps later)
                aperture_diameter = np.sqrt(
                    (aperture_centers.x[1] - aperture_centers.x[0]) ** 2
                    + (aperture_centers.y[1] - aperture_centers.y[0]) ** 2
                )
            # Calculate extent of pupil-plane simulation (meters)
            pupil_plane_diameter = (
                max(
                    aperture_centers.x.max() - aperture_centers.x.min(),
                    aperture_centers.y.max() - aperture_centers.y.min(),
                )
                + aperture_diameter
            )
            # Add a little extra for edges, not convinced not cutting them off
            pupil_plane_diameter *= 1.05
            aperture_config = ["circular", aperture_diameter]

        # Build a Keck-like layout. Hardocded for now.
        elif self.cfg.mirror.layout == "keck":
            num_apertures = 36
            telescope_radius = 10  # meters
            aper_coords = hcipy.make_hexagonal_grid(
                1.6, 3, False
            )  # 1.6m between points, 3 rows, pack
            aperture_centers = hcipy.CartesianGrid(
                aper_coords[1:].T
            )  # Remove center coordinate, reform
            aperture_config = ["hexagonal", 1.8, np.pi / 2]
            pupil_plane_diameter = 12

        return num_apertures, pupil_plane_diameter, aperture_centers, aperture_config

    def _set_spiders(self) -> Optional[dict]:
        """
        Create spider configuration from spider dataclass.

        Returns:
            spider_config: dict (or None) specifying spider config
        """
        if self.cfg.spiders is not None:
            spider_config = {"width": self.cfg.spiders.width}
            # If spider angle is not defined, set it randomly
            if self.cfg.spiders.angle is None:
                spider_config["random_angle"] = True
            else:
                spider_config["angle"] = self.cfg.spiders.angle
        else:
            #  If spider_width is None, pass an empty config (no spider)
            spider_config = None
        return spider_config

    def _set_filters(self) -> List[dict]:
        """
        Create filter configurations from filter dataclass.

        Returns:
            filter_configs: list of dicts of filter configs
        """
        filter_configs = []
        for filter_cfg in self.cfg.filters:
            # convert FilterConfig to dict
            filter = pyrallis.encode(filter_cfg)
            config = {
                "central_lam": filter["central_lambda"],
                "focal_extent": filter["focal_extent"],
                "focal_res": filter["focal_resolution"],
                "frac_bandwidth": filter["frac_bandwidth"],
                "num_samples": filter["bandwidth_samples"],
            }
            filter_configs.append(config)
        return filter_configs

    def _set_detector(self) -> Optional[List[dict]]:
        """
        Create detector configuration from detector dataclass.

        Returns:
            detector_configs: list of kwarg dicts for HCIPy NoisyDetector or None
        """
        detector_configs = []
        # If integrated photon flux is set, make sure a detector is setup
        if self.cfg.detector.integrated_photon_flux is not None:
            for filter_config in self.cfg.filters:
                detector_config = {
                    "read_noise": self.cfg.detector.read_noise,
                    "include_photon_noise": True,
                }
                detector_configs.append(detector_config)
        else:
            detector_configs = None
        return detector_configs

    def _set_dm(self) -> Optional[list]:
        """
        Create deformable mirror configuration from dm dataclass

        Returns:
            dm_config: list of kwargs for HCIPy dm constructor
        """
        # IF DM config is set, setup for DM approximation of Piston, tip, tilt actuation
        # This is slow, but might be important for fine-tuning the model for bench demo
        if self.cfg.dm is not None and self.cfg.dm.num_actuators_across_pupil > 0:
            dm_config = [
                self.cfg.dm.num_actuators_across_pupil,
                self.cfg.dm.actuator_spacing,
            ]
        else:
            dm_config = None
        return dm_config

    def set_extended_object(self, image):
        """Set extended object image to be convolved with telescope PSF"""
        # Notes:
        # - Plate scale (angular pixel size) fixed in simulator
        #   by focal-plane extent and resolution, scale the images?
        # - In current code, if render is enabled, pupil-plane res and image
        #   resolution must match.  This is of course extremely arbitrary
        #   as they absolutely don't need to be related.
        # - (New: 2020-12): Image and focal plane res need to match when adding
        #                   noise for now.  I should probably fix this later
        #                   (though it does help keep things explicit)
        self.ext_im = image

    def generate_atmosphere(
        self,
        r0,  # (meters) Fried paramater: atmosphere coherence length
        outer_scale=200,  # (meters) outer scale
        velocity=10,  # (meters / second) Layer velocity
        multi=False,  # (bool) Whether to use a single turbulence layer or HCIPys multi-layer
        scintillation=False,  # (bool) Whether to simulate scintilation
    ):
        """Helper to create an HCIPy atmosphere that will be applied when running sim"""

        self.atmosphere_fried_paramater = r0
        self.atmosphere_outer_scale = outer_scale
        self.atmosphere_velocity = velocity

        # Calculate C_n^2 from given Fried param, r0 @ 550nm
        self.cn2 = hcipy.Cn_squared_from_fried_parameter(
            self.atmosphere_fried_paramater, 550e-9
        )

        if multi:
            # Multi-layer atmosphere
            layers = hcipy.make_standard_atmospheric_layers(
                self.mas_psf_sampler.pupil_grid, self.atmosphere_outer_scale
            )
            for i_l in range(len(layers)):
                # Set velocity of each layer to vector of specified magnitude with random direction
                layers[i_l].velocity = self.mas_psf_sampler._from_mag_gen_rand_vec(
                    self.atmosphere_velocity
                )

            self.atmos = hcipy.MultiLayerAtmosphere(layers, scintillation=scintillation)
            self.atmos.Cn_squared = self.cn2

            self.atmos.reset()
        else:
            # Single layer atmosphere
            self.atmos = hcipy.InfiniteAtmosphericLayer(
                self.mas_psf_sampler.pupil_grid,
                self.cn2,
                self.atmosphere_outer_scale,
                self.atmosphere_velocity,
                100,  # Height of single layer in meters, but may not be important for now
            )
        self.simulation_time = 0

    def set_atmos_slew_wind(
        self, slew_deg_sec  # Two element vector (x, y): slew degrees per second
    ):
        """Takes a slew vector in deg/sec and imposes pseudo 'wind velocity' to all atmos layers"""

        if self.atmos is None:
            print("Warning: failed to impose slew speed as no atmosphere is set")
        else:
            # Convert degrees per second into radians
            slew_rad_sec = np.pi * np.array(slew_deg_sec) / 180

            # Iterate the old fashioned way since we're modifying the elements
            for i_l in range(len(self.atmos.layers)):
                l = self.atmos.layers[i_l]
                v, h = l.velocity, l.height
                self.atmos.layers[i_l].velocity += h * slew_rad_sec

    def evolve_to(
        self, simulation_time  # (seconds) Absolute time (from 0) of simulation
    ):
        """Evolve simulation (practically speaking, the atmosphere) until the time specified"""

        self.simulation_time = simulation_time
        if self.atmos != None:
            self.atmos.evolve_until(self.simulation_time)

    def reset(self):
        """Reset simulation (practically speaking, just the atmosphere)"""
        self.simulation_time = 0
        if self.atmos != None:
            self.atmos.reset()

    def get_observation(
        self, piston_tip_tilt=None, dm_actuate=None, int_phot_flux=None
    ):
        """
        Return an observation from the telescope simulator given the current state (atmosphere, ext. image if any, photon flux if any)

        Inputs
        ------
          piston_tip_tilt : (float) (n_apertures, 3): Piston, tip, and tilt actuation for each sub-aperture as multiplied by the
                                                      corresponding scales setup during initialization
          dm_actuate : (float) (n_active_actuators) : If DM is setup and direct actuation enable, accepts piston actuation for all active actuators
          int_phot_flux : (float) : (optional) Set a new photon flux for this observation (photons/m^2)

        Outputs
        -------

        X: Stack of focal plane observations.  PSF by default, extended image convolved with PSF if provided
        Y: Returns the optimal P/T/T (n_aper, 3) phases to get optimal strehl (measured vs atmosphere)
        strehls: If meas_strehls set, returns strehl vs perfectly phase mirror

        """
        if int_phot_flux is not None:
            self.int_phot_flux = int_phot_flux

        X, Y, strehls = self.mas_psf_sampler.sample(
            piston_tip_tilt,  # (n_aper, 3) piston, tip, tilts to set telescope to
            dm_actuate=dm_actuate,
            atmos=self.atmos,  # Pass in HCIPy atmosphere, applied to each pupil-plane (or None is fine)
            convolve_im=self.ext_im,  # Image to convolve PSF with
            # (Note: assuemd matches sampler filters angular extent/pixel scale)
            int_phot_flux=self.int_phot_flux,  # Photons/m^2 for the entire FOV
            meas_strehl=True,  # If True, returns third output which is the measured strehl for each filter
        )

        return X, Y, strehls

    def get_integrated_frame(
        self,
        integration_time=1,  # Integration time in seconds
        n_subframes=20,  # number of subframes to build up frame
        piston_tip_tilt=None,  # Fixed actuation to apply
        dm_actuate=None,
        int_phot_flux=None,
        render_subframes=False,
    ):
        if int_phot_flux is not None:
            self.int_phot_flux = int_phot_flux

        t0 = self.simulation_time

        sub_times = np.linspace(t0, t0 + integration_time, n_subframes + 1)[1:]

        all_Ys, all_strehls = [], []
        for i_t, t_sub in enumerate(sub_times):

            self.evolve_to(t_sub)

            X, Y, strehls = self.mas_psf_sampler.sample(
                piston_tip_tilt,  # (n_aper, 3) piston, tip, tilts to set telescope to
                dm_actuate=dm_actuate,
                atmos=self.atmos,  # Pass in HCIPy atmosphere, applied to each pupil-plane (or None is fine)
                convolve_im=self.ext_im,  # Image to convolve PSF with
                # (Note: assuemd matches sampler filters angular extent/pixel scale)
                int_phot_flux=None,
                meas_strehl=True,  # If True, returns third output which is the measured strehl for each filter
            )
            all_Ys += [Y]
            all_strehls += [strehls]

            if render_subframes:
                plt.figure(figsize=[12, 4])
                self.render(X, strehls)
                plt.show()

            if i_t == 0:
                int_frame = np.copy(X)
            else:
                int_frame += X

        noise_Xs = []
        for i_samp in range(X.shape[2]):
            noise_samp = self.mas_psf_sampler._addNoiseToObservation(
                observation=int_frame[..., i_samp], int_phot_flux=self.int_phot_flux
            )
            noise_Xs += [noise_samp[..., None]]
        final_obs = np.concatenate(noise_Xs, axis=2)

        self.simulation_time += integration_time

        return final_obs, all_Ys, all_strehls

    def pupil_plane_phase_screen(self, np_array=False):
        """Returns the pupil-plane phase screen"""
        return self.mas_psf_sampler.getPhaseScreen(self.atmos, np_array=np_array)

    def render(self, X, strehls=None):
        """Plots the current observation"""

        plt.clf()

        ### Getting the phase screens to plot isn't as pretty as I'd like
        awf1 = self.pupil_plane_phase_screen()

        plt.subplot(121)
        hcipy.imshow_field(
            awf1,
            mask=self.mas_psf_sampler.aper,
            cmap="twilight_shifted",
            vmin=-np.pi,
            vmax=np.pi,
        )
        plt.ylabel("pupil plane (m)")
        plt.colorbar()

        ### Plot pupil and PSF with this atmosphere
        obs = X[..., 0]

        plt.subplot(122)
        im = plt.imshow(
            np.log10(obs),
            vmin=self.plt_focal_logmin,
            cmap="inferno",
            extent=self.plt_focal_extent,
        )
        plt.ylabel("focal plane (arcsec)")
        cbar = plt.colorbar(im)

        if strehls is not None:
            plt.title(f"strehl {strehls[0]:.03f}")

        plt.pause(0.1)


def main():
    # parse CLI args or yaml config
    cfg = pyrallis.parse(config_class=SimulateMultiAperatureConfig)

    n_steps = cfg.num_steps
    t_delta = cfg.step_dt
    add_ptt_errs_sigma = cfg.add_ptt_perturbations_sigma
    apply_corrections = cfg.apply_optimal_actuator_corrections
    render = cfg.render

    # Build simulator from the rest of the keyword arguments
    telescope_sim = SimulateMultiApertureTelescope(cfg)

    # Shape of piston, tip, tilt actuation
    ptt_shape = (telescope_sim.num_apertures, 3)
    ptt_actuation = np.zeros(ptt_shape)

    # Setup plot if rendering is enabled
    if render:
        plt.figure(figsize=[12, 4])
        plt.show(block=False)

    # Iterate through all time-steps
    ts = np.linspace(0, n_steps * t_delta, n_steps, endpoint=False)
    for t in ts:
        print(t, end=" ")

        telescope_sim.evolve_to(t)

        # Add random errors (shouldn't do anything if 0 (default))
        if add_ptt_errs_sigma is not None:
            ptt_actuation += np.random.normal(0, add_ptt_errs_sigma, ptt_shape)

        X, Y, strehls = telescope_sim.get_observation(ptt_actuation)

        # If cfg.apply_optimal_actuator_corrections is set, run the simulation again with optimal actuator fits
        if apply_corrections:
            if telescope_sim.direct_dm_actuation:
                # If direct DM actuation is setup, Y will be the optimal fit of the pupil plane errors
                X, Y, strehls = telescope_sim.get_observation(
                    ptt_actuation, dm_actuate=-Y
                )
            else:
                # Otherwise, Y is the typical set of PTT actuations fit to the pupil plane
                X, Y, strehls = telescope_sim.get_observation(ptt_actuation - Y)

        if render:
            telescope_sim.render(X, strehls)

    print("")


if __name__ == "__main__":
    main()
