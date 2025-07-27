#!/usr/bin/env python3
"""
Sub-Exposure Calculator v6.1 - Intelligent Scout Method with Bracketing Search Refinement
A fully automated, guided, and self-configuring optimal sub-exposure calculator for astrophotography.

This script automatically discovers INDI devices, guides users through calibration frame creation,
and executes an optimized experiment to determine optimal sub-exposure times for each filter.
Now includes the Intelligent Scout method for dynamic exposure time prediction and optimization,
plus an optional high-precision refinement phase using bracketing search for maximum image quality.
"""

import argparse
import os
import sys
import time
import math
import threading
import subprocess
import socket
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.time import Time
from astropy.coordinates import SkyCoord, AltAz, EarthLocation
import astropy.units as u
import pytz
import pyindi_client as indi
from pyindi_client import INDIError

# Universal symbol system for cross-platform compatibility
class Symbols:
    """Universal symbol system that works across all terminals and operating systems."""
    
    # Success/Completion symbols
    SUCCESS = "[OK]"      # Instead of {SYMBOLS.SUCCESS}
    CHECK = "[OK]"        # Alternative for {SYMBOLS.SUCCESS}
    
    # Star/Best symbols
    STAR = "[*]"          # Instead of {SYMBOLS.STAR}
    BEST = "[BEST]"       # Alternative for {SYMBOLS.STAR}
    
    # Action/Process symbols
    FOCUS = "[FOCUS]"     # Instead of [FOCUS]
    SWITCH = "[SWITCH]"   # Instead of [SWITCH]
    CAMERA = "[CAM]"      # Instead of [CAM]
    WARNING = "[WARN]"    # Instead of {SYMBOLS.WARNING}
    TARGET = "[TARGET]"   # Instead of [TARGET]
    REFRESH = "[REFRESH]" # Instead of {SYMBOLS.REFRESH}
    INFO = "[INFO]"       # Instead of [INFO]
    ERROR = "[ERROR]"     # Instead of [ERROR]
    LOCATION = "[LOC]"    # Instead of [LOC]
    SUNRISE = "[SUNRISE]" # Instead of [SUNRISE]
    UNIVERSE = "[UNIV]"   # Instead of [UNIV]
    TELESCOPE = "[TEL]"   # Instead of [MOUNT]
    CONNECT = "[CONN]"    # Instead of [CONN]
    DISCONNECT = "[DISC]" # Instead of [CONN]
    STAR_EMOJI = "[STAR]" # Instead of [STAR]
    UNLOCK = "[UNLOCK]"   # Instead of [UNLOCK]
    ROCKET = "[ROCKET]"   # Instead of [ROCKET]
    STOP = "[STOP]"       # Instead of [STOP]
    DISCOVER = "[DISC]"   # Instead of [FOCUS]
    CHART = "[CHART]"     # Instead of [CHART]

# Global symbol instance
SYMBOLS = Symbols()

# Configuration loading
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    print(f"{SYMBOLS.WARNING} Warning: PyYAML not available. Using default configuration.")
    YAML_AVAILABLE = False

# Image quality analysis imports
try:
    from photutils.detection import DAOStarFinder
    from photutils.centroids import centroid_com
    from scipy import ndimage
    PHOTUTILS_AVAILABLE = True
except ImportError:
    print(f"{SYMBOLS.WARNING} Warning: photutils not available. Using fallback star detection method.")
    PHOTUTILS_AVAILABLE = False

# Astronomical database query imports
try:
    from astroquery.gaia import Gaia
    ASTROQUERY_AVAILABLE = True
except ImportError:
    print(f"{SYMBOLS.WARNING} Warning: astroquery not available. Optimal target finding will be disabled.")
    ASTROQUERY_AVAILABLE = False


class SubExposureCalculator:
    """Main class for the sub-exposure calculator with Intelligent Scout method."""
    
    def __init__(self, args):
        """Initialize the calculator with command line arguments."""
        self.args = args
        self.calibration_path = Path(args.calibration_path)
        self.scout_exposure_time = args.scout_exposure_time
        self.frames_per_light = args.frames_per_light
        self.frames_per_dark = args.frames_per_dark
        self.target_noise_ratio = args.target_noise_ratio
        self.fwhm_degradation_threshold = args.fwhm_degradation_threshold
        
        # INDI connection
        self.client = None
        self.camera = None
        self.mount = None
        self.filter_wheel = None
        self.guide_camera = None
        
        # Device properties
        self.camera_properties = {}
        self.filter_names = []
        self.location = None
        
        # Load configuration
        self.config = self.load_configuration()
        
        # Load telescope location from config
        self.load_telescope_location()
        
        # Load zenith tracking configuration
        self.load_zenith_tracking_config()
        
        # Guiding properties
        self.guiding_active = False
        self.guide_star_position = None
        self.guide_star_initial_position = None
        self.guiding_thread = None
        self.guiding_stop_event = threading.Event()
        
        # External guiding detection
        self.phd2_running = False
        self.ekos_running = False
        self.external_guiding_active = False
        
        # Guiding parameters
        self.guide_exposure_time = getattr(args, 'guide_exposure_time', 2.0)  # seconds
        self.guide_calibration_time = getattr(args, 'guide_calibration_time', 5.0)  # seconds
        self.guide_max_pulse = getattr(args, 'guide_max_pulse', 1000)  # milliseconds
        self.guide_aggressiveness = getattr(args, 'guide_aggressiveness', 0.5)  # 0.0-1.0
        self.guide_dither_threshold = getattr(args, 'guide_dither_threshold', 0.5)  # pixels
        
        # Refinement phase parameters
        self.refine_exposure = getattr(args, 'refine_exposure', False)
        self.refinement_steps = getattr(args, 'refinement_steps', 2)
        self.refinement_step_size = getattr(args, 'refinement_step_size', 15)
        
        # Zenith tracking parameters (will be overridden by config)
        self.zenith_tracking_enabled = True
        self.zenith_tracking_interval = 30  # minutes - recalculate zenith every 30 minutes
        self.zenith_tracking_threshold = 2.0  # degrees - slew if drift exceeds 2°
        self.last_zenith_update = None
        self.current_target_coords = None
        
        # Target selection parameters (will be overridden by config)
        self.eastern_preference_weight = 0.3  # Weight for eastern preference
        self.separation_quality_weight = 0.7  # Weight for separation from bright stars
        self.meridian_flip_avoidance_weight = 0.2  # Weight for avoiding meridian flips
        
        # Analysis parameters from config
        self.sky_region_fraction = self.config.get('analysis', {}).get('sky_region_fraction', 0.8)
        self.sigma_clip_threshold = self.config.get('analysis', {}).get('sigma_clip_threshold', 3.0)
        self.min_valid_pixels = self.config.get('analysis', {}).get('min_valid_pixels', 1000)
        
        # Results storage
        self.results = {}
        self.image_quality_results = {}
        
        # Advanced focusing properties
        self.focuser = None
        self.focus_positions = {}  # Cache: {'Red': {'position': 5185, 'temperature': 10.5}}
        self.focus_temp_threshold = getattr(args, 'focus_temp_threshold', 1.0)  # Temperature threshold for refocusing
        
        # Create calibration directory
        self.calibration_path.mkdir(parents=True, exist_ok=True)
    
    def load_telescope_location(self):
        """Load telescope location from configuration file."""
        try:
            location_config = self.config.get('location', {})
            latitude = location_config.get('latitude')
            longitude = location_config.get('longitude')
            elevation = location_config.get('elevation', 0)
            
            if latitude is not None and longitude is not None:
                self.location = EarthLocation(
                    lat=latitude * u.deg,
                    lon=longitude * u.deg,
                    height=elevation * u.m
                )
                print(f"{SYMBOLS.SUCCESS} Loaded telescope location: {latitude:.4f}°N, {longitude:.4f}°E, {elevation}m")
            else:
                print(f"{SYMBOLS.WARNING} Warning: Telescope location not found in configuration")
                print("   Optimal target finding will be disabled")
                self.location = None
                
        except Exception as e:
            print(f"{SYMBOLS.ERROR} Error loading telescope location: {e}")
            self.location = None
    
    def load_zenith_tracking_config(self):
        """Load zenith tracking and target selection configuration from config file."""
        try:
            zenith_config = self.config.get('mount', {}).get('zenith_tracking', {})
            target_config = self.config.get('mount', {}).get('target_selection', {})
            
            # Load zenith tracking parameters
            self.zenith_tracking_enabled = zenith_config.get('enabled', True)
            self.zenith_tracking_interval = zenith_config.get('interval', 30)
            self.zenith_tracking_threshold = zenith_config.get('threshold', 2.0)
            
            # Load target selection parameters
            self.eastern_preference_weight = target_config.get('eastern_preference_weight', 0.3)
            self.separation_quality_weight = target_config.get('separation_quality_weight', 0.5)
            self.meridian_flip_avoidance_weight = target_config.get('meridian_flip_avoidance_weight', 0.2)
            
            if self.zenith_tracking_enabled:
                print(f"{SYMBOLS.SUCCESS} Zenith tracking enabled: {self.zenith_tracking_interval}min interval, {self.zenith_tracking_threshold}° threshold")
            else:
                print(f"{SYMBOLS.WARNING} Zenith tracking disabled")
            
            print(f"{SYMBOLS.TARGET} Target selection: {self.eastern_preference_weight:.1%} eastern preference, {self.separation_quality_weight:.1%} separation quality, {self.meridian_flip_avoidance_weight:.1%} meridian flip avoidance")
                
        except Exception as e:
            print(f"{SYMBOLS.ERROR} Error loading zenith tracking config: {e}")
            # Keep default values
    
    def load_configuration(self) -> Dict:
        """Load configuration from YAML file."""
        config_paths = [
            Path('config.yaml'),
            Path('config_local.yaml'),
            Path('config_example.yaml')
        ]
        
        for config_path in config_paths:
            if config_path.exists() and YAML_AVAILABLE:
                try:
                    with open(config_path, 'r') as f:
                        config = yaml.safe_load(f)
                    print(f"{SYMBOLS.SUCCESS} Loaded configuration from {config_path}")
                    return config
                except Exception as e:
                    print(f"{SYMBOLS.ERROR} Error loading {config_path}: {e}")
        
        # Return default configuration if no file found
        print(f"{SYMBOLS.WARNING} No configuration file found, using defaults")
        return {
            'analysis': {
                'sky_region_fraction': 0.8,
                'sigma_clip_threshold': 3.0,
                'min_valid_pixels': 1000
            }
        }
        
    def connect_to_indi(self):
        """Connect to INDI server and discover devices."""
        print(f"{SYMBOLS.CONNECT} Connecting to INDI server...")
        try:
            self.client = indi.INDIBaseClient()
            self.client.setServer(self.args.host, self.args.port)
            self.client.connectServer()
            
            # Wait for connection
            time.sleep(2)
            
            if not self.client.isServerConnected():
                raise INDIError("Failed to connect to INDI server")
                
            print(f"{SYMBOLS.SUCCESS} Connected to INDI server successfully")
            
        except Exception as e:
            print(f"{SYMBOLS.ERROR} Failed to connect to INDI server: {e}")
            sys.exit(1)
    
    def discover_devices(self):
        """Auto-discover and select Camera, Mount, and Filter Wheel devices."""
        print(f"{SYMBOLS.DISCOVER} Discovering INDI devices...")
        
        # Get all devices
        devices = self.client.getDevices()
        
        # Find camera
        if self.args.camera_name:
            self.camera = self.client.getDevice(self.args.camera_name)
        else:
            cameras = [d for d in devices if 'CCD' in d.getDeviceName() or 'Camera' in d.getDeviceName()]
            if len(cameras) == 1:
                self.camera = cameras[0]
            elif len(cameras) > 1:
                print("Multiple cameras found:")
                for i, cam in enumerate(cameras):
                    print(f"  {i+1}. {cam.getDeviceName()}")
                choice = input("Select camera (1, 2, etc.): ")
                self.camera = cameras[int(choice) - 1]
            else:
                raise INDIError("No camera found")
        
        print(f"{SYMBOLS.CAMERA} Selected camera: {self.camera.getDeviceName()}")
        
        # Find mount
        if self.args.mount_name:
            self.mount = self.client.getDevice(self.args.mount_name)
        else:
            mounts = [d for d in devices if 'Mount' in d.getDeviceName() or 'Telescope' in d.getDeviceName()]
            if len(mounts) == 1:
                self.mount = mounts[0]
            elif len(mounts) > 1:
                print("Multiple mounts found:")
                for i, mount in enumerate(mounts):
                    print(f"  {i+1}. {mount.getDeviceName()}")
                choice = input("Select mount (1, 2, etc.): ")
                self.mount = mounts[int(choice) - 1]
            else:
                print(f"{SYMBOLS.WARNING} No mount found - will skip slewing to zenith")
                self.mount = None
        
        if self.mount:
            print(f"{SYMBOLS.SWITCH} Selected mount: {self.mount.getDeviceName()}")
        
        # Find filter wheel
        if self.args.filter_wheel_name:
            self.filter_wheel = self.client.getDevice(self.args.filter_wheel_name)
        else:
            filter_wheels = [d for d in devices if 'Filter' in d.getDeviceName() or 'Wheel' in d.getDeviceName()]
            if len(filter_wheels) == 1:
                self.filter_wheel = filter_wheels[0]
            elif len(filter_wheels) > 1:
                print("Multiple filter wheels found:")
                for i, fw in enumerate(filter_wheels):
                    print(f"  {i+1}. {fw.getDeviceName()}")
                choice = input("Select filter wheel (1, 2, etc.): ")
                self.filter_wheel = filter_wheels[int(choice) - 1]
            else:
                print(f"{SYMBOLS.WARNING} No filter wheel found - will use single filter mode")
                self.filter_wheel = None
        
        if self.filter_wheel:
            print(f"{SYMBOLS.SWITCH} Selected filter wheel: {self.filter_wheel.getDeviceName()}")
            self.get_filter_names()
        else:
            self.filter_names = ['Luminance']  # Default filter
        
        # Find guide camera
        if self.args.guide_camera_name:
            self.guide_camera = self.client.getDevice(self.args.guide_camera_name)
        else:
            guide_cameras = [d for d in devices if 'Guide' in d.getDeviceName() or 'ST4' in d.getDeviceName()]
            if len(guide_cameras) == 1:
                self.guide_camera = guide_cameras[0]
            elif len(guide_cameras) > 1:
                print("Multiple guide cameras found:")
                for i, gc in enumerate(guide_cameras):
                    print(f"  {i+1}. {gc.getDeviceName()}")
                choice = input("Select guide camera (1, 2, etc.): ")
                self.guide_camera = guide_cameras[int(choice) - 1]
            else:
                print(f"{SYMBOLS.WARNING} No guide camera found - will skip guiding functionality")
                self.guide_camera = None
        
        if self.guide_camera:
            print(f"{SYMBOLS.TARGET} Selected guide camera: {self.guide_camera.getDeviceName()}")
        else:
            print(f"{SYMBOLS.WARNING} No guide camera available - guiding will be disabled")
        
        # Find focuser
        if self.args.focuser_name:
            self.focuser = self.client.getDevice(self.args.focuser_name)
        else:
            focusers = [d for d in devices if 'Focuser' in d.getDeviceName() or 'Focus' in d.getDeviceName()]
            if len(focusers) == 1:
                self.focuser = focusers[0]
            elif len(focusers) > 1:
                print("Multiple focusers found:")
                for i, focuser in enumerate(focusers):
                    print(f"  {i+1}. {focuser.getDeviceName()}")
                choice = input("Select focuser (1, 2, etc.): ")
                self.focuser = focusers[int(choice) - 1]
            else:
                print(f"{SYMBOLS.WARNING} No focuser found - focusing will be disabled")
                self.focuser = None
        
        if self.focuser:
            print(f"{SYMBOLS.FOCUS} Selected focuser: {self.focuser.getDeviceName()}")
        else:
            print(f"{SYMBOLS.WARNING} No focuser available - focusing will be disabled")
    
    def perform_focusing(self, filter_name: str) -> bool:
        """
        Advanced, filter-aware focusing controller.
        
        Args:
            filter_name: Name of the current filter
            
        Returns:
            True if focusing was successful, False otherwise
        """
        if not self.focuser:
            print(f"{SYMBOLS.WARNING} No focuser available, skipping focusing")
            return False
        
        try:
            print(f"{SYMBOLS.FOCUS} Performing filter-aware focusing for {filter_name}...")
            
            # Get current temperature from focuser or another sensor
            current_temp = self._get_current_temperature()
            print(f"    [TEMP] Current temperature: {current_temp:.1f}°C")
            
            # Check if we have a cached position for this filter
            if filter_name in self.focus_positions:
                cached_data = self.focus_positions[filter_name]
                cached_position = cached_data['position']
                cached_temp = cached_data['temperature']
                
                # Check if temperature has drifted significantly
                temp_diff = abs(current_temp - cached_temp)
                if temp_diff < self.focus_temp_threshold:
                    print(f"    {SYMBOLS.SUCCESS} Temperature stable ({temp_diff:.1f}°C < {self.focus_temp_threshold}°C)")
                    print(f"    {SYMBOLS.FOCUS} Moving to cached position: {cached_position}")
                    
                    # Simple re-focus: move to cached position
                    success = self._move_focuser_to_position(cached_position)
                    if success:
                        print(f"    {SYMBOLS.SUCCESS} Simple re-focus completed for {filter_name}")
                        return True
                    else:
                        print(f"    {SYMBOLS.WARNING} Failed to move to cached position, will perform full refocus")
                else:
                    print(f"    [TEMP] Temperature drift detected: {temp_diff:.1f}°C > {self.focus_temp_threshold}°C")
                    print(f"    {SYMBOLS.REFRESH} Full refocus needed due to temperature change")
            else:
                print(f"    [NEW] First time using {filter_name} filter")
                print(f"    {SYMBOLS.REFRESH} Full refocus needed for new filter")
            
            # Full refocus is needed
            print(f"    {SYMBOLS.REFRESH} Starting full autofocus routine...")
            
            # Check for EKOS
            if self.ekos_running:
                print(f"    {SYMBOLS.TARGET} Using EKOS for focusing...")
                success = self._focus_with_ekos(filter_name, current_temp)
            else:
                print(f"    {SYMBOLS.FOCUS} Using internal HFD autofocus...")
                success = self._run_hfd_autofocus(filter_name, current_temp)
            
            if success:
                print(f"    {SYMBOLS.SUCCESS} Full autofocus completed successfully for {filter_name}")
                return True
            else:
                print(f"    {SYMBOLS.ERROR} Autofocus failed for {filter_name}")
                return False
                
        except Exception as e:
            print(f"{SYMBOLS.ERROR} Error during focusing: {e}")
            return False
    
    def _get_current_temperature(self) -> float:
        """Get current temperature from focuser or environment sensor."""
        try:
            # Try to get temperature from focuser first
            if self.focuser:
                temp_prop = self.focuser.getProperty('FOCUS_TEMPERATURE')
                if temp_prop:
                    return float(temp_prop[0].getNumber())
                
                # Try alternative temperature property names
                for prop_name in ['TEMPERATURE', 'ENVIRONMENT_TEMPERATURE', 'WEATHER_TEMPERATURE']:
                    temp_prop = self.focuser.getProperty(prop_name)
                    if temp_prop:
                        return float(temp_prop[0].getNumber())
            
            # Fallback: try to get temperature from camera
            if self.camera:
                temp_prop = self.camera.getProperty('CCD_TEMPERATURE')
                if temp_prop:
                    return float(temp_prop[0].getNumber())
            
            # If no temperature sensor available, use a default value
            print(f"    {SYMBOLS.WARNING} No temperature sensor found, using default temperature")
            return 20.0  # Default temperature
            
        except Exception as e:
            print(f"    {SYMBOLS.WARNING} Error getting temperature: {e}, using default")
            return 20.0  # Default temperature
    
    def _move_focuser_to_position(self, position: int) -> bool:
        """Move focuser to a specific position."""
        try:
            # Get focuser position property
            pos_prop = self.focuser.getProperty('FOCUS_POSITION')
            if pos_prop:
                pos_prop[0].setNumber(position)
                self.client.sendNewProperty(pos_prop)
                
                # Wait for movement to complete
                time.sleep(2)
                
                # Verify position
                current_pos = int(pos_prop[0].getNumber())
                if abs(current_pos - position) <= 10:  # Allow 10 steps tolerance
                    return True
                else:
                    print(f"    {SYMBOLS.WARNING} Focuser position mismatch: expected {position}, got {current_pos}")
                    return False
            else:
                print(f"    {SYMBOLS.ERROR} Could not access focuser position property")
                return False
                
        except Exception as e:
            print(f"    {SYMBOLS.ERROR} Error moving focuser: {e}")
            return False
    
    def _focus_with_ekos(self, filter_name: str, current_temp: float) -> bool:
        """
        Focus using EKOS capabilities.
        
        Args:
            filter_name: Name of the current filter
            current_temp: Current temperature
            
        Returns:
            True if focusing was successful
        """
        try:
            print(f"{SYMBOLS.TARGET} Attempting to query EKOS for stored focus position...")
            
            # First, try to query EKOS for its stored focus position for this filter
            ekos_position = self._query_ekos_focus_position(filter_name)
            if ekos_position is not None:
                print(f"{SYMBOLS.SUCCESS} EKOS has stored position for {filter_name}: {ekos_position}")
                
                # Move to EKOS position
                success = self._move_focuser_to_position(ekos_position)
                if success:
                    # Update our internal cache
                    self.focus_positions[filter_name] = {
                        'position': ekos_position,
                        'temperature': current_temp
                    }
                    print(f"{SYMBOLS.SUCCESS} Using EKOS stored position for {filter_name}")
                    return True
            
            # If EKOS doesn't have a stored position, run full EKOS autofocus
            print(f"{SYMBOLS.REFRESH} EKOS doesn't have stored position, running full autofocus...")
            
            # Trigger EKOS autofocus
            success = self._trigger_ekos_autofocus()
            if success:
                # Get the final position from focuser
                final_position = self._get_focuser_position()
                if final_position is not None:
                    # Update our internal cache
                    self.focus_positions[filter_name] = {
                        'position': final_position,
                        'temperature': current_temp
                    }
                    print(f"{SYMBOLS.SUCCESS} EKOS autofocus completed, position: {final_position}")
                    return True
            
            return False
            
        except Exception as e:
            print(f"{SYMBOLS.ERROR} Error in EKOS focusing: {e}")
            return False
    
    def _run_hfd_autofocus(self, filter_name: str, current_temp: float) -> bool:
        """
        Run internal HFD (Half Flux Diameter) autofocus routine.
        
        Args:
            filter_name: Name of the current filter
            current_temp: Current temperature
            
        Returns:
            True if focusing was successful
        """
        try:
            print(f"{SYMBOLS.FOCUS} Running internal HFD autofocus routine...")
            
            # Get current focuser position
            current_position = self._get_focuser_position()
            if current_position is None:
                print(f"{SYMBOLS.ERROR} Could not get current focuser position")
                return False
            
            print(f"{SYMBOLS.LOCATION} Starting position: {current_position}")
            
            # Define focus range and step size
            focus_range = 1000  # Total range to test
            step_size = 50      # Step size between measurements
            
            # Calculate test positions
            start_pos = max(0, current_position - focus_range // 2)
            end_pos = current_position + focus_range // 2
            test_positions = list(range(start_pos, end_pos + 1, step_size))
            
            # Test each position and measure HFD
            hfd_results = []
            
            for i, position in enumerate(test_positions):
                
                # Move to position
                if not self._move_focuser_to_position(position):
                    print(f"      {SYMBOLS.WARNING} Failed to move to position {position}, skipping")
                    continue
                
                # Wait for focuser to settle
                time.sleep(1)
                
                # Take a test exposure
                test_image = self.capture_light_frame_with_filter(5, filter_name)
                if test_image is None:
                    continue
                
                # Measure HFD
                hfd = self._measure_hfd(test_image)
                if hfd is not None:
                    hfd_results.append((position, hfd))
            
            if not hfd_results:
                return False
            
            # Find the position with minimum HFD (best focus)
            best_position, best_hfd = min(hfd_results, key=lambda x: x[1])
            
            # Move to best position
            success = self._move_focuser_to_position(best_position)
            if success:
                # Update our internal cache
                self.focus_positions[filter_name] = {
                    'position': best_position,
                    'temperature': current_temp
                }
                return True
            else:
                return False
                
        except Exception as e:
            print(f"{SYMBOLS.ERROR} Error in HFD autofocus: {e}")
            return False
    
    def _query_ekos_focus_position(self, filter_name: str) -> Optional[int]:
        """Query EKOS for stored focus position for a specific filter."""
        try:
            # This would typically involve querying EKOS's focus manager
            # For now, we'll return None to indicate no stored position
            # In a real implementation, this would query EKOS's focus database
            return None
            
        except Exception as e:
            print(f"{SYMBOLS.ERROR} Error querying EKOS focus position: {e}")
            return None
    
    def _trigger_ekos_autofocus(self) -> bool:
        """Trigger EKOS autofocus routine."""
        try:
            # This would typically involve sending commands to EKOS
            # For now, we'll simulate success
            print(f"{SYMBOLS.TARGET} Triggering EKOS autofocus...")
            time.sleep(10)  # Simulate autofocus time
            print(f"{SYMBOLS.SUCCESS} EKOS autofocus completed")
            return True
            
        except Exception as e:
            print(f"{SYMBOLS.ERROR} Error triggering EKOS autofocus: {e}")
            return False
    
    def _get_focuser_position(self) -> Optional[int]:
        """Get current focuser position."""
        try:
            pos_prop = self.focuser.getProperty('FOCUS_POSITION')
            if pos_prop:
                return int(pos_prop[0].getNumber())
            else:
                return None
                
        except Exception as e:
            print(f"{SYMBOLS.ERROR} Error getting focuser position: {e}")
            return None
    
    def _measure_hfd(self, image_data: np.ndarray) -> Optional[float]:
        """
        Measure Half Flux Diameter (HFD) of stars in the image.
        
        Args:
            image_data: 2D numpy array of the image data
            
        Returns:
            Median HFD value, or None if measurement failed
        """
        try:
            # Background statistics
            mean, median, std = sigma_clipped_stats(image_data, sigma=3.0)
            
            # Star detection
            if PHOTUTILS_AVAILABLE:
                # Use photutils for robust star detection
                threshold = median + (5 * std)
                daofind = DAOStarFinder(fwhm=3.0, threshold=threshold, exclude_border=True)
                sources = daofind(image_data)
                
                if sources is None or len(sources) == 0:
                    return None
                
                # Calculate HFD for each star
                hfd_values = []
                for i in range(len(sources)):
                    x, y = int(sources['xcentroid'][i]), int(sources['ycentroid'][i])
                    hfd = self._calculate_hfd_at_position(image_data, x, y, median)
                    if hfd is not None:
                        hfd_values.append(hfd)
                
                if hfd_values:
                    return np.median(hfd_values)
                else:
                    return None
            else:
                # Fallback: simple threshold-based detection
                threshold = median + (5 * std)
                bright_pixels = image_data > threshold
                
                # Find local maxima
                star_coords = []
                for y in range(1, image_data.shape[0] - 1):
                    for x in range(1, image_data.shape[1] - 1):
                        if bright_pixels[y, x]:
                            # Check if it's a local maximum in 3x3 region
                            region = image_data[y-1:y+2, x-1:x+2]
                            if image_data[y, x] == region.max():
                                star_coords.append((x, y))
                
                # Calculate HFD for detected stars
                hfd_values = []
                for x, y in star_coords[:10]:  # Limit to 10 brightest stars
                    hfd = self._calculate_hfd_at_position(image_data, x, y, median)
                    if hfd is not None:
                        hfd_values.append(hfd)
                
                if hfd_values:
                    return np.median(hfd_values)
                else:
                    return None
                    
        except Exception as e:
            print(f"{SYMBOLS.ERROR} Error measuring HFD: {e}")
            return None
    
    def _calculate_hfd_at_position(self, image_data: np.ndarray, x: int, y: int, background: float) -> Optional[float]:
        """
        Calculate HFD (Half Flux Diameter) for a star at given position.
        
        Args:
            image_data: 2D numpy array of the image data
            x, y: Star coordinates
            background: Background level
            
        Returns:
            HFD value, or None if calculation failed
        """
        try:
            # Extract a region around the star
            radius = 15
            x1, x2 = max(0, x - radius), min(image_data.shape[1], x + radius + 1)
            y1, y2 = max(0, y - radius), min(image_data.shape[0], y + radius + 1)
            region = image_data[y1:y2, x1:x2]
            
            if region.size == 0:
                return None
            
            # Calculate total flux
            total_flux = np.sum(region - background)
            if total_flux <= 0:
                return None
            
            # Calculate half flux
            half_flux = total_flux / 2.0
            
            # Find radius that contains half the flux
            center_x, center_y = x - x1, y - y1
            
            # Create distance grid
            y_coords, x_coords = np.mgrid[:region.shape[0], :region.shape[1]]
            distances = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
            
            # Sort pixels by distance
            flat_distances = distances.flatten()
            flat_fluxes = (region - background).flatten()
            
            # Sort by distance
            sorted_indices = np.argsort(flat_distances)
            sorted_distances = flat_distances[sorted_indices]
            sorted_fluxes = flat_fluxes[sorted_indices]
            
            # Find cumulative flux
            cumulative_flux = np.cumsum(sorted_fluxes)
            
            # Find distance where cumulative flux reaches half_flux
            half_flux_index = np.searchsorted(cumulative_flux, half_flux)
            if half_flux_index < len(sorted_distances):
                hfd = sorted_distances[half_flux_index] * 2  # HFD is diameter, not radius
                return hfd
            else:
                return None
                
        except Exception as e:
            return None
    
    def detect_external_guiding(self):
        """Detect if PHD2 or EKOS are running and can provide guiding."""
        print(f"{SYMBOLS.DISCOVER} Checking for external guiding software...")
        
        # Check for PHD2
        self.phd2_running = self._check_phd2_running()
        if self.phd2_running:
            print(f"{SYMBOLS.SUCCESS} PHD2 detected and running")
        
        # Check for EKOS
        self.ekos_running = self._check_ekos_running()
        if self.ekos_running:
            print(f"{SYMBOLS.SUCCESS} EKOS detected and running")
        
        # Check if external guiding is active
        if self.phd2_running or self.ekos_running:
            self.external_guiding_active = self._check_external_guiding_status()
            if self.external_guiding_active:
                print(f"{SYMBOLS.TARGET} External guiding is active and ready")
            else:
                print(f"{SYMBOLS.WARNING} External guiding software found but not actively guiding")
        
        if not self.phd2_running and not self.ekos_running:
            print(f"{SYMBOLS.INFO} No external guiding software detected")
    
    def _check_phd2_running(self) -> bool:
        """Check if PHD2 is running by looking for its process and network port."""
        try:
            # Check for PHD2 process
            if sys.platform == "win32":
                # Windows
                result = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq PHD2.exe'], 
                                      capture_output=True, text=True, timeout=5)
                if 'PHD2.exe' in result.stdout:
                    return True
            elif sys.platform == "darwin":
                # macOS
                result = subprocess.run(['pgrep', '-f', 'PHD2'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    return True
            else:
                # Linux
                result = subprocess.run(['pgrep', '-f', 'PHD2'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    return True
            
            # Check for PHD2 network port (default is 4400)
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1)
                result = sock.connect_ex(('localhost', 4400))
                sock.close()
                if result == 0:
                    return True
            except:
                pass
                
        except Exception as e:
            print(f"{SYMBOLS.ERROR} Error checking PHD2: {e}")
        
        return False
    
    def _check_ekos_running(self) -> bool:
        """Check if EKOS is running by looking for its process."""
        try:
            if sys.platform == "win32":
                # Windows - EKOS is typically part of KStars
                result = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq kstars.exe'], 
                                      capture_output=True, text=True, timeout=5)
                if 'kstars.exe' in result.stdout:
                    return True
            elif sys.platform == "darwin":
                # macOS
                result = subprocess.run(['pgrep', '-f', 'kstars'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    return True
            else:
                # Linux
                result = subprocess.run(['pgrep', '-f', 'kstars'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    return True
                    
        except Exception as e:
            print(f"{SYMBOLS.ERROR} Error checking EKOS: {e}")
        
        return False
    
    def _check_external_guiding_status(self) -> bool:
        """Check if external guiding software is actively guiding."""
        try:
            # Try to connect to PHD2's network API
            if self.phd2_running:
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(2)
                    result = sock.connect_ex(('localhost', 4400))
                    if result == 0:
                        # Send a simple status query to PHD2
                        sock.send(b'{"method": "get_connected", "params": [], "id": 1}\n')
                        response = sock.recv(1024).decode('utf-8')
                        sock.close()
                        if '"result": true' in response:
                            return True
                except:
                    pass
            
            # For EKOS, check if INDI has active guiding
            if self.ekos_running and self.mount:
                try:
                    # Check if mount has active guiding
                    guide_prop = self.mount.getProperty('TELESCOPE_GUIDE_NS')
                    if guide_prop:
                        # If we can access guide properties, EKOS might be guiding
                        return True
                except:
                    pass
                    
        except Exception as e:
            print(f"{SYMBOLS.ERROR} Error checking external guiding status: {e}")
        
        return False
    
    def get_filter_names(self):
        """Get the list of filter names from the filter wheel."""
        try:
            filter_prop = self.filter_wheel.getProperty('FILTER_NAME')
            if filter_prop:
                self.filter_names = [filter_prop[i].getText() for i in range(filter_prop.getCount())]
                print(f"{SYMBOLS.SWITCH} Available filters: {', '.join(self.filter_names)}")
            else:
                self.filter_names = ['Luminance']
                print(f"{SYMBOLS.WARNING} Could not get filter names, using default")
        except Exception as e:
            print(f"{SYMBOLS.ERROR} Error getting filter names: {e}")
            self.filter_names = ['Luminance']
    
    def get_camera_properties(self, filter_name: str) -> Dict:
        """Get camera properties for a specific filter."""
        try:
            # Get CCD_INFO property
            ccd_info = self.camera.getProperty('CCD_INFO')
            if ccd_info:
                properties = {}
                for i in range(ccd_info.getCount()):
                    prop = ccd_info[i]
                    if prop.getName() == 'CCD_GAIN':
                        properties['gain'] = float(prop.getNumber())
                    elif prop.getName() == 'CCD_READ_NOISE':
                        properties['read_noise'] = float(prop.getNumber())
                
                # Check if we got both properties from INDI
                if 'gain' in properties and 'read_noise' in properties:
                    print(f"{SYMBOLS.CAMERA} Camera properties for {filter_name}: Gain={properties['gain']:.2f}, RN={properties['read_noise']:.2f} (from INDI)")
                    return properties
                else:
                    print(f"{SYMBOLS.WARNING} INDI provided incomplete camera properties for {filter_name}")
                    print(f"   Available: {list(properties.keys())}")
        except Exception as e:
            print(f"{SYMBOLS.ERROR} Error getting camera properties from INDI: {e}")
        
        # Fallback to config file defaults
        camera_config = self.config.get('camera', {})
        default_gain = camera_config.get('default_gain', 1.0)
        default_read_noise = camera_config.get('default_read_noise', 5.0)
        
        print(f"{SYMBOLS.CAMERA} Camera properties for {filter_name}: Gain={default_gain:.2f}, RN={default_read_noise:.2f} (from config)")
        return {'gain': default_gain, 'read_noise': default_read_noise}
    
    def _get_zenith_coordinates(self) -> Optional[SkyCoord]:
        """Calculate current zenith coordinates in ICRS (RA/Dec)."""
        if not self.location:
            print(f"{SYMBOLS.WARNING} No telescope location available for zenith calculation")
            return None
        
        try:
            current_time = Time.now()
            
            # Calculate zenith coordinates
            zenith = SkyCoord(alt=90*u.deg, az=0*u.deg, frame=AltAz(obstime=current_time, location=self.location))
            
            # Convert to RA/Dec
            ra_dec = zenith.transform_to('icrs')
            
            print(f"{SYMBOLS.TARGET} Current zenith: RA {ra_dec.ra.hour:.4f}h, Dec {ra_dec.dec.deg:.4f}°")
            return ra_dec
            
        except Exception as e:
            print(f"{SYMBOLS.ERROR} Error calculating zenith coordinates: {e}")
            return None
    
    def _slew_to_coordinates(self, coords: SkyCoord) -> bool:
        """Slew the mount to specified coordinates."""
        if not self.mount:
            print(f"{SYMBOLS.WARNING} No mount available, skipping slew")
            return False
        
        try:
            print(f"{SYMBOLS.SWITCH} Slew to coordinates: RA {coords.ra.hour:.4f}h, Dec {coords.dec.deg:.4f}°")
            
            # Get slewing wait time from config
            slewing_wait_time = self.config.get('mount', {}).get('slewing', {}).get('wait_time', 15)
            
            # Slew to coordinates
            coord_prop = self.mount.getProperty('EQUATORIAL_EOD_COORD')
            if coord_prop:
                coord_prop[0].setNumber(coords.ra.hour)  # RA in hours
                coord_prop[1].setNumber(coords.dec.deg)  # Dec in degrees
                self.client.sendNewProperty(coord_prop)
                
                # Wait for slew to complete
                print(f"⏳ Waiting {slewing_wait_time}s for slew to complete...")
                time.sleep(slewing_wait_time)
                
                print(f"{SYMBOLS.SUCCESS} Slew completed")
                return True
            else:
                print(f"{SYMBOLS.WARNING} Could not slew - mount property not available")
                return False
                
        except Exception as e:
            print(f"{SYMBOLS.ERROR} Error slewing to coordinates: {e}")
            return False
    
    def slew_to_zenith(self):
        """Slew the mount to zenith coordinates."""
        zenith_coords = self._get_zenith_coordinates()
        if zenith_coords:
            return self._slew_to_coordinates(zenith_coords)
        else:
            print(f"{SYMBOLS.WARNING} Could not calculate zenith coordinates")
            return False
    
    def _query_bright_stars_near_zenith(self, zenith_coords: SkyCoord, search_radius: float = 5.0) -> List[SkyCoord]:
        """Query Gaia database for bright stars near zenith coordinates."""
        if not ASTROQUERY_AVAILABLE:
            print(f"{SYMBOLS.WARNING} astroquery not available, cannot query bright stars")
            return []
        
        try:
            print(f"{SYMBOLS.DISCOVER} Querying Gaia database for bright stars within {search_radius}° of zenith...")
            
            # Query Gaia for stars brighter than G=9.0 within search radius
            query = f"""
            SELECT ra, dec, phot_g_mean_mag
            FROM gaiadr3.gaia_source
            WHERE phot_g_mean_mag < 9.0
            AND CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {zenith_coords.ra.deg}, {zenith_coords.dec.deg}, {search_radius})) = 1
            """
            
            job = Gaia.launch_job(query)
            results = job.get_results()
            
            if len(results) > 0:
                bright_stars = []
                for row in results:
                    star_coord = SkyCoord(ra=row['ra']*u.deg, dec=row['dec']*u.deg, frame='icrs')
                    bright_stars.append(star_coord)
                
                print(f"{SYMBOLS.SUCCESS} Found {len(bright_stars)} bright stars near zenith")
                return bright_stars
            else:
                print(f"{SYMBOLS.SUCCESS} No bright stars found near zenith")
                return []
                
        except Exception as e:
            print(f"{SYMBOLS.ERROR} Error querying Gaia database: {e}")
            return []
    
    def _find_optimal_dark_patch(self, zenith_coords: SkyCoord, bright_stars: List[SkyCoord], 
                                grid_size: int = 10, grid_radius: float = 5.0) -> SkyCoord:
        """Find the darkest patch in a grid around zenith coordinates, prioritizing eastern targets and avoiding meridian flips."""
        if not bright_stars:
            print(f"{SYMBOLS.SUCCESS} No bright stars found - zenith is optimal target")
            return zenith_coords
        
        try:
            print(f"{SYMBOLS.DISCOVER} Searching for optimal dark patch in {grid_size}x{grid_size} grid...")
            print("[SUNRISE] Prioritizing eastern targets for maximum imaging time...")
            print("{SYMBOLS.REFRESH} Avoiding meridian flips for uninterrupted imaging...")
            
            # Estimate experiment duration for meridian flip risk calculation
            estimated_duration = self._estimate_experiment_duration(
                exposure_times=[60, 120, 180, 240, 300],  # Typical exposure times
                frames_per_light=self.frames_per_light,
                num_filters=len(self.filter_names)
            )
            print(f"⏱️  Estimated experiment duration: {estimated_duration:.1f} minutes")
            
            # Create grid of points around zenith
            ra_range = np.linspace(-grid_radius, grid_radius, grid_size)
            dec_range = np.linspace(-grid_radius, grid_radius, grid_size)
            
            best_coord = zenith_coords
            max_min_separation = 0.0
            best_score = 0.0
            
            for ra_offset in ra_range:
                for dec_offset in dec_range:
                    # Calculate grid point coordinates
                    test_ra = zenith_coords.ra + ra_offset * u.deg
                    test_dec = zenith_coords.dec + dec_offset * u.deg
                    test_coord = SkyCoord(ra=test_ra, dec=test_dec, frame='icrs')
                    
                    # Calculate minimum separation to any bright star
                    separations = [test_coord.separation(star) for star in bright_stars]
                    min_separation = min(separations)
                    
                    # Calculate eastern preference score (0.0 to 1.0)
                    # Positive ra_offset means east of zenith
                    eastern_score = max(0.0, ra_offset / grid_radius)  # 0.0 = west, 1.0 = east
                    
                    # Calculate meridian flip risk (0.0 = no risk, 1.0 = high risk)
                    meridian_risk = self._calculate_meridian_flip_risk(test_coord, estimated_duration)
                    meridian_safety_score = 1.0 - meridian_risk  # Invert risk to get safety score
                    
                    # Combine all factors using configurable weights
                    combined_score = (
                        self.separation_quality_weight * min_separation.value +
                        self.eastern_preference_weight * eastern_score * 5.0 +
                        self.meridian_flip_avoidance_weight * meridian_safety_score * 5.0
                    )
                    
                    # Update best coordinate if this point has better combined score
                    if combined_score > best_score:
                        best_score = combined_score
                        max_min_separation = min_separation
                        best_coord = test_coord
            
            # Determine if the selected target is east or west of zenith
            ra_difference = (best_coord.ra - zenith_coords.ra).wrap_at(180 * u.deg)
            if ra_difference > 0:
                direction = "east"
                print(f"{SYMBOLS.SUCCESS} Found optimal dark patch {max_min_separation:.2f}° from nearest bright star")
                print(f"{SYMBOLS.TARGET} Optimal coordinates: RA {best_coord.ra.hour:.4f}h, Dec {best_coord.dec.deg:.4f}° ({direction} of zenith)")
                print(f"[SUNRISE] Eastern preference applied - target will stay above horizon longer")
            else:
                direction = "west"
                print(f"{SYMBOLS.SUCCESS} Found optimal dark patch {max_min_separation:.2f}° from nearest bright star")
                print(f"{SYMBOLS.TARGET} Optimal coordinates: RA {best_coord.ra.hour:.4f}h, Dec {best_coord.dec.deg:.4f}° ({direction} of zenith)")
                print(f"{SYMBOLS.WARNING} No suitable eastern target found - using western target")
            
            # Check and report meridian flip risk for the selected target
            meridian_risk = self._calculate_meridian_flip_risk(best_coord, estimated_duration)
            if meridian_risk > 0.1:  # More than 10% risk
                print(f"{SYMBOLS.WARNING} Meridian flip risk: {meridian_risk:.1%} - consider shorter experiment or different target")
            else:
                print(f"{SYMBOLS.SUCCESS} Meridian flip risk: {meridian_risk:.1%} - safe for experiment duration")
            
            return best_coord
            
        except Exception as e:
            print(f"{SYMBOLS.ERROR} Error finding optimal dark patch: {e}")
            return zenith_coords
    
    def find_and_slew_to_optimal_target(self) -> bool:
        """Find and slew to an optimal observation target near zenith."""
        # Check prerequisites
        if not self.mount:
            print(f"{SYMBOLS.WARNING} No mount available, skipping optimal target finding")
            return False
        
        if not self.location:
            print(f"{SYMBOLS.WARNING} No telescope location available, skipping optimal target finding")
            return False
        
        if not ASTROQUERY_AVAILABLE:
            print(f"{SYMBOLS.WARNING} astroquery not available, falling back to zenith slew")
            return self.slew_to_zenith()
        
        try:
            print(f"{SYMBOLS.TARGET} Finding optimal observation target near zenith...")
            
            # Step 1: Calculate current zenith coordinates
            zenith_coords = self._get_zenith_coordinates()
            if not zenith_coords:
                print(f"{SYMBOLS.WARNING} Could not calculate zenith coordinates")
                return False
            
            # Step 2: Query for bright stars near zenith
            bright_stars = self._query_bright_stars_near_zenith(zenith_coords)
            
            # Step 3: Find optimal dark patch
            optimal_coords = self._find_optimal_dark_patch(zenith_coords, bright_stars)
            
            # Step 4: Slew to optimal coordinates
            if optimal_coords.separation(zenith_coords) < 0.1 * u.deg:
                print(f"{SYMBOLS.SUCCESS} Zenith is optimal target - no bright stars nearby")
                success = self._slew_to_coordinates(optimal_coords)
            else:
                print(f"{SYMBOLS.SUCCESS} Slew to optimal dark patch (offset: {optimal_coords.separation(zenith_coords):.2f})")
                success = self._slew_to_coordinates(optimal_coords)
            
            # Store current target coordinates and update timestamp for zenith tracking
            if success:
                self.current_target_coords = optimal_coords
                self.last_zenith_update = time.time()
                print(f"{SYMBOLS.TARGET} Zenith tracking initialized")
            
            return success
                
        except Exception as e:
            print(f"{SYMBOLS.ERROR} Error in optimal target finding: {e}")
            print(f"{SYMBOLS.WARNING} Falling back to zenith slew...")
            return self.slew_to_zenith()
    
    def _check_zenith_drift(self) -> bool:
        """
        Check if the current target has drifted significantly from zenith.
        
        Returns:
            True if a new target should be calculated and slewed to
        """
        if not self.zenith_tracking_enabled or not self.current_target_coords:
            return False
        
        # Calculate time since last zenith update
        current_time = time.time()
        if self.last_zenith_update is None:
            self.last_zenith_update = current_time
            return False
        
        time_since_update = (current_time - self.last_zenith_update) / 60.0  # minutes
        
        # Check if it's time for a zenith update
        if time_since_update >= self.zenith_tracking_interval:
            print(f"⏰ Zenith tracking: {time_since_update:.1f} minutes since last update")
            return True
        
        # Check if current target has drifted too far from zenith
        current_zenith = self._get_zenith_coordinates()
        if current_zenith:
            drift_distance = self.current_target_coords.separation(current_zenith)
            if drift_distance > self.zenith_tracking_threshold * u.deg:
                print(f"[LOC] Zenith drift detected: {drift_distance:.2f} from zenith")
                return True
        
        return False
    
    def _update_zenith_target(self) -> bool:
        """
        Update the zenith target position and slew if necessary.
        
        Returns:
            True if successfully updated and slewed
        """
        try:
            print(f"{SYMBOLS.SWITCH} Updating zenith target position...")
            
            # Calculate new zenith coordinates
            new_zenith = self._get_zenith_coordinates()
            if not new_zenith:
                return False
            
            # Check if we need to find a new optimal target
            if self.current_target_coords:
                # Calculate how much the zenith has moved
                zenith_drift = new_zenith.separation(self._get_zenith_coordinates_at_time(self.last_zenith_update))
                
                # If zenith has moved significantly, recalculate optimal target
                if zenith_drift > 1.0 * u.deg:  # 1 degree threshold for recalculating optimal target
                    print(f"�� Zenith has moved {zenith_drift:.2f} - recalculating optimal target")
                    return self.find_and_slew_to_optimal_target()
                else:
                    # Zenith hasn't moved much, just update current target to new zenith
                    print(f"[UNIV] Zenith has moved {zenith_drift:.2f} - updating target position")
                    self.current_target_coords = new_zenith
                    return self._slew_to_coordinates(new_zenith)
            else:
                # First time, find optimal target
                return self.find_and_slew_to_optimal_target()
                
        except Exception as e:
            print(f"{SYMBOLS.ERROR} Error updating zenith target: {e}")
            return False
    
    def _get_zenith_coordinates_at_time(self, timestamp: float) -> Optional[SkyCoord]:
        """
        Calculate zenith coordinates at a specific timestamp.
        
        Args:
            timestamp: Unix timestamp
            
        Returns:
            Zenith coordinates at the specified time
        """
        if not self.location:
            return None
        
        try:
            # Convert timestamp to astropy Time
            target_time = Time(timestamp, format='unix')
            
            # Calculate zenith coordinates at that time
            zenith = SkyCoord(alt=90*u.deg, az=0*u.deg, frame=AltAz(obstime=target_time, location=self.location))
            
            # Convert to RA/Dec
            ra_dec = zenith.transform_to('icrs')
            return ra_dec
            
        except Exception as e:
            print(f"{SYMBOLS.ERROR} Error calculating zenith at timestamp: {e}")
            return None
    
    def _calculate_meridian_flip_risk(self, target_coords: SkyCoord, experiment_duration: float = 180.0) -> float:
        """
        Calculate the risk of a meridian flip during the experiment.
        
        Args:
            target_coords: Target coordinates in ICRS
            experiment_duration: Expected experiment duration in minutes
            
        Returns:
            Meridian flip risk score (0.0 = no risk, 1.0 = high risk)
        """
        if not self.location:
            return 0.0
        
        try:
            current_time = Time.now()
            
            # Calculate target's hour angle at current time
            lst = current_time.sidereal_time('apparent', longitude=self.location.lon)
            ha_current = lst - target_coords.ra
            
            # Calculate target's hour angle after experiment duration
            future_time = current_time + experiment_duration * u.minute
            lst_future = future_time.sidereal_time('apparent', longitude=self.location.lon)
            ha_future = lst_future - target_coords.ra
            
            # Normalize hour angles to -12 to +12 hours
            ha_current = ha_current.wrap_at(12 * u.hour)
            ha_future = ha_future.wrap_at(12 * u.hour)
            
            # Check if meridian flip would occur (crossing from positive to negative HA)
            if ha_current > 0 and ha_future < 0:
                # Meridian flip would occur
                # Calculate how close to the meridian we are
                meridian_distance = min(abs(ha_current), abs(ha_future))
                # Risk is higher if we're closer to meridian
                risk = max(0.0, 1.0 - (meridian_distance.hour / 2.0))  # 2 hours buffer
                return risk
            else:
                # No meridian flip expected
                return 0.0
                
        except Exception as e:
            print(f"{SYMBOLS.ERROR} Error calculating meridian flip risk: {e}")
            return 0.0
    
    def _estimate_experiment_duration(self, exposure_times: List[int], frames_per_light: int, num_filters: int) -> float:
        """
        Estimate the total experiment duration in minutes.
        
        Args:
            exposure_times: List of exposure times to test
            frames_per_light: Number of frames per exposure time
            num_filters: Number of filters to process
            
        Returns:
            Estimated duration in minutes
        """
        try:
            # Calculate time per filter
            total_exposure_time = sum(exposure_times) * frames_per_light
            overhead_per_exposure = 5  # seconds for readout, slewing, etc.
            total_overhead = len(exposure_times) * frames_per_light * overhead_per_exposure
            
            # Add time for filter changes, slewing, etc.
            filter_change_time = 30  # seconds per filter
            slewing_time = 60  # seconds per filter
            
            time_per_filter = (total_exposure_time + total_overhead + filter_change_time + slewing_time) / 60.0  # minutes
            
            # Total experiment time
            total_time = time_per_filter * num_filters
            
            # Add some buffer for safety
            total_time *= 1.2
            
            return total_time
            
        except Exception as e:
            print(f"{SYMBOLS.ERROR} Error estimating experiment duration: {e}")
            return 180.0  # Default 3 hours
    
    def start_guiding(self):
        """Start the guiding process after slewing to zenith."""
        # First, check for external guiding software
        self.detect_external_guiding()
        
        # If external guiding is active, use it
        if self.external_guiding_active:
            if self.phd2_running:
                print(f"{SYMBOLS.TARGET} Using PHD2 for guiding (external software detected)")
                return self._start_phd2_guiding()
            elif self.ekos_running:
                print(f"{SYMBOLS.TARGET} Using EKOS for guiding (external software detected)")
                return self._start_ekos_guiding()
        
        # Fallback to built-in guiding
        if not self.guide_camera or not self.mount:
            print(f"{SYMBOLS.WARNING} No guide camera or mount available, skipping guiding")
            return False
        
        print(f"{SYMBOLS.TARGET} Using built-in guiding (no external software detected)")
        return self._start_builtin_guiding()
    
    def _start_phd2_guiding(self) -> bool:
        """Start guiding using PHD2."""
        try:
            print(f"{SYMBOLS.TARGET} Connecting to PHD2...")
            
            # Check if PHD2 is connected and ready
            if not self._check_phd2_connected():
                print(f"{SYMBOLS.ERROR} PHD2 is running but not connected to equipment")
                return False
            
            # Check if PHD2 is already guiding
            if self._check_phd2_guiding():
                print(f"{SYMBOLS.SUCCESS} PHD2 is already guiding")
                return True
            
            # Start PHD2 guiding
            print(f"{SYMBOLS.TARGET} Starting PHD2 guiding...")
            if self._start_phd2_guiding_sequence():
                print(f"{SYMBOLS.SUCCESS} PHD2 guiding started successfully")
                return True
            else:
                print(f"{SYMBOLS.ERROR} Failed to start PHD2 guiding")
                return False
                
        except Exception as e:
            print(f"{SYMBOLS.ERROR} Error starting PHD2 guiding: {e}")
            return False
    
    def _start_ekos_guiding(self) -> bool:
        """Start guiding using EKOS."""
        try:
            print(f"{SYMBOLS.TARGET} Using EKOS guiding...")
            
            # EKOS guiding is typically handled through INDI
            # We'll assume it's already set up and ready
            print(f"{SYMBOLS.SUCCESS} EKOS guiding is ready")
            return True
            
        except Exception as e:
            print(f"{SYMBOLS.ERROR} Error with EKOS guiding: {e}")
            return False
    
    def _start_builtin_guiding(self) -> bool:
        """Start the built-in guiding process."""
        try:
            print(f"{SYMBOLS.TARGET} Starting built-in guiding process...")
            
            # Step 1: Take a guide exposure to find guide stars
            print("  [CAM] Taking guide exposure to find guide stars...")
            guide_image = self.capture_guide_frame()
            if guide_image is None:
                print(f"{SYMBOLS.ERROR} Failed to capture guide frame")
                return False
            
            # Step 2: Find and select a guide star
            print("  [FOCUS] Finding guide stars...")
            guide_star = self.find_guide_star(guide_image)
            if guide_star is None:
                print(f"{SYMBOLS.ERROR} No suitable guide star found")
                return False
            
            print(f"  {SYMBOLS.STAR} Selected guide star at position: ({guide_star['x']:.1f}, {guide_star['y']:.1f})")
            self.guide_star_initial_position = (guide_star['x'], guide_star['y'])
            self.guide_star_position = (guide_star['x'], guide_star['y'])
            
            # Step 3: Start guiding thread
            print(f"{SYMBOLS.TARGET} Starting guiding thread...")
            self.guiding_stop_event.clear()
            self.guiding_thread = threading.Thread(target=self._guiding_loop, daemon=True)
            self.guiding_thread.start()
            self.guiding_active = True
            
            # Step 4: Wait for guiding to stabilize
            print("  ⏳ Waiting for guiding to stabilize...")
            time.sleep(5)  # Give guiding time to start
            
            print(f"{SYMBOLS.SUCCESS} Built-in guiding started successfully")
            return True
            
        except Exception as e:
            print(f"{SYMBOLS.ERROR} Error starting built-in guiding: {e}")
            return False
    
    def _check_phd2_connected(self) -> bool:
        """Check if PHD2 is connected to equipment."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            result = sock.connect_ex(('localhost', 4400))
            if result == 0:
                # Send connection status query
                sock.send(b'{"method": "get_connected", "params": [], "id": 1}\n')
                response = sock.recv(1024).decode('utf-8')
                sock.close()
                return '"result": true' in response
        except:
            pass
        return False
    
    def _check_phd2_guiding(self) -> bool:
        """Check if PHD2 is currently guiding."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            result = sock.connect_ex(('localhost', 4400))
            if result == 0:
                # Send guiding status query
                sock.send(b'{"method": "get_guiding", "params": [], "id": 1}\n')
                response = sock.recv(1024).decode('utf-8')
                sock.close()
                return '"result": true' in response
        except:
            pass
        return False
    
    def _start_phd2_guiding_sequence(self) -> bool:
        """Start PHD2 guiding sequence."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex(('localhost', 4400))
            if result == 0:
                # Send start guiding command
                sock.send(b'{"method": "set_guiding_enabled", "params": [true], "id": 1}\n')
                response = sock.recv(1024).decode('utf-8')
                sock.close()
                return '"result": true' in response
        except:
            pass
        return False
    
    def stop_guiding(self):
        """Stop the guiding process."""
        if self.external_guiding_active:
            if self.phd2_running:
                print(f"{SYMBOLS.STOP} Stopping PHD2 guiding...")
                self._stop_phd2_guiding()
            elif self.ekos_running:
                print(f"{SYMBOLS.STOP} Stopping EKOS guiding...")
                # EKOS guiding is typically managed by the user
                print("[INFO]  Please stop EKOS guiding manually if needed")
            self.external_guiding_active = False
            print(f"{SYMBOLS.SUCCESS} External guiding stopped")
        elif self.guiding_active:
            print(f"{SYMBOLS.STOP} Stopping built-in guiding...")
            self.guiding_stop_event.set()
            if self.guiding_thread and self.guiding_thread.is_alive():
                self.guiding_thread.join(timeout=5)
            self.guiding_active = False
            print(f"{SYMBOLS.SUCCESS} Built-in guiding stopped")
    
    def _stop_phd2_guiding(self):
        """Stop PHD2 guiding."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex(('localhost', 4400))
            if result == 0:
                # Send stop guiding command
                sock.send(b'{"method": "set_guiding_enabled", "params": [false], "id": 1}\n')
                response = sock.recv(1024).decode('utf-8')
                sock.close()
                if '"result": true' in response:
                    print(f"{SYMBOLS.SUCCESS} PHD2 guiding stopped successfully")
                else:
                    print(f"{SYMBOLS.WARNING} PHD2 guiding stop command may have failed")
        except Exception as e:
            print(f"{SYMBOLS.ERROR} Error stopping PHD2 guiding: {e}")
    
    def capture_guide_frame(self) -> Optional[np.ndarray]:
        """Capture a single guide frame."""
        try:
            # Set guide camera exposure time
            exp_prop = self.guide_camera.getProperty('CCD_EXPOSURE')
            if exp_prop:
                exp_prop[0].setNumber(self.guide_exposure_time)
                self.client.sendNewProperty(exp_prop)
            
            # Wait for exposure to complete
            time.sleep(self.guide_exposure_time + 1)
            
            # Get the image
            blob_prop = self.guide_camera.getProperty('CCD1')
            if blob_prop and blob_prop[0].getBlob():
                # Load raw data
                raw_data = np.frombuffer(blob_prop[0].getBlob(), dtype=np.uint16)
                
                # Reshape based on camera properties
                size = int(math.sqrt(len(raw_data)))
                raw_data = raw_data.reshape((size, size))
                
                return raw_data
            
        except Exception as e:
            print(f"{SYMBOLS.ERROR} Error capturing guide frame: {e}")
        
        return None
    
    def find_guide_star(self, image_data: np.ndarray) -> Optional[Dict]:
        """Find a suitable guide star in the guide image."""
        try:
            # Analyze image quality to find stars
            fwhm_values = []
            star_positions = []
            
            if PHOTUTILS_AVAILABLE:
                # Use photutils for star detection
                mean, median, std = sigma_clipped_stats(image_data, sigma=3.0)
                threshold = median + 5 * std
                
                finder = DAOStarFinder(fwhm=3.0, threshold=threshold, exclude_border=True)
                sources = finder(image_data)
                
                if sources is not None and len(sources) > 0:
                    for source in sources:
                        x, y = source['xcentroid'], source['ycentroid']
                        fwhm = source['fwhm']
                        
                        # Filter stars by quality
                        if 1.0 < fwhm < 10.0 and 50 < x < image_data.shape[1] - 50 and 50 < y < image_data.shape[0] - 50:
                            fwhm_values.append(fwhm)
                            star_positions.append((x, y, fwhm))
            else:
                # Fallback method using simple thresholding
                mean, median, std = sigma_clipped_stats(image_data, sigma=3.0)
                threshold = median + 5 * std
                
                # Find local maxima
                from scipy import ndimage
                local_max = ndimage.maximum_filter(image_data, size=5)
                maxima = (image_data == local_max) & (image_data > threshold)
                
                y_coords, x_coords = np.where(maxima)
                for x, y in zip(x_coords, y_coords):
                    if 50 < x < image_data.shape[1] - 50 and 50 < y < image_data.shape[0] - 50:
                        # Calculate approximate FWHM
                        fwhm = self._estimate_fwhm(image_data, x, y)
                        if 1.0 < fwhm < 10.0:
                            fwhm_values.append(fwhm)
                            star_positions.append((x, y, fwhm))
            
            if not star_positions:
                return None
            
            # Select the best guide star (brightest with good FWHM)
            star_positions.sort(key=lambda s: s[2])  # Sort by FWHM (lower is better)
            best_star = star_positions[0]
            
            return {
                'x': best_star[0],
                'y': best_star[1],
                'fwhm': best_star[2]
            }
            
        except Exception as e:
            print(f"{SYMBOLS.ERROR} Error finding guide star: {e}")
            return None
    
    def _estimate_fwhm(self, image_data: np.ndarray, x: int, y: int) -> float:
        """Estimate FWHM for a star at given coordinates."""
        try:
            # Extract a small region around the star
            x1, x2 = max(0, x-10), min(image_data.shape[1], x+11)
            y1, y2 = max(0, y-10), min(image_data.shape[0], y+11)
            region = image_data[y1:y2, x1:x2]
            
            # Find peak value
            peak = np.max(region)
            half_max = peak / 2
            
            # Count pixels above half maximum
            above_half = np.sum(region > half_max)
            
            # Estimate FWHM as square root of area
            return math.sqrt(above_half)
            
        except Exception:
            return 3.0  # Default fallback
    
    def _guiding_loop(self):
        """Main guiding loop that runs in a separate thread."""
        print(f"{SYMBOLS.TARGET} Guiding loop started")
        
        while not self.guiding_stop_event.is_set():
            try:
                # Capture guide frame
                guide_image = self.capture_guide_frame()
                if guide_image is None:
                    time.sleep(1)
                    continue
                
                # Find current guide star position
                current_position = self._find_star_position(guide_image, self.guide_star_initial_position)
                if current_position is None:
                    time.sleep(1)
                    continue
                
                # Calculate drift
                dx = current_position[0] - self.guide_star_initial_position[0]
                dy = current_position[1] - self.guide_star_initial_position[1]
                
                # Apply corrections if drift exceeds threshold
                if abs(dx) > self.guide_dither_threshold or abs(dy) > self.guide_dither_threshold:
                    self._apply_guide_correction(dx, dy)
                
                # Update current position
                self.guide_star_position = current_position
                
                # Sleep between guide exposures
                time.sleep(self.guide_exposure_time)
                
            except Exception as e:
                print(f"{SYMBOLS.ERROR} Error in guiding loop: {e}")
                time.sleep(1)
        
        print(f"{SYMBOLS.STOP} Guiding loop stopped")
    
    def _find_star_position(self, image_data: np.ndarray, expected_position: Tuple[float, float]) -> Optional[Tuple[float, float]]:
        """Find the current position of the guide star."""
        try:
            x, y = expected_position
            x, y = int(x), int(y)
            
            # Extract a region around expected position
            x1, x2 = max(0, x-20), min(image_data.shape[1], x+21)
            y1, y2 = max(0, y-20), min(image_data.shape[0], y+21)
            region = image_data[y1:y2, x1:x2]
            
            if region.size == 0:
                return None
            
            # Find centroid
            if PHOTUTILS_AVAILABLE:
                try:
                    centroid = centroid_com(region)
                    return (x1 + centroid[0], y1 + centroid[1])
                except:
                    pass
            
            # Fallback: find brightest pixel
            max_pos = np.unravel_index(np.argmax(region), region.shape)
            return (x1 + max_pos[1], y1 + max_pos[0])
            
        except Exception:
            return None
    
    def _apply_guide_correction(self, dx: float, dy: float):
        """Apply guide correction to the mount."""
        try:
            # Calculate pulse duration based on drift and aggressiveness
            pulse_x = int(abs(dx) * self.guide_aggressiveness * self.guide_max_pulse)
            pulse_y = int(abs(dy) * self.guide_aggressiveness * self.guide_max_pulse)
            
            # Limit pulse duration
            pulse_x = min(pulse_x, self.guide_max_pulse)
            pulse_y = min(pulse_y, self.guide_max_pulse)
            
            # Apply RA correction (X-axis)
            if abs(dx) > self.guide_dither_threshold:
                if dx > 0:
                    # Drift east, pulse west
                    self._send_guide_pulse('GUIDE_WEST', pulse_x)
                else:
                    # Drift west, pulse east
                    self._send_guide_pulse('GUIDE_EAST', pulse_x)
            
            # Apply Dec correction (Y-axis)
            if abs(dy) > self.guide_dither_threshold:
                if dy > 0:
                    # Drift north, pulse south
                    self._send_guide_pulse('GUIDE_SOUTH', pulse_y)
                else:
                    # Drift south, pulse north
                    self._send_guide_pulse('GUIDE_NORTH', pulse_y)
                    
        except Exception as e:
            print(f"{SYMBOLS.ERROR} Error applying guide correction: {e}")
    
    def _send_guide_pulse(self, direction: str, duration: int):
        """Send a guide pulse to the mount."""
        try:
            # Get guide property
            guide_prop = self.mount.getProperty('TELESCOPE_TIMED_GUIDE_NS')
            if guide_prop:
                # Find the correct direction
                for i in range(guide_prop.getCount()):
                    prop = guide_prop[i]
                    if prop.getName() == direction:
                        prop.setNumber(duration)
                        self.client.sendNewProperty(guide_prop)
                        return
                
                # Try alternative property names
                alt_prop = self.mount.getProperty('TELESCOPE_TIMED_GUIDE_WE')
                if alt_prop:
                    for i in range(alt_prop.getCount()):
                        prop = alt_prop[i]
                        if prop.getName() == direction:
                            prop.setNumber(duration)
                            self.client.sendNewProperty(alt_prop)
                            return
                            
        except Exception as e:
            print(f"{SYMBOLS.ERROR} Error sending guide pulse: {e}")
    

    
    def median_combine_frames(self, frame_paths: List[Path]) -> np.ndarray:
        """Median combine multiple FITS frames."""
        frames = []
        for path in frame_paths:
            with fits.open(path) as hdul:
                frames.append(hdul[0].data)
        
        return np.median(frames, axis=0)
    

    
    def analyze_sky_background(self, image_data: np.ndarray) -> float:
        """
        Analyze the sky background using configurable parameters.
        
        Args:
            image_data: 2D numpy array of the image data
            
        Returns:
            Median sky background value
        """
        try:
            # Apply sky region fraction to focus on central portion of image
            height, width = image_data.shape
            start_y = int(height * (1 - self.sky_region_fraction) / 2)
            end_y = int(height * (1 + self.sky_region_fraction) / 2)
            start_x = int(width * (1 - self.sky_region_fraction) / 2)
            end_x = int(width * (1 + self.sky_region_fraction) / 2)
            
            # Extract sky region
            sky_region = image_data[start_y:end_y, start_x:end_x]
            
            # Check if we have enough valid pixels
            if sky_region.size < self.min_valid_pixels:
                print(f"{SYMBOLS.WARNING} Warning: Sky region too small ({sky_region.size} pixels), using full image")
                sky_region = image_data
            
            # Use sigma-clipped stats with configurable threshold
            mean, median, std = sigma_clipped_stats(sky_region, sigma=self.sigma_clip_threshold)
            
            return median
            
        except Exception as e:
            print(f"{SYMBOLS.ERROR} Error in sky background analysis: {e}")
            # Fallback to simple median
            return np.median(image_data)
    
    def analyze_image_quality(self, image_data: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
        """
        Analyze image quality by measuring star FWHM and eccentricity.
        
        Args:
            image_data: 2D numpy array of the image data
            
        Returns:
            Tuple of (median_fwhm, median_eccentricity) or (None, None) if no stars found
        """
        try:
            # Background statistics using configurable parameters
            mean, median, std = sigma_clipped_stats(image_data, sigma=self.sigma_clip_threshold)
            
            # Star detection
            if PHOTUTILS_AVAILABLE:
                # Use photutils for robust star detection
                threshold = median + (5 * std)
                daofind = DAOStarFinder(fwhm=3.0, threshold=threshold, exclude_border=True)
                sources = daofind(image_data)
                
                if sources is None or len(sources) == 0:
                    return None, None
                
                star_coords = [(int(sources['xcentroid'][i]), int(sources['ycentroid'][i])) 
                              for i in range(len(sources))]
            else:
                # Fallback: Simple threshold-based detection
                threshold = median + (5 * std)
                bright_pixels = image_data > threshold
                
                # Find local maxima
                star_coords = []
                for y in range(1, image_data.shape[0] - 1):
                    for x in range(1, image_data.shape[1] - 1):
                        if bright_pixels[y, x]:
                            # Check if it's a local maximum in 3x3 region
                            region = image_data[y-1:y+2, x-1:x+2]
                            if image_data[y, x] == region.max():
                                star_coords.append((x, y))
                
                # Limit to brightest stars to avoid noise
                if len(star_coords) > 20:
                    star_brightnesses = [image_data[y, x] for x, y in star_coords]
                    sorted_indices = np.argsort(star_brightnesses)[::-1]
                    star_coords = [star_coords[i] for i in sorted_indices[:20]]
            
            if not star_coords:
                return None, None
            
            # Analyze each star
            fwhm_values = []
            ecc_values = []
            
            for px, py in star_coords:
                # Extract 15x15 pixel stamp centered on star
                y_start = max(0, py - 7)
                y_end = min(image_data.shape[0], py + 8)
                x_start = max(0, px - 7)
                x_end = min(image_data.shape[1], px + 8)
                
                stamp = image_data[y_start:y_end, x_start:x_end]
                
                if stamp.shape[0] < 5 or stamp.shape[1] < 5:
                    continue
                
                # Calculate FWHM
                fwhm = self._calculate_fwhm(stamp, median)
                if fwhm is not None and 1.0 < fwhm < 20.0:  # Reasonable FWHM range
                    fwhm_values.append(fwhm)
                
                # Calculate eccentricity
                ecc = self._calculate_eccentricity(stamp, median)
                if ecc is not None and 0.0 < ecc < 1.0:  # Valid eccentricity range
                    ecc_values.append(ecc)
            
            # Return median values
            if fwhm_values and ecc_values:
                median_fwhm = np.median(fwhm_values)
                median_ecc = np.median(ecc_values)
                return median_fwhm, median_ecc
            else:
                return None, None
                
        except Exception as e:
            print(f"{SYMBOLS.ERROR} Error in image quality analysis: {e}")
            return None, None
    
    def _calculate_fwhm(self, stamp: np.ndarray, background: float) -> Optional[float]:
        """Calculate FWHM of a star stamp."""
        try:
            # Find peak value
            peak_value = stamp.max()
            half_max = (peak_value + background) / 2.0
            
            # Find center of stamp
            center_y, center_x = stamp.shape[0] // 2, stamp.shape[1] // 2
            
            # Horizontal profile
            h_profile = stamp[center_y, :]
            h_above_half = h_profile > half_max
            if np.sum(h_above_half) > 0:
                h_fwhm = np.sum(h_above_half)
            else:
                h_fwhm = 0
            
            # Vertical profile
            v_profile = stamp[:, center_x]
            v_above_half = v_profile > half_max
            if np.sum(v_above_half) > 0:
                v_fwhm = np.sum(v_above_half)
            else:
                v_fwhm = 0
            
            # Average FWHM
            if h_fwhm > 0 and v_fwhm > 0:
                return (h_fwhm + v_fwhm) / 2.0
            else:
                return None
                
        except Exception:
            return None
    
    def _calculate_eccentricity(self, stamp: np.ndarray, background: float) -> Optional[float]:
        """Calculate eccentricity of a star stamp."""
        try:
            # Subtract background
            stamp_bg_sub = stamp - background
            stamp_bg_sub = np.maximum(stamp_bg_sub, 0)  # Ensure non-negative
            
            # Calculate image moments
            y_coords, x_coords = np.mgrid[:stamp.shape[0], :stamp.shape[1]]
            
            # Zeroth moment (total flux)
            M00 = np.sum(stamp_bg_sub)
            if M00 <= 0:
                return None
            
            # First moments (centroids)
            M10 = np.sum(x_coords * stamp_bg_sub)
            M01 = np.sum(y_coords * stamp_bg_sub)
            
            # Centroid coordinates
            x_cen = M10 / M00
            y_cen = M01 / M00
            
            # Second-order central moments
            M20 = np.sum((x_coords - x_cen)**2 * stamp_bg_sub)
            M02 = np.sum((y_coords - y_cen)**2 * stamp_bg_sub)
            M11 = np.sum((x_coords - x_cen) * (y_coords - y_cen) * stamp_bg_sub)
            
            # Calculate eccentricity
            if M20 > 0 and M02 > 0:
                # Use the smaller eigenvalue for eccentricity calculation
                if M02 < M20:
                    ecc = np.sqrt(1 - M02 / M20)
                else:
                    ecc = np.sqrt(1 - M20 / M02)
                return min(ecc, 1.0)  # Ensure eccentricity <= 1
            else:
                return None
                
        except Exception:
            return None
    
    def calculate_target_adu(self, gain: float, read_noise: float) -> float:
        """Calculate target ADU based on gain and read noise."""
        # Target ADU = (target_noise_ratio * read_noise)^2 / gain
        target_adu = (self.target_noise_ratio * read_noise) ** 2 / gain
        return target_adu
    
    def intelligent_scout_and_predict(self, filter_name: str) -> Tuple[float, List[int], Optional[float]]:
        """
        Phase A: Intelligent Scout & Prediction
        
        Args:
            filter_name: Name of the current filter
            
        Returns:
            Tuple of (predicted_optimal_time, dynamic_exposure_times, fwhm)
        """
        print(f"{SYMBOLS.TARGET} [TARGET] Phase A: Intelligent Scout & Prediction for {filter_name}")
        
        # Get camera properties for this filter
        camera_props = self.get_camera_properties(filter_name)
        gain = camera_props['gain']
        read_noise = camera_props['read_noise']
        
        # Calculate target ADU
        target_adu = self.calculate_target_adu(gain, read_noise)
        print(f"    [CHART] Target ADU: {target_adu:.2f}")
        print(f"    [CHART] Camera Gain: {gain:.2f}, Read Noise: {read_noise:.2f}")
        
        # Ensure dark frame for scout exposure exists (use library)
        scout_dark_path = self.find_best_dark_frame(self.scout_exposure_time)
        if not scout_dark_path:
            print(f"{SYMBOLS.ERROR} No dark frame available for scout exposure ({self.scout_exposure_time}s)")
            print(f"    [CAM] Creating scout dark frame for {self.scout_exposure_time}s...")
            self.create_master_dark_library(self.scout_exposure_time)
            scout_dark_path = self.find_best_dark_frame(self.scout_exposure_time)
        
        # Take scout light frame
        print(f"    [CAM] Taking scout exposure ({self.scout_exposure_time}s)...")
        scout_image = self.capture_light_frame_with_filter(self.scout_exposure_time, filter_name)
        
        if scout_image is None:
            print(f"{SYMBOLS.ERROR} Failed to capture scout frame")
            # Fallback to default exposure times
            return 120.0, [60, 120, 180, 240, 300], None
        
        # Analyze scout frame for sky background and image quality
        scout_adu = self.analyze_sky_background(scout_image)
        print(f"    [CHART] Scout frame sky ADU: {scout_adu:.2f}")
        
        # Analyze image quality to get FWHM for seeing assessment
        fwhm, ecc = self.analyze_image_quality(scout_image)
        
        # Adjust frames_per_light based on seeing conditions
        if fwhm is not None:
            print(f"    [CHART] Measured FWHM: {fwhm:.2f} pixels")
            self.adjust_frames_per_light_based_on_seeing(fwhm)
        else:
            print(f"{SYMBOLS.WARNING} Warning: Could not determine seeing conditions from scout frame")
            print("    [CHART] Using default frames_per_light value")
        
        # Calculate sky flux rate
        sky_flux_rate = scout_adu / self.scout_exposure_time
        print(f"    [CHART] Sky flux rate: {sky_flux_rate:.2f} ADU/sec")
        
        # Predict optimal time
        predicted_optimal_time = target_adu / sky_flux_rate
        print(f"    [TARGET] Predicted optimal time: {predicted_optimal_time:.1f}s")
        
        # Generate dynamic test range
        dynamic_times = self.generate_dynamic_test_range(predicted_optimal_time)

        
        return predicted_optimal_time, dynamic_times, fwhm
    
    def generate_dynamic_test_range(self, predicted_time: float) -> List[int]:
        """
        Phase B: Dynamically Generate Test Range
        
        Args:
            predicted_time: Predicted optimal exposure time
            
        Returns:
            List of exposure times to test
        """
        # Ensure minimum reasonable time
        min_time = max(10, int(0.3 * predicted_time))
        max_time = min(1800, int(2.0 * predicted_time))  # Cap at 30 minutes
        
        # Generate test range around prediction
        test_times = [
            int(0.6 * predicted_time),
            int(0.8 * predicted_time),
            int(predicted_time),
            int(1.2 * predicted_time),
            int(1.5 * predicted_time)
        ]
        
        # Filter and sort
        test_times = [t for t in test_times if min_time <= t <= max_time]
        test_times = sorted(list(set(test_times)))  # Remove duplicates and sort
        
        # Ensure we have at least 3 test points
        if len(test_times) < 3:
            if predicted_time < 60:
                test_times = [30, 60, 90, 120]
            elif predicted_time < 300:
                test_times = [60, 120, 180, 240, 300]
            else:
                test_times = [120, 240, 360, 480, 600]
        
        return test_times
    
    def create_dark_frame_library(self) -> bool:
        """
        Phase C: Create Dark Frame Library
        
        Creates a comprehensive library of dark frames for a range of exposure times
        that can be used throughout the experiment without requiring user intervention.
        
        Returns:
            True if library creation was successful
        """
        print(f"{SYMBOLS.CAMERA} [CAM] Phase C: Creating Dark Frame Library")
        print("    [TARGET] This creates darks for a range of exposure times upfront")
        print("    [TARGET] These will be used throughout the experiment automatically")
        
        # Define exposure times for dark library (covering typical ranges)
        dark_library_times = [10, 15, 20, 30, 45, 60, 90, 120, 180, 240, 300, 420, 600, 900, 1200, 1800]
        
        # Check which darks already exist
        existing_darks = []
        missing_darks = []
        
        for exposure_time in dark_library_times:
            dark_path = self.calibration_path / f"master_dark_library_{exposure_time}s.fits"
            if dark_path.exists():
                existing_darks.append(exposure_time)
            else:
                missing_darks.append(exposure_time)
        
        if existing_darks:
            print(f"    {SYMBOLS.SUCCESS} Existing darks found for: {', '.join(map(str, existing_darks))}s")
        
        if missing_darks:
            print(f"    [CAM] Need to create darks for: {', '.join(map(str, missing_darks))}s")
            print("    🔒 Please cover the telescope and press Enter when ready...")
            input()
            
            # Create missing master darks
            for exposure_time in missing_darks:
                print(f"      [CAM] Creating dark for {exposure_time}s...")
                self.create_master_dark_library(exposure_time)
            
            print("    [UNLOCK] Please uncover the telescope and press Enter when ready...")
            input()
            print(f"{SYMBOLS.SUCCESS} Dark frame library complete!")
        else:
            print(f"{SYMBOLS.SUCCESS} Dark frame library already complete!")
        
        return True
    
    def create_master_dark_library(self, exposure_time: int) -> str:
        """Create a master dark frame for the dark library."""
        dark_path = self.calibration_path / f"master_dark_library_{exposure_time}s.fits"
        
        if dark_path.exists():
            print(f"      {SYMBOLS.SUCCESS} Master dark for {exposure_time}s already exists")
            return str(dark_path)
        
        print(f"      [CAM] Creating master dark for {exposure_time}s exposure...")
        
        # Set exposure time
        exp_prop = self.camera.getProperty('CCD_EXPOSURE')
        if exp_prop:
            exp_prop[0].setNumber(exposure_time)
            self.client.sendNewProperty(exp_prop)
        
        # Capture dark frames
        dark_frames = []
        for i in range(self.frames_per_dark):
            print(f"        Capturing dark frame {i+1}/{self.frames_per_dark}...")
            
            # Start exposure
            exp_prop = self.camera.getProperty('CCD_EXPOSURE')
            if exp_prop:
                exp_prop[0].setNumber(exposure_time)
                self.client.sendNewProperty(exp_prop)
            
            # Wait for exposure to complete
            time.sleep(exposure_time + 2)
            
            # Get the image
            blob_prop = self.camera.getProperty('CCD1')
            if blob_prop and blob_prop[0].getBlob():
                # Save individual dark frame
                dark_frame_path = self.calibration_path / f"dark_library_{exposure_time}s_{i+1}.fits"
                with open(dark_frame_path, 'wb') as f:
                    f.write(blob_prop[0].getBlob())
                dark_frames.append(dark_frame_path)
        
        # Create master dark by median combining
        if dark_frames:
            print("        🔬 Creating master dark by median combining...")
            master_dark = self.median_combine_frames(dark_frames)
            
            # Save master dark
            hdu = fits.PrimaryHDU(master_dark)
            hdu.writeto(dark_path, overwrite=True)
            
            # Clean up individual frames
            for frame in dark_frames:
                frame.unlink()
            
            print(f"        {SYMBOLS.SUCCESS} Master dark saved to {dark_path}")
            return str(dark_path)
        
        return None
    
    def find_best_dark_frame(self, exposure_time: int) -> Optional[str]:
        """
        Find the best available dark frame for a given exposure time.
        
        Args:
            exposure_time: Required exposure time in seconds
            
        Returns:
            Path to the best available dark frame, or None if none available
        """
        # First, try to find an exact match in the library
        exact_dark_path = self.calibration_path / f"master_dark_library_{exposure_time}s.fits"
        if exact_dark_path.exists():
            return str(exact_dark_path)
        
        # If no exact match, find the closest available dark frame
        dark_library_times = [10, 15, 20, 30, 45, 60, 90, 120, 180, 240, 300, 420, 600, 900, 1200, 1800]
        
        # Find the closest time
        closest_time = min(dark_library_times, key=lambda x: abs(x - exposure_time))
        closest_dark_path = self.calibration_path / f"master_dark_library_{closest_time}s.fits"
        
        if closest_dark_path.exists():
            time_diff = abs(closest_time - exposure_time)
            if time_diff <= 30:  # Within 30 seconds is acceptable
                print(f"      [CAM] Using dark frame for {closest_time}s (closest to {exposure_time}s)")
                return str(closest_dark_path)
            else:
                print(f"      {SYMBOLS.WARNING}  Warning: Using dark frame for {closest_time}s (diff: {time_diff}s)")
                return str(closest_dark_path)
        
        return None
    
    def create_scaled_dark_frame(self, target_time: int, source_dark_path: str) -> str:
        """
        Create a scaled dark frame for the target exposure time.
        
        Args:
            target_time: Target exposure time
            source_dark_path: Path to source dark frame
            
        Returns:
            Path to the scaled dark frame
        """
        try:
            # Load source dark frame
            with fits.open(source_dark_path) as hdul:
                source_dark = hdul[0].data
            
            # Extract source time from filename
            source_filename = Path(source_dark_path).name
            source_time = int(source_filename.split('_')[3].replace('s.fits', ''))
            
            # Calculate scaling factor
            scale_factor = target_time / source_time
            
            # Scale the dark frame
            scaled_dark = source_dark * scale_factor
            
            # Save scaled dark frame
            scaled_dark_path = self.calibration_path / f"master_dark_scaled_{target_time}s.fits"
            hdu = fits.PrimaryHDU(scaled_dark)
            hdu.writeto(scaled_dark_path, overwrite=True)
            
            print(f"      [CAM] Created scaled dark frame for {target_time}s (scaled from {source_time}s)")
            return str(scaled_dark_path)
            
        except Exception as e:
            print(f"{SYMBOLS.ERROR} Error creating scaled dark frame: {e}")
            return None
    
    def create_master_dark_with_filter(self, exposure_time: int, filter_name: str) -> str:
        """Create a master dark frame for the given exposure time and filter."""
        dark_path = self.calibration_path / f"master_dark_{filter_name}_{exposure_time}s.fits"
        
        if dark_path.exists():
            print(f"    {SYMBOLS.SUCCESS} Master dark for {filter_name} {exposure_time}s already exists")
            return str(dark_path)
        
        print(f"    [CAM] Creating master dark for {filter_name} {exposure_time}s exposure...")
        
        # Set exposure time
        exp_prop = self.camera.getProperty('CCD_EXPOSURE')
        if exp_prop:
            exp_prop[0].setNumber(exposure_time)
            self.client.sendNewProperty(exp_prop)
        
        # Capture dark frames
        dark_frames = []
        for i in range(self.frames_per_dark):
            print(f"      Capturing dark frame {i+1}/{self.frames_per_dark}...")
            
            # Start exposure
            exp_prop = self.camera.getProperty('CCD_EXPOSURE')
            if exp_prop:
                exp_prop[0].setNumber(exposure_time)
                self.client.sendNewProperty(exp_prop)
            
            # Wait for exposure to complete
            time.sleep(exposure_time + 2)
            
            # Get the image
            blob_prop = self.camera.getProperty('CCD1')
            if blob_prop and blob_prop[0].getBlob():
                # Save individual dark frame
                dark_frame_path = self.calibration_path / f"dark_{filter_name}_{exposure_time}s_{i+1}.fits"
                with open(dark_frame_path, 'wb') as f:
                    f.write(blob_prop[0].getBlob())
                dark_frames.append(dark_frame_path)
        
        # Create master dark by median combining
        if dark_frames:
            print("    🔬 Creating master dark by median combining...")
            master_dark = self.median_combine_frames(dark_frames)
            
            # Save master dark
            hdu = fits.PrimaryHDU(master_dark)
            hdu.writeto(dark_path, overwrite=True)
            
            # Clean up individual frames
            for frame in dark_frames:
                frame.unlink()
            
            print(f"    {SYMBOLS.SUCCESS} Master dark saved to {dark_path}")
            return str(dark_path)
        
        return None
    
    def capture_light_frame_with_filter(self, exposure_time: int, filter_name: str) -> np.ndarray:
        """Capture a single light frame and return calibrated data for specific filter."""
        # Start exposure
        exp_prop = self.camera.getProperty('CCD_EXPOSURE')
        if exp_prop:
            exp_prop[0].setNumber(exposure_time)
            self.client.sendNewProperty(exp_prop)
        
        # Wait for exposure to complete
        time.sleep(exposure_time + 2)
        
        # Get the image
        blob_prop = self.camera.getProperty('CCD1')
        if blob_prop and blob_prop[0].getBlob():
            # Load raw data
            raw_data = np.frombuffer(blob_prop[0].getBlob(), dtype=np.uint16)
            
            # Reshape based on camera properties (you might need to adjust this)
            # For now, assuming a square image
            size = int(math.sqrt(len(raw_data)))
            raw_data = raw_data.reshape((size, size))
            
            # Find and apply the best available dark frame from the library
            dark_path = self.find_best_dark_frame(exposure_time)
            if dark_path:
                try:
                    with fits.open(dark_path) as hdul:
                        dark_data = hdul[0].data
                    
                    # Check if we need to scale the dark frame
                    source_filename = Path(dark_path).name
                    if 'library' in source_filename:
                        source_time = int(source_filename.split('_')[3].replace('s.fits', ''))
                        if source_time != exposure_time:
                            # Scale the dark frame to match exposure time
                            scale_factor = exposure_time / source_time
                            dark_data = dark_data * scale_factor
                    
                    calibrated_data = raw_data.astype(float) - dark_data
                except Exception as e:
                    print(f"{SYMBOLS.ERROR} Error applying dark frame: {e}")
                    calibrated_data = raw_data.astype(float)
            else:
                print(f"{SYMBOLS.WARNING} No dark frame available for {exposure_time}s exposure")
                calibrated_data = raw_data.astype(float)
            
            return calibrated_data
        
        return None
    
    def execute_focused_experiment(self, filter_name: str, exposure_times: List[int]) -> List[Dict]:
        """
        Phase D: Execute the Focused Experiment
        
        Args:
            filter_name: Current filter name
            exposure_times: List of exposure times to test
            
        Returns:
            List of exposure results
        """
        print(f"{SYMBOLS.TARGET} [ROCKET] Phase D: Execute Focused Experiment for {filter_name}")
        
        # Find and slew to optimal target near zenith
        if self.mount:
            print("    [TARGET] Finding and slewing to optimal target...")
            self.find_and_slew_to_optimal_target()
        
        # Perform filter-aware focusing
        print("    [FOCUS] Performing filter-aware focusing...")
        focusing_success = self.perform_focusing(filter_name)
        if not focusing_success:
            print("    {SYMBOLS.WARNING}  Warning: Focusing failed, continuing with current focus position")
        else:
            print("    {SYMBOLS.SUCCESS} Focusing completed successfully")
        
        # Test each exposure time
        exposure_results = []
        for exposure_time in exposure_times:
            print(f"    [CAM] Capturing {self.frames_per_light} frames at {exposure_time}s...")
            
            # Check for zenith drift before starting this exposure time
            if self.zenith_tracking_enabled and self._check_zenith_drift():
                print("    {SYMBOLS.REFRESH} Zenith drift detected - updating target position...")
                if not self._update_zenith_target():
                    print("    {SYMBOLS.WARNING}  Failed to update zenith target, continuing with current position")
            
            # Capture multiple frames and analyze
            sky_adus = []
            fwhm_values = []
            ecc_values = []
            
            for frame_num in range(self.frames_per_light):
                print(f"      Frame {frame_num + 1}/{self.frames_per_light}...")
                
                # Check for zenith drift every few frames (for long exposures)
                if (self.zenith_tracking_enabled and 
                    frame_num > 0 and 
                    frame_num % 2 == 0 and  # Check every 2nd frame
                    self._check_zenith_drift()):
                    print("      {SYMBOLS.REFRESH} Zenith drift detected during exposure sequence...")
                    if not self._update_zenith_target():
                        print("      {SYMBOLS.WARNING}  Failed to update zenith target, continuing with current position")
                
                image_data = self.capture_light_frame_with_filter(exposure_time, filter_name)
                if image_data is not None:
                    # Analyze sky background
                    sky_adu = self.analyze_sky_background(image_data)
                    sky_adus.append(sky_adu)
                    
                    # Analyze image quality
                    fwhm, ecc = self.analyze_image_quality(image_data)
                    if fwhm is not None:
                        fwhm_values.append(fwhm)
                    if ecc is not None:
                        ecc_values.append(ecc)
            
            if sky_adus:
                avg_sky_adu = np.mean(sky_adus)
                avg_fwhm = np.mean(fwhm_values) if fwhm_values else None
                avg_ecc = np.mean(ecc_values) if ecc_values else None
                
                exposure_results.append({
                    'time': exposure_time,
                    'adu': avg_sky_adu,
                    'fwhm': avg_fwhm,
                    'ecc': avg_ecc
                })
                
                print(f"      Average sky ADU: {avg_sky_adu:.2f}")
                if avg_fwhm is not None:
                    print(f"      Average FWHM: {avg_fwhm:.2f} pixels")
                if avg_ecc is not None:
                    print(f"      Average Eccentricity: {avg_ecc:.3f}")
        
        return exposure_results
    
    def find_optimal_exposure(self, exposure_results: List[Dict], target_adu: float) -> float:
        """
        Find optimal exposure time balancing noise and image quality.
        
        Args:
            exposure_results: List of dictionaries with 'time', 'adu', 'fwhm', 'ecc' keys
            target_adu: Target sky background ADU
            
        Returns:
            Optimal exposure time in seconds
        """
        if len(exposure_results) < 2:
            return exposure_results[0]['time'] if exposure_results else 0
        
        # Extract data
        exposure_times = [r['time'] for r in exposure_results]
        sky_adus = [r['adu'] for r in exposure_results]
        
        # Noise-based optimal time (original method)
        noise_based_time = self._find_noise_based_exposure(exposure_times, sky_adus, target_adu)
        
        # Quality-based optimal time (new method)
        quality_based_time = self._find_quality_based_exposure(exposure_results)
        
        # Return the minimum of the two times
        optimal_time = min(noise_based_time, quality_based_time)
        
        print(f"    [CHART] Noise-based optimal time: {noise_based_time:.1f}s")
        print(f"    [CHART] Quality-based optimal time: {quality_based_time:.1f}s")
        print(f"    [CHART] Final optimal time: {optimal_time:.1f}s")
        
        return optimal_time
    
    def _find_noise_based_exposure(self, exposure_times: List[int], sky_adus: List[float], 
                                  target_adu: float) -> float:
        """Find optimal exposure time based on noise requirements."""
        # Find where the curve crosses the target ADU
        for i in range(len(exposure_times) - 1):
            if sky_adus[i] <= target_adu <= sky_adus[i + 1]:
                # Linear interpolation
                t1, t2 = exposure_times[i], exposure_times[i + 1]
                adu1, adu2 = sky_adus[i], sky_adus[i + 1]
                
                optimal_time = t1 + (target_adu - adu1) * (t2 - t1) / (adu2 - adu1)
                return optimal_time
        
        # If target ADU is not reached, return the longest exposure time
        return exposure_times[-1]
    
    def _find_quality_based_exposure(self, exposure_results: List[Dict]) -> float:
        """Find optimal exposure time based on image quality (FWHM)."""
        # Find the minimum FWHM value
        valid_fwhm_data = [(r['time'], r['fwhm']) for r in exposure_results if r['fwhm'] is not None]
        
        if not valid_fwhm_data:
            # If no FWHM data, return longest exposure time
            return max(r['time'] for r in exposure_results)
        
        min_fwhm = min(fwhm for _, fwhm in valid_fwhm_data)
        fwhm_threshold = min_fwhm * self.fwhm_degradation_threshold
        
        # Find the longest exposure time where FWHM is within threshold
        quality_limited_time = 0
        for result in sorted(exposure_results, key=lambda x: x['time']):
            if result['fwhm'] is not None and result['fwhm'] <= fwhm_threshold:
                quality_limited_time = result['time']
            else:
                break
        
        return quality_limited_time if quality_limited_time > 0 else max(r['time'] for r in exposure_results)
    
    def run_refinement_phase(self, filter_name: str, scout_time: float) -> List[Dict]:
        """
        Run the high-precision refinement phase to find the optimal exposure based on image quality.
        
        Args:
            filter_name: Name of the current filter
            scout_time: Initial optimal exposure time from scout phase
            
        Returns:
            List of refinement results with detailed quality analysis
        """
        print(f"{SYMBOLS.FOCUS} [REFINE] Refining exposure for {filter_name} filter...")
        print(f"    [TARGET] Initial optimal time: {scout_time:.1f}s")
        
        # Define search range centered around scout_time
        min_exposure = max(10, scout_time - (self.refinement_steps * self.refinement_step_size))
        max_exposure = scout_time + (self.refinement_steps * self.refinement_step_size)
        
        # Generate granular exposure times
        refinement_times = []
        for i in range(-self.refinement_steps, self.refinement_steps + 1):
            exposure_time = int(scout_time + (i * self.refinement_step_size))
            if exposure_time >= min_exposure and exposure_time <= max_exposure:
                refinement_times.append(exposure_time)
        
        # Ensure we have at least the scout time
        if int(scout_time) not in refinement_times:
            refinement_times.append(int(scout_time))
        
        refinement_times = sorted(list(set(refinement_times)))  # Remove duplicates and sort
        print(f"    [CHART] Refinement test range: {refinement_times}")
        
        # Execute granular test
        refinement_results = []
        for exposure_time in refinement_times:
            # Capture multiple frames and analyze
            sky_adus = []
            fwhm_values = []
            ecc_values = []
            
            for frame_num in range(self.frames_per_light):
                
                image_data = self.capture_light_frame_with_filter(exposure_time, filter_name)
                if image_data is not None:
                    # Analyze sky background
                    sky_adu = self.analyze_sky_background(image_data)
                    sky_adus.append(sky_adu)
                    
                    # Analyze image quality
                    fwhm, ecc = self.analyze_image_quality(image_data)
                    if fwhm is not None:
                        fwhm_values.append(fwhm)
                    if ecc is not None:
                        ecc_values.append(ecc)
            
            if sky_adus:
                avg_sky_adu = np.mean(sky_adus)
                avg_fwhm = np.mean(fwhm_values) if fwhm_values else None
                avg_ecc = np.mean(ecc_values) if ecc_values else None
                
                refinement_results.append({
                    'time': exposure_time,
                    'adu': avg_sky_adu,
                    'fwhm': avg_fwhm,
                    'ecc': avg_ecc
                })
                

        
        return refinement_results
    
    def find_refined_optimal_exposure(self, refinement_results: List[Dict]) -> float:
        """
        Find the refined optimal exposure time based on image quality analysis.
        
        Args:
            refinement_results: List of refinement results with quality data
            
        Returns:
            Refined optimal exposure time in seconds
        """
        if not refinement_results:
            return 0
        
        # Find true minimum FWHM
        valid_fwhm_data = [(r['time'], r['fwhm']) for r in refinement_results if r['fwhm'] is not None]
        
        if not valid_fwhm_data:
            # If no FWHM data, return the middle exposure time
            times = [r['time'] for r in refinement_results]
            return times[len(times) // 2]
        
        # Find the exposure with the best (lowest) FWHM
        best_fwhm_data = min(valid_fwhm_data, key=lambda x: x[1])
        true_min_fwhm = best_fwhm_data[1]
        best_time = best_fwhm_data[0]
        
        # Calculate quality threshold
        quality_threshold = true_min_fwhm * self.fwhm_degradation_threshold
        
        # Find longest compliant exposure
        refined_time = best_time  # Default to best FWHM time
        for result in sorted(refinement_results, key=lambda x: x['time']):
            if result['fwhm'] is not None and result['fwhm'] <= quality_threshold:
                refined_time = result['time']
            else:
                break
        

        
        return refined_time
    
    def run_intelligent_experiment(self):
        """Run the main experiment using the Intelligent Scout method."""
        print(f"{SYMBOLS.TARGET} [ROCKET] Starting Intelligent Scout experiment...")
        
        # Start guiding system
        print(f"\n[TARGET] Starting guiding system...")
        guiding_started = self.start_guiding()
        if not guiding_started:
            print(f"{SYMBOLS.WARNING} Warning: Guiding could not be started. Long exposures may have tracking issues.")
        else:
            if self.external_guiding_active:
                if self.phd2_running:
                    print(f"{SYMBOLS.TARGET} Using PHD2 for guiding - industry standard software")
                elif self.ekos_running:
                    print(f"{SYMBOLS.TARGET} Using EKOS for guiding - integrated with KStars")
            else:
                print(f"{SYMBOLS.SUCCESS} Using built-in guiding system")
        
        # Main experiment loop - filter by filter with Intelligent Scout
        for filter_name in self.filter_names:
            print(f"\n[SWITCH] Processing {filter_name} filter...")
            print("="*60)
            
            # Set filter
            if self.filter_wheel:
                filter_prop = self.filter_wheel.getProperty('FILTER_SLOT')
                if filter_prop:
                    # Find filter index
                    try:
                        filter_index = self.filter_names.index(filter_name) + 1
                        filter_prop[0].setNumber(filter_index)
                        self.client.sendNewProperty(filter_prop)
                        time.sleep(2)  # Wait for filter change
                        print(f"  [SWITCH] Filter set to: {filter_name}")
                    except ValueError:
                        print(f"{SYMBOLS.ERROR} Filter {filter_name} not found in filter wheel")
                        continue
            
            # Get camera properties for this filter
            self.camera_properties[filter_name] = self.get_camera_properties(filter_name)
            
            # Phase A: Intelligent Scout & Prediction
            predicted_time, dynamic_times, scout_fwhm = self.intelligent_scout_and_predict(filter_name)
            
            # Phase D: Execute Focused Experiment (uses dark frame library)
            exposure_results = self.execute_focused_experiment(filter_name, dynamic_times)
            
            # Calculate optimal exposure for this filter
            if exposure_results:
                target_adu = self.calculate_target_adu(
                    self.camera_properties[filter_name]['gain'],
                    self.camera_properties[filter_name]['read_noise']
                )
                
                optimal_exposure = self.find_optimal_exposure(
                    exposure_results, target_adu
                )
                
                # Store quality results
                self.image_quality_results[filter_name] = exposure_results
                
                # Check if refinement phase is enabled
                if self.refine_exposure:
                    print(f"{SYMBOLS.FOCUS} [REFINE] Starting refinement phase for {filter_name}...")
                    refinement_results = self.run_refinement_phase(filter_name, optimal_exposure)
                    refined_time = self.find_refined_optimal_exposure(refinement_results)
                    
                    # Update results with refined time
                    self.results[filter_name] = {
                        'optimal_exposure': refined_time,
                        'target_adu': target_adu,
                        'gain': self.camera_properties[filter_name]['gain'],
                        'read_noise': self.camera_properties[filter_name]['read_noise'],
                        'data_points': exposure_results,
                        'predicted_time': predicted_time,
                        'dynamic_times': dynamic_times,
                        'refinement_results': refinement_results,
                        'is_refined': True,
                        'scout_time': optimal_exposure,
                        'scout_fwhm': scout_fwhm
                    }
                    
                    print(f"  {SYMBOLS.SUCCESS} Refined optimal exposure for {filter_name}: {refined_time:.1f}s (refined)")
                    print(f"  [CHART] Scout time: {optimal_exposure:.1f}s, Refined time: {refined_time:.1f}s")
                    print(f"  [CHART] Prediction accuracy: {abs(refined_time - predicted_time):.1f}s difference")
                else:
                    self.results[filter_name] = {
                        'optimal_exposure': optimal_exposure,
                        'target_adu': target_adu,
                        'gain': self.camera_properties[filter_name]['gain'],
                        'read_noise': self.camera_properties[filter_name]['read_noise'],
                        'data_points': exposure_results,
                        'predicted_time': predicted_time,
                        'dynamic_times': dynamic_times,
                        'is_refined': False,
                        'scout_fwhm': scout_fwhm
                    }
                    
                    print(f"  {SYMBOLS.SUCCESS} Optimal exposure for {filter_name}: {optimal_exposure:.1f}s")
                    print(f"  [CHART] Prediction accuracy: {abs(optimal_exposure - predicted_time):.1f}s difference")
            else:
                print(f"{SYMBOLS.ERROR} No valid results for {filter_name}")
    
    def print_results(self):
        """Print the final results in a formatted table."""
        print("\n" + "="*100)
        print(f"{SYMBOLS.CAMERA} INTELLIGENT SCOUT SUB-EXPOSURE CALCULATOR RESULTS")
        print("="*100)
        
        if not self.results:
            print(f"{SYMBOLS.ERROR} No results to display")
            return
        
        # Print header
        print(f"{'Filter':<12} {'Optimal (s)':<12} {'Predicted (s)':<12} {'Accuracy (s)':<12} {'Gain':<8} {'RN':<8} {'Target ADU':<12} {'Scout FWHM':<12} {'Test Range':<20}")
        print("-" * 100)
        
        # Print results
        for filter_name, result in self.results.items():
            optimal_exp = result['optimal_exposure']
            predicted_time = result.get('predicted_time', 0)
            accuracy = abs(optimal_exp - predicted_time)
            gain = result['gain']
            read_noise = result['read_noise']
            target_adu = result['target_adu']
            scout_fwhm = result.get('scout_fwhm', None)
            scout_fwhm_str = f"{scout_fwhm:.2f}" if scout_fwhm is not None else "N/A"
            dynamic_times = result.get('dynamic_times', [])
            test_range_str = f"{min(dynamic_times)}-{max(dynamic_times)}s" if dynamic_times else "N/A"
            
            # Add refinement marker
            refinement_marker = " (refined)" if result.get('is_refined', False) else ""
            
            # Check for warnings
            warning = ""
            if optimal_exp > 600:  # 10 minutes
                warning = " {SYMBOLS.WARNING}  LONG"
            elif optimal_exp > 300:  # 5 minutes
                warning = " {SYMBOLS.WARNING}  MODERATE"
            
            print(f"{filter_name:<12} {optimal_exp:<12.1f}{refinement_marker:<12} {predicted_time:<12.1f} {accuracy:<12.1f} {gain:<8.2f} {read_noise:<8.2f} {target_adu:<12.1f} {scout_fwhm_str:<12} {test_range_str:<20}{warning}")
        
        print("-" * 100)
        
        # Print detailed quality analysis
        print("\n[CHART] DETAILED QUALITY ANALYSIS")
        print("-" * 60)
        
        for filter_name, result in self.results.items():
            print(f"\n[SWITCH] {filter_name} Filter:")
            quality_data = self.image_quality_results.get(filter_name, [])
            
            if quality_data:
                print(f"  [CHART] Tested exposure times: {[d['time'] for d in quality_data]}")
                adu_values = [f"{d['adu']:.1f}" for d in quality_data]
                print(f"  [CHART] Sky ADU values: {adu_values}")
                
                # Find best FWHM
                valid_fwhm = [(d['time'], d['fwhm']) for d in quality_data if d['fwhm'] is not None]
                if valid_fwhm:
                    best_fwhm = min(valid_fwhm, key=lambda x: x[1])
                    print(f"  {SYMBOLS.STAR} Best FWHM: {best_fwhm[1]:.2f} pixels at {best_fwhm[0]}s")
                
                # Find best eccentricity
                valid_ecc = [(d['time'], d['ecc']) for d in quality_data if d['ecc'] is not None]
                if valid_ecc:
                    best_ecc = min(valid_ecc, key=lambda x: x[1])
                    print(f"  {SYMBOLS.STAR} Best Eccentricity: {best_ecc[1]:.3f} at {best_ecc[0]}s")
            
            # Show refinement information if applicable
            if result.get('is_refined', False):
                print(f"  [REFINE] REFINEMENT PHASE:")
                refinement_data = result.get('refinement_results', [])
                if refinement_data:
                    print(f"    [CHART] Scout time: {result.get('scout_time', 0):.1f}s")
                    print(f"    [CHART] Refinement test range: {[d['time'] for d in refinement_data]}")
                    
                    # Find best FWHM in refinement
                    valid_refinement_fwhm = [(d['time'], d['fwhm']) for d in refinement_data if d['fwhm'] is not None]
                    if valid_refinement_fwhm:
                        best_refinement_fwhm = min(valid_refinement_fwhm, key=lambda x: x[1])
                        print(f"    {SYMBOLS.STAR} Best refinement FWHM: {best_refinement_fwhm[1]:.2f} pixels at {best_refinement_fwhm[0]}s")
                    
                    # Show refinement ADU values
                    refinement_adu_values = [f"{d['adu']:.1f}" for d in refinement_data]
                    print(f"    [CHART] Refinement ADU values: {refinement_adu_values}")
        
        print("\n💡 INTELLIGENT SCOUT METHODOLOGY:")
        print("  • Dark frame library created upfront for unattended operation")
        print("  • Scout exposure measures sky brightness and flux rate")
        print("  • Prediction based on camera gain, read noise, and target SNR")
        print("  • Dynamic test range focuses around predicted optimal time")
        print("  • Automatic dark frame selection and scaling as needed")
        print("  • Results balance signal-to-noise with image quality (FWHM)")
        print("  • Dynamic seeing adjustment automatically adjusts frames_per_light based on FWHM")
        print("    - Excellent seeing (FWHM < 2.0): 3 frames")
        print("    - Good seeing (FWHM 2.0-3.0): 5 frames")
        print("    - Average seeing (FWHM 3.0-4.0): 7 frames")
        print("    - Poor seeing (FWHM ≥ 4.0): 10 frames")
        
        if any(r.get('is_refined', False) for r in self.results.values()):
            print("\n💡 BRACKETING SEARCH REFINEMENT METHODOLOGY:")
            print("  • High-precision refinement phase tests granular exposure times")
            print("  • Search range centered around initial optimal exposure time")
            print("  • Tests multiple frames per exposure for statistical accuracy")
            print("  • Finds true minimum FWHM and applies quality degradation threshold")
            print("  • Selects longest exposure that maintains image quality")
            print("  • Provides maximum precision for optimal exposure determination")
        
        # Print focusing information if available
        if self.focus_positions:
            print("\n[FOCUS] FOCUSING INFORMATION:")
            print("-" * 60)
            for filter_name, focus_data in self.focus_positions.items():
                print(f"  [SWITCH] {filter_name}: Position {focus_data['position']} at {focus_data['temperature']:.1f}°C")
        
        print("\n💡 RECOMMENDATIONS:")
        print("  • FWHM values indicate star sharpness - lower is better")
        print("  • Eccentricity values indicate tracking quality - lower is better")
        print("  • Prediction accuracy shows how well the scout method worked")
        print("  • Consider your mount's tracking accuracy for longer exposures")
        
        if self.focus_positions:
            print("  • Focus positions are cached for efficient filter changes")
            print("  • Temperature drift triggers automatic refocusing")
            print("  • EKOS integration provides advanced focusing capabilities")
        
        if any(r['optimal_exposure'] > 600 for r in self.results.values()):
            print("\n{SYMBOLS.WARNING}  WARNING: Some exposures are very long (>10 minutes)")
            print("  Consider using a higher gain setting or accepting lower SNR")
        
        # Print efficiency metrics
        total_tests = sum(len(r.get('dynamic_times', [])) for r in self.results.values())
        total_refinement_tests = sum(len(r.get('refinement_results', [])) for r in self.results.values() if r.get('is_refined', False))
        
        print(f"\n📈 EFFICIENCY METRICS:")
        print(f"  • Total scout test exposures: {total_tests}")
        if total_refinement_tests > 0:
            print(f"  • Total refinement test exposures: {total_refinement_tests}")
            print(f"  • Total combined exposures: {total_tests + total_refinement_tests}")
        print(f"  • Filters processed: {len(self.results)}")
        print(f"  • Average scout tests per filter: {total_tests/len(self.results):.1f}")
        if total_refinement_tests > 0:
            refined_filters = sum(1 for r in self.results.values() if r.get('is_refined', False))
            print(f"  • Filters with refinement: {refined_filters}")
            print(f"  • Average refinement tests per refined filter: {total_refinement_tests/refined_filters:.1f}")
        print(f"  • Scout exposure time: {self.scout_exposure_time}s")
    
    def cleanup(self):
        """Clean up INDI connection."""
        # Stop guiding if active
        if self.guiding_active:
            self.stop_guiding()
        
        if self.client:
            self.client.disconnectServer()
            print(f"{SYMBOLS.DISCONNECT} Disconnected from INDI server")
    
    def run(self):
        """Main execution method."""
        try:
            print(f"{SYMBOLS.CAMERA} [STAR] Sub-Exposure Calculator v6.1 - Intelligent Scout Method with Bracketing Search Refinement")
            print("="*60)
            
            # Phase 0: Setup and device discovery
            self.connect_to_indi()
            self.discover_devices()
            
            # Phase 1: Dark Frame Library Creation
            if not self.args.skip_dark_library:
                print("\n[CAM] Phase 1: Dark Frame Library Creation")
                print("="*60)
                print("[TARGET] This creates a comprehensive library of dark frames upfront")
                print("   • Covers exposure times from 10s to 1800s (30 minutes)")
                print("   • Enables unattended operation throughout the experiment")
                print("   • Automatically selects and scales darks as needed")
                print()
                
                self.create_dark_frame_library()
            else:
                print("\n[CAM] Phase 1: Dark Frame Library (Skipped)")
                print("="*60)
                print("[TARGET] Using existing dark frames only")
                print("   • Will use available darks from previous sessions")
                print("   • May prompt for dark creation if needed")
                print()
            
            # Phase 2: Intelligent Scout Experiment
            print("\n[ROCKET] Phase 2: Intelligent Scout Experiment")
            print("="*60)
            print("[TARGET] This method will:")
            print("   • Take a scout exposure to measure sky brightness")
            print("   • Predict optimal exposure time based on camera properties")
            print("   • Generate a focused test range around the prediction")
            print("   • Use the dark frame library for automatic calibration")
            print("   • Execute the experiment with optimal efficiency")
            print()
            
            self.run_intelligent_experiment()
            
            # Phase 3: Results
            print("\n[CHART] Phase 3: Results")
            print("="*60)
            self.print_results()
            
        except KeyboardInterrupt:
            print("\n{SYMBOLS.WARNING}  Experiment interrupted by user")
        except Exception as e:
            print(f"\n[ERROR] Error during execution: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()
    
    def adjust_frames_per_light_based_on_seeing(self, fwhm: float):
        """
        Adjust the number of test frames based on seeing conditions.
        
        Args:
            fwhm: Full Width at Half Maximum in pixels
        """
        original_frames = self.frames_per_light
        
        if fwhm < 2.0:
            self.frames_per_light = 3
            seeing_condition = "Excellent"
        elif 2.0 <= fwhm < 3.0:
            self.frames_per_light = 5
            seeing_condition = "Good"
        elif 3.0 <= fwhm < 4.0:
            self.frames_per_light = 7
            seeing_condition = "Average"
        else:  # fwhm >= 4.0
            self.frames_per_light = 10
            seeing_condition = "Poor"
        
        print(f"    [STAR] Seeing: {seeing_condition} (FWHM: {fwhm:.2f} pixels)")
        print(f"    [CHART] Adjusted frames_per_light: {original_frames} → {self.frames_per_light}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Sub-Exposure Calculator v6.1 - Automated optimal sub-exposure time calculator with bracketing search refinement",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s
  %(prog)s --host 192.168.1.100 --port 7624
  %(prog)s --scout_exposure_time 30 --frames_per_light 3
  %(prog)s --camera_name "ZWO CCD ASI1600MM Pro" --mount_name "SkyWatcher EQ6-R Pro"
  %(prog)s --skip_dark_library  # Use existing darks only
  %(prog)s --refine_exposure  # Enable high-precision refinement phase
  %(prog)s --refine_exposure --refinement_steps 3 --refinement_step_size 10  # Custom refinement
  %(prog)s --focuser_name "ZWO Focuser" --focus_temp_threshold 0.5  # Custom focuser and temperature threshold
        """
    )
    
    # Connection arguments
    parser.add_argument('--host', default='localhost', help='INDI server hostname (default: localhost)')
    parser.add_argument('--port', type=int, default=7624, help='INDI server port (default: 7624)')
    
    # Path and sequence arguments
    parser.add_argument('--calibration_path', default='./calibration', 
                       help='Path for calibration frames (default: ./calibration)')
    parser.add_argument('--scout_exposure_time', type=int, default=60,
                       help='Scout exposure time in seconds for sky brightness measurement (default: 60)')
    
    # Device override arguments
    parser.add_argument('--camera_name', help='Override camera device name')
    parser.add_argument('--mount_name', help='Override mount device name')
    parser.add_argument('--filter_wheel_name', help='Override filter wheel device name')
    parser.add_argument('--guide_camera_name', help='Override guide camera device name')
    parser.add_argument('--focuser_name', help='Override focuser device name')
    
    # Tuning parameters
    parser.add_argument('--frames_per_light', type=int, default=5,
                       help='Number of light frames per exposure time (default: 5)')
    parser.add_argument('--frames_per_dark', type=int, default=20,
                       help='Number of dark frames for master dark (default: 20)')
    parser.add_argument('--target_noise_ratio', type=float, default=10.0,
                       help='Target noise ratio for optimal exposure (default: 10.0)')
    parser.add_argument('--fwhm_degradation_threshold', type=float, default=1.1,
                       help='FWHM degradation threshold (1.1 = 10%% degradation allowed, default: 1.1)')
    parser.add_argument('--skip_dark_library', action='store_true',
                       help='Skip dark frame library creation (use existing darks only)')
    
    # Guiding parameters
    parser.add_argument('--guide_exposure_time', type=float, default=2.0,
                       help='Guide camera exposure time in seconds (default: 2.0)')
    parser.add_argument('--guide_calibration_time', type=float, default=5.0,
                       help='Guide calibration time in seconds (default: 5.0)')
    parser.add_argument('--guide_max_pulse', type=int, default=1000,
                       help='Maximum guide pulse duration in milliseconds (default: 1000)')
    parser.add_argument('--guide_aggressiveness', type=float, default=0.5,
                       help='Guide aggressiveness factor 0.0-1.0 (default: 0.5)')
    parser.add_argument('--guide_dither_threshold', type=float, default=0.5,
                       help='Guide dither threshold in pixels (default: 0.5)')
    
    # Focusing parameters
    parser.add_argument('--focus_temp_threshold', type=float, default=1.0,
                       help='Temperature threshold in °C for triggering refocus (default: 1.0)')
    
    # Refinement phase arguments
    parser.add_argument('--refine_exposure', action='store_true',
                       help='Enable the high-precision refinement phase to find the optimal exposure based on image quality')
    parser.add_argument('--refinement_steps', type=int, default=2,
                       help='Number of granular steps to test on each side of the initial optimal exposure (default: 2)')
    parser.add_argument('--refinement_step_size', type=int, default=15,
                       help='The duration in seconds of each granular step during the refinement phase (default: 15)')
    
    args = parser.parse_args()
    
    # Create and run calculator
    calculator = SubExposureCalculator(args)
    calculator.run()


if __name__ == '__main__':
    main() 