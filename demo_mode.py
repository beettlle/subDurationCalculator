#!/usr/bin/env python3
"""
Demo Mode for Sub-Exposure Calculator v6.1
This script simulates the sub-exposure calculator without requiring actual INDI devices.
Useful for testing, demonstration, and understanding the workflow.
Now includes the optional bracketing search refinement phase for high-precision exposure optimization.
"""

import argparse
import time
import random
import math
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Configuration loading
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    print("⚠️  Warning: PyYAML not available. Using default configuration.")
    YAML_AVAILABLE = False

# Image quality analysis imports (for demo simulation)
try:
    from photutils.detection import DAOStarFinder
    from photutils.centroids import centroid_com
    from scipy import ndimage
    PHOTUTILS_AVAILABLE = True
except ImportError:
    print("⚠️  Warning: photutils not available. Using fallback star detection method.")
    PHOTUTILS_AVAILABLE = False

class DemoSubExposureCalculator:
    """Demo version of the sub-exposure calculator."""
    
    def __init__(self, args):
        """Initialize the demo calculator."""
        self.args = args
        self.calibration_path = Path(args.calibration_path)
        self.scout_exposure_time = getattr(args, 'scout_exposure_time', 10)
        self.frames_per_light = args.frames_per_light
        self.frames_per_dark = args.frames_per_dark
        self.target_noise_ratio = args.target_noise_ratio
        self.fwhm_degradation_threshold = getattr(args, 'fwhm_degradation_threshold', 1.1)
        
        # Load configuration
        self.config = self.load_configuration()
        
        # Analysis parameters from config
        self.sky_region_fraction = self.config.get('analysis', {}).get('sky_region_fraction', 0.8)
        self.sigma_clip_threshold = self.config.get('analysis', {}).get('sigma_clip_threshold', 3.0)
        self.min_valid_pixels = self.config.get('analysis', {}).get('min_valid_pixels', 1000)
        
        # Demo filter names
        self.filter_names = ['Luminance', 'Red', 'Green', 'Blue', 'Ha', 'OIII', 'SII']
        
        # Demo camera properties (will be overridden by config values)
        self.camera_properties = {
            'gain': 1.0,
            'read_noise': 5.2
        }
        
        # Update camera properties from config
        camera_config = self.config.get('camera', {})
        self.camera_properties['gain'] = camera_config.get('default_gain', 1.0)
        self.camera_properties['read_noise'] = camera_config.get('default_read_noise', 5.2)
        
        # Refinement phase parameters
        self.refine_exposure = getattr(args, 'refine_exposure', False)
        self.refinement_steps = getattr(args, 'refinement_steps', 2)
        self.refinement_step_size = getattr(args, 'refinement_step_size', 15)
        
        # Results storage
        self.results = {}
        self.image_quality_results = {}
        
        # Advanced focusing properties (demo simulation)
        self.focuser = None
        self.focus_positions = {}  # Cache: {'Red': {'position': 5185, 'temperature': 10.5}}
        self.focus_temp_threshold = getattr(args, 'focus_temp_threshold', 1.0)  # Temperature threshold for refocusing
        
        # Create calibration directory
        self.calibration_path.mkdir(parents=True, exist_ok=True)
    
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
                    print(f"✅ Loaded configuration from {config_path}")
                    return config
                except Exception as e:
                    print(f"⚠️  Error loading {config_path}: {e}")
        
        # Return default configuration if no file found
        print("⚠️  No configuration file found, using defaults")
        return {
            'analysis': {
                'sky_region_fraction': 0.8,
                'sigma_clip_threshold': 3.0,
                'min_valid_pixels': 1000
            },
            'camera': {
                'default_gain': 1.0,
                'default_read_noise': 5.2
            }
        }
    
    def simulate_device_discovery(self):
        """Simulate device discovery."""
        print("🔌 Connecting to INDI server...")
        time.sleep(1)
        print("✅ Connected to INDI server successfully")
        
        print("🔍 Discovering INDI devices...")
        time.sleep(0.5)
        print("📷 Selected camera: ZWO CCD ASI1600MM Pro (Demo)")
        print("🔭 Selected mount: SkyWatcher EQ6-R Pro (Demo)")
        print("🎨 Selected filter wheel: ZWO EFW (Demo)")
        print("🎯 Selected guide camera: ZWO ASI120MM Mini (Demo)")
        print("🔍 Selected focuser: ZWO Focuser (Demo)")
        print(f"🎨 Available filters: {', '.join(self.filter_names)}")
        
        # Simulate external guiding detection
        print("🔍 Checking for external guiding software...")
        time.sleep(0.5)
        print("ℹ️  No external guiding software detected")
    
    def simulate_calibration(self):
        """Simulate calibration frame creation."""
        print("\n📸 Phase 1: Calibration Frame Verification")
        print("=" * 50)
        
        # In the intelligent scout approach, we create a dark frame library upfront
        # that covers a range of exposure times for automatic selection
        print("📸 Creating dark frame library for intelligent scout approach...")
        print("🔒 Please cover the telescope and press Enter when ready...")
        input()
        
        # Create dark frame library for common exposure times
        dark_library_times = [10, 15, 20, 30, 45, 60, 90, 120, 180, 240, 300, 420, 600, 900, 1200, 1800]
        
        for exposure_time in dark_library_times:
            print(f"📸 Creating dark for {exposure_time}s exposure...")
            print(f"  Capturing {self.frames_per_dark} dark frames...")
            for i in range(self.frames_per_dark):
                print(f"    Dark frame {i+1}/{self.frames_per_dark}...")
                time.sleep(0.05)  # Faster simulation
            
            print("🔬 Creating master dark by median combining...")
            time.sleep(0.2)  # Faster simulation
            
            # Create a dummy dark file
            dark_path = self.calibration_path / f"master_dark_library_{exposure_time}s.fits"
            dark_path.touch()
            print(f"✅ Master dark saved to {dark_path}")
        
        print("\n🔓 Please uncover the telescope and press Enter when ready...")
        input()
        print("✅ Dark frame library complete - ready for intelligent scout experiment")
    
    def simulate_slew_to_zenith(self):
        """Simulate slewing to zenith."""
        print("🔭 Slew to zenith...")
        time.sleep(2)
        print("⏳ Waiting for slew to complete...")
        time.sleep(3)
        print("✅ Slew to zenith completed")
    
    def simulate_start_guiding(self):
        """Simulate starting the guiding process."""
        print("🎯 Using built-in guiding (no external software detected)")
        print("🎯 Starting built-in guiding process...")
        print("  📸 Taking guide exposure to find guide stars...")
        time.sleep(1)
        print("  🔍 Finding guide stars...")
        time.sleep(1)
        print("  ⭐ Selected guide star at position: (256.5, 128.3)")
        print("  🎯 Starting guiding thread...")
        time.sleep(1)
        print("  ⏳ Waiting for guiding to stabilize...")
        time.sleep(2)
        print("✅ Built-in guiding started successfully")
        print("✅ Using built-in guiding system")
    
    def simulate_focusing(self, filter_name: str) -> bool:
        """
        Simulate advanced, filter-aware focusing for demo purposes.
        
        Args:
            filter_name: Name of the current filter
            
        Returns:
            True if focusing was successful (always True in demo)
        """
        print(f"    🔍 Performing filter-aware focusing for {filter_name}...")
        
        # Simulate current temperature
        current_temp = 20.0 + random.uniform(-2.0, 2.0)  # Simulate temperature around 20°C
        print(f"    🌡️  Current temperature: {current_temp:.1f}°C")
        
        # Check if we have a cached position for this filter
        if filter_name in self.focus_positions:
            cached_data = self.focus_positions[filter_name]
            cached_position = cached_data['position']
            cached_temp = cached_data['temperature']
            
            # Check if temperature has drifted significantly
            temp_diff = abs(current_temp - cached_temp)
            if temp_diff < self.focus_temp_threshold:
                print(f"    ✅ Temperature stable ({temp_diff:.1f}°C < {self.focus_temp_threshold}°C)")
                print(f"    🔍 Moving to cached position: {cached_position}")
                time.sleep(1)  # Simulate focuser movement
                print(f"    ✅ Simple re-focus completed for {filter_name}")
                return True
            else:
                print(f"    🌡️  Temperature drift detected: {temp_diff:.1f}°C > {self.focus_temp_threshold}°C")
                print(f"    🔄 Full refocus needed due to temperature change")
        else:
            print(f"    🆕 First time using {filter_name} filter")
            print(f"    🔄 Full refocus needed for new filter")
        
        # Simulate full autofocus routine
        print(f"    🔄 Starting full autofocus routine...")
        print(f"    🔍 Using internal HFD autofocus...")
        
        # Simulate focuser movement and HFD measurement
        base_position = 5000 + random.uniform(-500, 500)
        print(f"    📍 Starting position: {int(base_position)}")
        
        # Simulate testing multiple positions
        test_positions = [int(base_position - 200), int(base_position - 100), 
                         int(base_position), int(base_position + 100), int(base_position + 200)]
        
        for i, position in enumerate(test_positions):
            time.sleep(0.2)  # Simulate focuser movement and exposure
            
            # Simulate HFD measurement
            hfd = 3.0 + random.uniform(-0.5, 0.5) + abs(position - base_position) * 0.01
            print(f"      📊 Position {position}: HFD = {hfd:.2f}")
        
        # Find best position (closest to base_position)
        best_position = int(base_position)
        best_hfd = 3.0
        print(f"    ⭐ Best focus: position {best_position} with HFD {best_hfd:.2f}")
        
        # Simulate moving to best position
        print(f"    🔍 Moving to best focus position...")
        time.sleep(1)  # Simulate focuser movement
        
        # Update cache
        self.focus_positions[filter_name] = {
            'position': best_position,
            'temperature': current_temp
        }
        
        print(f"    ✅ HFD autofocus completed successfully")
        return True
    
    def simulate_camera_properties(self, filter_name: str) -> Dict:
        """Simulate getting camera properties."""
        # Get camera defaults from config
        camera_config = self.config.get('camera', {})
        base_gain = camera_config.get('default_gain', 1.0)
        base_rn = camera_config.get('default_read_noise', 5.2)
        
        # Add some realistic variation
        gain_variation = random.uniform(-0.1, 0.1)
        rn_variation = random.uniform(-0.5, 0.5)
        
        properties = {
            'gain': base_gain + gain_variation,
            'read_noise': base_rn + rn_variation
        }
        
        print(f"📊 Camera properties for {filter_name}: Gain={properties['gain']:.2f}, RN={properties['read_noise']:.2f} (simulated)")
        return properties
    
    def simulate_light_frame_capture(self, exposure_time: int) -> Tuple[float, Optional[float], Optional[float]]:
        """Simulate capturing a light frame and return simulated sky ADU, FWHM, and eccentricity."""
        # Simulate realistic sky background ADU based on exposure time
        # Sky background typically increases linearly with exposure time
        base_sky_adu = 50  # ADU per second
        sky_adu = base_sky_adu * exposure_time
        
        # Add some realistic noise
        noise = random.uniform(-0.1, 0.1) * sky_adu
        sky_adu += noise
        
        # Add some filter-specific variations
        filter_factor = random.uniform(0.8, 1.2)
        sky_adu *= filter_factor
        
        # Simulate FWHM (typically degrades with longer exposures due to tracking)
        base_fwhm = 2.5  # pixels
        fwhm_degradation = exposure_time / 600.0  # Degrade over 10 minutes
        fwhm = base_fwhm + random.uniform(-0.2, 0.2) + fwhm_degradation * random.uniform(0, 1.0)
        fwhm = max(fwhm, 1.5)  # Minimum reasonable FWHM
        
        # Simulate eccentricity (should be low for good tracking)
        base_ecc = 0.1
        ecc_degradation = exposure_time / 600.0  # Degrade over 10 minutes
        ecc = base_ecc + random.uniform(-0.05, 0.05) + ecc_degradation * random.uniform(0, 0.3)
        ecc = min(max(ecc, 0.0), 0.8)  # Keep within reasonable bounds
        
        return max(sky_adu, 10), fwhm, ecc  # Ensure minimum sky ADU
    
    def simulate_image_quality_analysis(self, exposure_time: int) -> Tuple[Optional[float], Optional[float]]:
        """Simulate image quality analysis for demo purposes."""
        # Simulate FWHM (typically degrades with longer exposures due to tracking)
        base_fwhm = 2.5  # pixels
        fwhm_degradation = exposure_time / 600.0  # Degrade over 10 minutes
        fwhm = base_fwhm + random.uniform(-0.2, 0.2) + fwhm_degradation * random.uniform(0, 1.0)
        fwhm = max(fwhm, 1.5)  # Minimum reasonable FWHM
        
        # Simulate eccentricity (should be low for good tracking)
        base_ecc = 0.1
        ecc_degradation = exposure_time / 600.0  # Degrade over 10 minutes
        ecc = base_ecc + random.uniform(-0.05, 0.05) + ecc_degradation * random.uniform(0, 0.3)
        ecc = min(max(ecc, 0.0), 0.8)  # Keep within reasonable bounds
        
        return fwhm, ecc
    
    def calculate_target_adu(self, gain: float, read_noise: float) -> float:
        """Calculate target ADU based on gain and read noise."""
        target_adu = (self.target_noise_ratio * read_noise) ** 2 / gain
        return target_adu
    
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
        
        print(f"    🌟 Seeing: {seeing_condition} (FWHM: {fwhm:.2f} pixels)")
        print(f"    📊 Adjusted frames_per_light: {original_frames} → {self.frames_per_light}")
    
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
    
    def simulate_intelligent_scout_and_predict(self, filter_name: str) -> Tuple[float, List[int], float]:
        """
        Simulate Phase A: Intelligent Scout & Prediction
        
        Args:
            filter_name: Name of the current filter
            
        Returns:
            Tuple of (predicted_optimal_time, dynamic_exposure_times, fwhm)
        """
        print(f"  🎯 Phase A: Intelligent Scout & Prediction for {filter_name}")
        
        # Get camera properties for this filter
        camera_props = self.simulate_camera_properties(filter_name)
        gain = camera_props['gain']
        read_noise = camera_props['read_noise']
        
        # Calculate target ADU
        target_adu = self.calculate_target_adu(gain, read_noise)
        print(f"    📊 Target ADU: {target_adu:.2f}")
        print(f"    📊 Camera Gain: {gain:.2f}, Read Noise: {read_noise:.2f}")
        
        # Simulate capturing a scout frame
        print(f"    📸 Taking scout exposure ({self.scout_exposure_time}s)...")
        time.sleep(0.5)  # Simulate exposure time
        
        # Simulate analyzing the frame to get scout_adu and fwhm
        scout_adu = self.simulate_light_frame_capture(self.scout_exposure_time)[0]
        fwhm = random.uniform(1.5, 5.0)  # Random FWHM between 1.5 and 5.0 pixels
        
        print(f"    📊 Scout frame sky ADU: {scout_adu:.2f}")
        print(f"    📊 Measured FWHM: {fwhm:.2f} pixels")
        
        # Adjust frames_per_light based on seeing conditions
        self.adjust_frames_per_light_based_on_seeing(fwhm)
        
        # Calculate sky flux rate
        sky_flux_rate = scout_adu / self.scout_exposure_time
        print(f"    📊 Sky flux rate: {sky_flux_rate:.2f} ADU/sec")
        
        # Predict optimal time
        predicted_optimal_time = target_adu / sky_flux_rate
        print(f"    🎯 Predicted optimal time: {predicted_optimal_time:.1f}s")
        
        # Generate dynamic test range
        dynamic_times = self.generate_dynamic_test_range(predicted_optimal_time)

        
        return predicted_optimal_time, dynamic_times, fwhm
    
    def find_optimal_exposure(self, exposure_times: List[int], sky_adus: List[float], 
                            target_adu: float) -> float:
        """Find optimal exposure time using linear interpolation."""
        if len(exposure_times) < 2:
            return exposure_times[0] if exposure_times else 0
        
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
    
    def run_refinement_phase(self, filter_name: str, scout_time: float) -> List[Dict]:
        """
        Run the high-precision refinement phase to find the optimal exposure based on image quality.
        
        Args:
            filter_name: Name of the current filter
            scout_time: Initial optimal exposure time from scout phase
            
        Returns:
            List of refinement results with detailed quality analysis
        """
        print(f"  🔎 Refining exposure for {filter_name} filter...")
        print(f"    🎯 Initial optimal time: {scout_time:.1f}s")
        
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
        # Execute granular test
        refinement_results = []
        for exposure_time in refinement_times:
            
            # Capture multiple frames and analyze
            sky_adus = []
            fwhm_values = []
            ecc_values = []
            
            for frame_num in range(self.frames_per_light):
                time.sleep(0.1)  # Simulate exposure time
                
                sky_adu, fwhm, ecc = self.simulate_light_frame_capture(exposure_time)
                sky_adus.append(sky_adu)
                if fwhm is not None:
                    fwhm_values.append(fwhm)
                if ecc is not None:
                    ecc_values.append(ecc)
            
            if sky_adus:
                avg_sky_adu = sum(sky_adus) / len(sky_adus)
                avg_fwhm = sum(fwhm_values) / len(fwhm_values) if fwhm_values else None
                avg_ecc = sum(ecc_values) / len(ecc_values) if ecc_values else None
                
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
    
    def run_demo_experiment(self):
        """Run the demo experiment."""
        print("\n🚀 Phase 2: Automated Experiment")
        print("=" * 50)
        
        # Simulate slewing to zenith
        self.simulate_slew_to_zenith()
        
        # Simulate starting guiding
        print("\n🎯 Starting guiding system...")
        self.simulate_start_guiding()
        
        # Main experiment loop
        for filter_name in self.filter_names:
            print(f"\n🎨 Switching to {filter_name} filter...")
            time.sleep(1)  # Simulate filter change
            
            # Perform filter-aware focusing
            print("    🔍 Performing filter-aware focusing...")
            focusing_success = self.simulate_focusing(filter_name)
            if not focusing_success:
                print("    ⚠️  Warning: Focusing failed, continuing with current focus position")
            else:
                print("    ✅ Focusing completed successfully")
            
            # Get camera properties for this filter
            self.camera_properties = self.simulate_camera_properties(filter_name)
            
            # Phase A: Intelligent Scout & Prediction
            predicted_optimal_time, dynamic_times, scout_fwhm = self.simulate_intelligent_scout_and_predict(filter_name)
            
            # Test each exposure time in the dynamic range
            exposure_results = []
            for exposure_time in dynamic_times:
                print(f"  📸 Capturing {self.frames_per_light} frames at {exposure_time}s...")
                
                # Capture multiple frames and analyze
                sky_adus = []
                fwhm_values = []
                ecc_values = []
                
                for frame_num in range(self.frames_per_light):
                    print(f"    Frame {frame_num + 1}/{self.frames_per_light}...")
                    time.sleep(0.2)  # Simulate exposure time
                    
                    sky_adu, fwhm, ecc = self.simulate_light_frame_capture(exposure_time)
                    sky_adus.append(sky_adu)
                    if fwhm is not None:
                        fwhm_values.append(fwhm)
                    if ecc is not None:
                        ecc_values.append(ecc)
                
                if sky_adus:
                    avg_sky_adu = sum(sky_adus) / len(sky_adus)
                    avg_fwhm = sum(fwhm_values) / len(fwhm_values) if fwhm_values else None
                    avg_ecc = sum(ecc_values) / len(ecc_values) if ecc_values else None
                    
                    exposure_results.append({
                        'time': exposure_time,
                        'adu': avg_sky_adu,
                        'fwhm': avg_fwhm,
                        'ecc': avg_ecc
                    })
                    
                    print(f"    Average sky ADU: {avg_sky_adu:.2f}")
                    if avg_fwhm is not None:
                        print(f"    Average FWHM: {avg_fwhm:.2f} pixels")
                    if avg_ecc is not None:
                        print(f"    Average Eccentricity: {avg_ecc:.3f}")
            
            # Calculate optimal exposure for this filter
            if exposure_results:
                # Extract data for compatibility
                exposure_times = [r['time'] for r in exposure_results]
                sky_adus = [r['adu'] for r in exposure_results]
                
                target_adu = self.calculate_target_adu(
                    self.camera_properties['gain'],
                    self.camera_properties['read_noise']
                )
                
                optimal_exposure = self.find_optimal_exposure(
                    list(exposure_times), list(sky_adus), target_adu
                )
                
                # Store quality results
                self.image_quality_results[filter_name] = exposure_results
                
                # Check if refinement phase is enabled
                if self.refine_exposure:
                    print(f"  🔎 Starting refinement phase for {filter_name}...")
                    refinement_results = self.run_refinement_phase(filter_name, optimal_exposure)
                    refined_time = self.find_refined_optimal_exposure(refinement_results)
                    
                    # Update results with refined time
                    self.results[filter_name] = {
                        'optimal_exposure': refined_time,
                        'target_adu': target_adu,
                        'gain': self.camera_properties['gain'],
                        'read_noise': self.camera_properties['read_noise'],
                        'data_points': exposure_results,
                        'refinement_results': refinement_results,
                        'is_refined': True,
                        'scout_time': optimal_exposure,
                        'scout_fwhm': scout_fwhm
                    }
                    
                    print(f"  ✅ Refined optimal exposure for {filter_name}: {refined_time:.1f}s (refined)")
                    print(f"  📊 Scout time: {optimal_exposure:.1f}s, Refined time: {refined_time:.1f}s")
                else:
                    self.results[filter_name] = {
                        'optimal_exposure': optimal_exposure,
                        'target_adu': target_adu,
                        'gain': self.camera_properties['gain'],
                        'read_noise': self.camera_properties['read_noise'],
                        'data_points': exposure_results,
                        'is_refined': False,
                        'scout_fwhm': scout_fwhm
                    }
                    
                    print(f"  ✅ Optimal exposure for {filter_name}: {optimal_exposure:.1f}s")
    
    def print_results(self):
        """Print the final results in a formatted table."""
        print("\n📊 Phase 3: Results")
        print("=" * 50)
        print("📊 SUB-EXPOSURE CALCULATOR RESULTS")
        print("=" * 80)
        
        if not self.results:
            print("❌ No results to display")
            return
        
        # Print header
        print(f"{'Filter':<15} {'Optimal Exp (s)':<15} {'Gain':<8} {'Read Noise':<12} {'Target ADU':<12} {'Scout FWHM':<12} {'Median FWHM':<12} {'Median Ecc':<10}")
        print("-" * 100)
        
        # Print results
        for filter_name, result in self.results.items():
            optimal_exp = result['optimal_exposure']
            gain = result['gain']
            read_noise = result['read_noise']
            target_adu = result['target_adu']
            scout_fwhm = result.get('scout_fwhm', None)
            scout_fwhm_str = f"{scout_fwhm:.2f}" if scout_fwhm is not None else "N/A"
            
            # Add refinement marker
            refinement_marker = " (refined)" if result.get('is_refined', False) else ""
            
            # Get quality data for optimal exposure
            quality_data = self.image_quality_results.get(filter_name, [])
            optimal_quality = None
            for data_point in quality_data:
                if abs(data_point['time'] - optimal_exp) < 1.0:  # Find closest match
                    optimal_quality = data_point
                    break
            
            median_fwhm = f"{optimal_quality['fwhm']:.2f}" if optimal_quality and optimal_quality['fwhm'] is not None else "N/A"
            median_ecc = f"{optimal_quality['ecc']:.2f}" if optimal_quality and optimal_quality['ecc'] is not None else "N/A"
            
            # Check for warnings
            warning = ""
            if optimal_exp > 600:  # 10 minutes
                warning = " ⚠️  LONG EXPOSURE"
            elif optimal_exp > 300:  # 5 minutes
                warning = " ⚠️  MODERATE EXPOSURE"
            
            print(f"{filter_name:<15} {optimal_exp:<15.1f}{refinement_marker:<15} {gain:<8.2f} {read_noise:<12.2f} {target_adu:<12.1f} {scout_fwhm_str:<12} {median_fwhm:<12} {median_ecc:<10}{warning}")
        
        print("-" * 100)
        print("💡 Recommendations:")
        print("  • Optimal exposure times now balance signal-to-noise with star quality (FWHM)")
        print("  • FWHM values indicate star sharpness - lower is better")
        print("  • Eccentricity values indicate tracking quality - lower is better")
        print("  • Consider your mount's tracking accuracy when using longer exposures")
        print("  • You may need to adjust based on your specific imaging conditions")
        print("  • This was a DEMO run - results are simulated for demonstration purposes")
        
        # Show focusing information if available
        if self.focus_positions:
            print("\n🔍 FOCUSING INFORMATION (Demo):")
            print("-" * 60)
            for filter_name, focus_data in self.focus_positions.items():
                print(f"  🎨 {filter_name}: Position {focus_data['position']} at {focus_data['temperature']:.1f}°C")
            print("  • Focus positions are cached for efficient filter changes")
            print("  • Temperature drift triggers automatic refocusing")
            print("  • Demo simulates realistic focusing behavior")
        
        # Show refinement information if applicable
        refined_filters = [name for name, result in self.results.items() if result.get('is_refined', False)]
        if refined_filters:
            print(f"\n🔎 REFINEMENT PHASE SUMMARY:")
            print(f"  • Filters with refinement: {', '.join(refined_filters)}")
            print(f"  • Refinement steps: {self.refinement_steps} on each side")
            print(f"  • Refinement step size: {self.refinement_step_size}s")
            print(f"  • High-precision bracketing search completed for optimal image quality")
        
        if any(r['optimal_exposure'] > 600 for r in self.results.values()):
            print("\n⚠️  WARNING: Some exposures are very long (>10 minutes)")
            print("  Consider using a higher gain setting or accepting lower SNR")
    
    def run(self):
        """Main execution method."""
        try:
            print("🌟 Sub-Exposure Calculator v6.1 - DEMO MODE")
            print("=" * 60)
            print("⚠️  This is a DEMO version with simulated data")
            print("   Use the real script for actual measurements")
            print("🔍 ADVANCED FOCUSING ENABLED - Filter-aware focusing system")
            print("🌟 DYNAMIC SEEING ADJUSTMENT ENABLED - Automatic frames_per_light adjustment")
            if self.refine_exposure:
                print("🔎 REFINEMENT PHASE ENABLED - High-precision bracketing search")
            print("=" * 60)
            
            # Phase 0: Setup and device discovery
            self.simulate_device_discovery()
            
            # Phase 1: Guided calibration
            self.simulate_calibration()
            
            # Phase 2: Automated experiment
            self.run_demo_experiment()
            
            # Phase 3: Results
            self.print_results()
            
        except KeyboardInterrupt:
            print("\n⚠️  Demo interrupted by user")
        except Exception as e:
            print(f"\n❌ Error during demo execution: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Main entry point for demo mode."""
    parser = argparse.ArgumentParser(
        description="Sub-Exposure Calculator v6.0 - DEMO MODE",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This is a DEMO version that simulates the sub-exposure calculator without requiring
actual INDI devices. Use this for testing, demonstration, and understanding the workflow.
Now includes the Intelligent Scout method with dynamic seeing adjustment.

Examples:
  %(prog)s
  %(prog)s --scout_exposure_time 15 --frames_per_light 3
  %(prog)s --refine_exposure --refinement_steps 3 --refinement_step_size 10
  %(prog)s --focus_temp_threshold 0.5  # Custom temperature threshold for focusing
        """
    )
    
    # Path and sequence arguments
    parser.add_argument('--calibration_path', default='./calibration_demo', 
                       help='Path for calibration frames (default: ./calibration_demo)')
    parser.add_argument('--scout_exposure_time', type=int, default=10,
                       help='Scout exposure time in seconds for sky brightness measurement (default: 10)')
    
    # Tuning parameters
    parser.add_argument('--frames_per_light', type=int, default=3,
                       help='Number of light frames per exposure time (default: 3)')
    parser.add_argument('--frames_per_dark', type=int, default=5,
                       help='Number of dark frames for master dark (default: 5)')
    parser.add_argument('--target_noise_ratio', type=float, default=10.0,
                       help='Target noise ratio for optimal exposure (default: 10.0)')
    parser.add_argument('--fwhm_degradation_threshold', type=float, default=1.1,
                       help='FWHM degradation threshold (1.1 = 10%% degradation allowed, default: 1.1)')
    
    # Focusing parameters (demo simulation)
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
    
    # Create and run demo calculator
    calculator = DemoSubExposureCalculator(args)
    calculator.run()


if __name__ == '__main__':
    main() 