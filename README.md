# Sub-Exposure Calculator v6.1

A fully automated, guided, and self-configuring optimal sub-exposure calculator for astrophotography. This intelligent command-line Python script automates finding the optimal sub-exposure time for your imaging setup using the Intelligent Scout method with optional high-precision refinement.

## Key Features

### Automated Optimal Target Finding

The script can now automatically find a suitable observation target near the zenith that is free of bright stars. The algorithm prioritizes targets east of zenith to maximize imaging time, as eastern targets stay above the horizon longer. It also avoids targets that would cause meridian flips during the experiment, preventing costly interruptions. This leads to higher-quality data by preventing bright stars from saturating the sensor and skewing the sky background measurements. This feature requires an internet connection and the astroquery library.

### Zenith Tracking

The system automatically tracks the zenith position as time passes and adjusts the telescope position to maintain optimal target selection. This prevents the telescope from drifting away from the zenith due to Earth's rotation during long experiments.

### Auto-Discovery

Automatically detects connected INDI devices (camera, mount, filter wheel, guide camera)

### Intelligent Scout Method

Uses a short test exposure to predict optimal times and generate focused test ranges

### High-Precision Refinement

Optional bracketing search refinement for maximum image quality

### Guided Calibration

Interactively guides you through creating missing Master Dark frames

### Smart Guiding

Prefers PHD2 or EKOS if running, falls back to built-in guiding

### Automatic Guiding

Starts guiding after slewing to optimal target for accurate long exposures

### Advanced Filter-Aware Focusing

Intelligently manages focus positions for each filter while minimizing unnecessary autofocus operations. Integrates with EKOS when available and provides robust fallback using internal HFD (Half Flux Diameter) autofocus.

### Efficient Experiment

Executes the core experiment in an optimized order to minimize mechanical operations

### Sensible Defaults

Uses logical defaults for common parameters

### Intelligent Analysis

Uses sigma-clipped statistics for robust sky background measurement

### Configurable Analysis

YAML-based configuration for fine-tuning analysis parameters

### Multi-Filter Support

Calculates optimal exposure times for each filter in your filter wheel

### Image Quality Analysis

Measures star FWHM and eccentricity to balance SNR with image quality

### Dynamic Seeing Adjustment

Automatically adjusts the number of test frames based on real-time seeing conditions measured from scout images. The script measures the Full Width at Half Maximum (FWHM) of stars in the scout image and dynamically adjusts the number of frames per exposure time to ensure optimal statistical accuracy:

- **Excellent Seeing (FWHM < 2.0 pixels)**: 3 frames - Fewer frames needed due to stable conditions
- **Good Seeing (FWHM 2.0-3.0 pixels)**: 5 frames - Standard number of frames for typical conditions  
- **Average Seeing (FWHM 3.0-4.0 pixels)**: 7 frames - More frames for statistical robustness
- **Poor Seeing (FWHM ≥ 4.0 pixels)**: 10 frames - Maximum frames for challenging conditions

This feature ensures that the experiment adapts to current atmospheric conditions, providing more reliable results in variable seeing while optimizing experiment duration in excellent conditions.

## Installation

### Linux (Recommended)
1. Clone this repository:
```bash
git clone <repository-url>
cd subDurationCalculator
```

2. Install system dependencies (required for dbus-python compilation):
```bash
sudo apt-get update
sudo apt-get install -y libdbus-1-dev libglib2.0-dev pkg-config
```

3. Install Python dependencies:
```bash
pip install -r requirements.txt
```

The astroquery library will be installed automatically and is required for the optimal target finding feature.

4. Ensure your INDI server is running and devices are connected.

### macOS Installation

For macOS users, there are several installation options:

#### Option 1: Demo Mode (Recommended for macOS)
The easiest way to get started on macOS is to use the demo mode, which simulates the full functionality without requiring INDI devices.

1. **Install macOS-compatible dependencies:**
   ```bash
   pip install -r requirements_macos.txt
   ```

2. **Test the installation:**
   ```bash
   python test_installation.py
   ```

3. **Run the demo mode:**
   ```bash
   python demo_mode.py
   ```

#### Option 2: Full INDI Installation (Advanced)
If you need to control actual INDI devices on macOS, you can install the full INDI stack. This is more complex and requires additional tools.

1. **Install Xcode Command Line Tools:**
   ```bash
   xcode-select --install
   ```

2. **Install Homebrew dependencies:**
   ```bash
   brew install dbus pkg-config cmake swig
   ```

3. **Install INDI from source:**
   ```bash
   # Clone INDI repository
   git clone https://github.com/indilib/indi.git
   cd indi
   
   # Build and install
   mkdir build && cd build
   cmake -DCMAKE_INSTALL_PREFIX=/usr/local ..
   make -j$(nproc)
   sudo make install
   ```

4. **Install pyindi-client:**
   ```bash
   pip install pyindi-client
   ```

#### Option 3: Docker/Linux VM (Alternative)
For the most reliable INDI experience, consider running the calculator in a Linux environment:

**Using Docker:**
```dockerfile
FROM ubuntu:22.04

RUN apt-get update && apt-get install -y \
    python3-pip python3-dev \
    libindi-dev libindi1 \
    swig cmake build-essential

WORKDIR /app
COPY . .
RUN pip3 install -r requirements.txt

CMD ["python3", "sub_exposure_calculator.py"]
```

**Build and run:**
```bash
docker build -t sub-exposure-calculator .
docker run -it --network host sub-exposure-calculator
```

## Quick Start

Get your Sub-Exposure Calculator running in 5 minutes!

### Step 1: Test Installation
```bash
python test_installation.py
```

### Step 2: Start Your INDI Server

Make sure your INDI server is running and your devices are connected. You can use:
- INDI Control Panel
- KStars with INDI
- Or any other INDI client

### Step 3: Run the Calculator

#### Basic Run (Auto-discovery)
```bash
python sub_exposure_calculator.py
```

#### With Custom Settings
```bash
python sub_exposure_calculator.py \
    --scout_exposure_time 60 \
    --frames_per_light 3 \
    --target_noise_ratio 8.0 \
    --refine_exposure \
    --refinement_steps 3 \
    --refinement_step_size 10
```

### Step 4: Follow the Prompts

The calculator will guide you through:
1. **Device Selection** (if multiple devices found)
2. **Calibration Setup** - Cover telescope once, then all darks are captured automatically
3. **Experiment** - Uncover telescope when ready
4. **Results** - View optimal exposure times

## Usage

### Basic Usage
```bash
python sub_exposure_calculator.py
```

### Advanced Usage

```bash
python sub_exposure_calculator.py \
    --host 192.168.1.100 \
    --port 7624 \
    --scout_exposure_time 30 \
    --frames_per_light 3 \
    --target_noise_ratio 8.0 \
    --refine_exposure \
    --refinement_steps 3 \
    --refinement_step_size 10
```

### Configuration File Usage

The calculator automatically loads configuration from YAML files in this order:
1. `config.yaml` (your local configuration)
2. `config_local.yaml` (local overrides)
3. `config_example.yaml` (fallback example)

To use custom analysis settings:
1. Copy `config_example.yaml` to `config.yaml`
2. Modify the analysis parameters as needed
3. Run the script - it will automatically load your configuration

Example configuration:
```yaml
analysis:
  sky_region_fraction: 0.9    # Use 90% of image for analysis
  sigma_clip_threshold: 2.5   # More aggressive outlier removal
  min_valid_pixels: 2000      # Require more pixels for reliability
```

### Command Line Options

#### Connection Settings
- `--host TEXT`: INDI server hostname (default: localhost)
- `--port INTEGER`: INDI server port (default: 7624)

#### Paths & Sequences

- `--calibration_path PATH`: Path for calibration frames (default: ./calibration)
- `--scout_exposure_time INTEGER`: Scout exposure time in seconds for sky brightness measurement (default: 60)

#### Device Overrides

- `--camera_name TEXT`: Override camera device name
- `--mount_name TEXT`: Override mount device name
- `--filter_wheel_name TEXT`: Override filter wheel device name
- `--guide_camera_name TEXT`: Override guide camera device name
- `--focuser_name TEXT`: Override focuser device name

#### Tuning Parameters

- `--frames_per_light INTEGER`: Number of light frames per exposure time (default: 5)
- `--frames_per_dark INTEGER`: Number of dark frames for master dark (default: 20)
- `--target_noise_ratio FLOAT`: Target noise ratio for optimal exposure (default: 10.0)
- `--fwhm_degradation_threshold FLOAT`: FWHM degradation threshold (1.1 = 10% degradation allowed, default: 1.1)
- `--skip_dark_library`: Skip dark frame library creation (use existing darks only)

#### Guiding Parameters

- `--guide_exposure_time FLOAT`: Guide camera exposure time in seconds (default: 2.0)
- `--guide_calibration_time FLOAT`: Guide calibration time in seconds (default: 5.0)
- `--guide_max_pulse INTEGER`: Maximum guide pulse duration in milliseconds (default: 1000)
- `--guide_aggressiveness FLOAT`: Guide aggressiveness factor 0.0-1.0 (default: 0.5)
- `--guide_dither_threshold FLOAT`: Guide dither threshold in pixels (default: 0.5)

#### Refinement Phase Parameters

- `--refine_exposure`: Enable the high-precision refinement phase to find the optimal exposure based on image quality
- `--refinement_steps INTEGER`: Number of granular steps to test on each side of the initial optimal exposure (default: 3)
- `--refinement_step_size INTEGER`: The duration in seconds of each granular step during the refinement phase (default: 15)

#### Focusing Parameters

- `--focus_temp_threshold FLOAT`: Temperature threshold in °C for refocusing (default: 1.0)

## Configuration Parameters

The calculator can be configured using a YAML configuration file. Copy `config_example.yaml` to `config.yaml` and modify as needed. Below is a comprehensive explanation of all parameters:

### INDI Server Configuration
```yaml
indi:
  host: localhost          # INDI server hostname
  port: 7624              # INDI server port
  timeout: 30             # Connection timeout in seconds
```
**Purpose**: Controls connection to your INDI server.  
**When to modify**: Change `host` and `port` if your INDI server runs on a different machine or port. Increase `timeout` for slower network connections.

### Device Names

```yaml
devices:
  camera_name: ""          # e.g., "ZWO CCD ASI1600MM Pro"
  mount_name: ""           # e.g., "SkyWatcher EQ6-R Pro"
  filter_wheel_name: ""    # e.g., "ZWO EFW"
  guide_camera_name: ""    # e.g., "ZWO ASI120MM Mini"
  focuser_name: ""         # e.g., "ZWO Focuser"
```
**Purpose**: Override automatic device discovery with specific device names.  
**When to modify**: Use when auto-discovery fails or you want to ensure specific devices are used. Leave empty for auto-discovery.

### Experiment Parameters

```yaml
experiment:
  scout_exposure_time: 60     # seconds for initial sky brightness measurement
  exposure_times: [60, 120, 180, 240, 300, 420, 600]  # seconds (for demo mode)
  frames_per_light: 5         # Number of frames per exposure time
  frames_per_dark: 20         # Number of dark frames for master dark
  target_noise_ratio: 10.0    # Target SNR for optimal exposure
  fwhm_degradation_threshold: 1.1  # 1.1 = 10% degradation allowed
  skip_dark_library: false    # Skip dark frame library creation
```

**scout_exposure_time**: Duration of the initial test exposure used to measure sky brightness and predict optimal exposure times.  
**When to modify**: 
- **Shorter (30-45s)**: For bright skies or fast optics (f/4 or faster)
- **Longer (90-120s)**: For dark skies or slow optics (f/8 or slower)

**frames_per_light**: Initial number of exposures taken at each test exposure time. This value serves as a default or fallback if seeing conditions cannot be measured from the scout image. The script automatically adjusts this value based on real-time seeing conditions measured from the scout image.

**When to modify**:
- **Lower (3-5)**: For faster testing when you expect good seeing conditions
- **Higher (7-10)**: For more robust results when you expect variable or poor seeing conditions
- **Note**: The script will automatically override this value based on measured seeing conditions, so this serves as a baseline setting

**frames_per_dark**: Number of dark frames used to create master dark frames.  
**When to modify**:
- **Fewer (10-15)**: For faster calibration (acceptable for most cameras)
- **More (30-50)**: For maximum calibration quality (recommended for noisy cameras)

**target_noise_ratio**: Target signal-to-noise ratio for optimal exposure calculation.  
**When to modify**:
- **Lower (5-8)**: For faster exposures, acceptable for bright objects
- **Higher (12-15)**: For maximum quality, recommended for faint objects

**fwhm_degradation_threshold**: How much star sharpness degradation is allowed.  
**When to modify**:
- **Lower (1.05-1.08)**: For maximum image quality, stricter requirements
- **Higher (1.15-1.2)**: For better SNR, allows more degradation

### Refinement Phase Parameters

```yaml
refinement:
  enabled: false          # Enable high-precision refinement phase
  steps: 3               # Number of granular steps on each side
  step_size: 15          # Duration in seconds of each step
```

**enabled**: Whether to run the high-precision refinement phase after the initial scout.  
**When to modify**: Enable for maximum precision in finding optimal exposure times.

**steps**: Number of exposure times to test on each side of the predicted optimal time.  
**When to modify**:
- **Fewer (2)**: For faster refinement (less precise)
- **More (4-5)**: For maximum precision (slower)

**step_size**: Time increment between refinement test exposures.  
**When to modify**:
- **Smaller (10s)**: For fast optics (f/4 or faster) or precise requirements
- **Larger (20-30s)**: For slow optics (f/8 or slower) or faster testing

### Analysis Settings

```yaml
analysis:
  sky_region_fraction: 0.8    # Use 80% of image for sky analysis
  sigma_clip_threshold: 3.0   # Sigma clipping threshold
  min_valid_pixels: 1000      # Minimum pixels for valid analysis
```

**sky_region_fraction**: Fraction of the image to use for sky background analysis. Focuses analysis on the central portion to avoid edge artifacts and vignetting.  
**When to modify**:
- **Lower (0.6-0.7)**: For heavily vignetted images or dense star fields
- **Standard (0.8)**: For well-corrected optics and typical conditions
- **Higher (0.9)**: For sparse star fields with large empty areas
- **Maximum (1.0)**: Use entire image (not recommended due to edge effects)

**sigma_clip_threshold**: Threshold for sigma clipping to remove stars and outliers from sky analysis. Controls how aggressive the outlier rejection is.  
**When to modify**:
- **Lower (2.0-2.5)**: For very clean images with few stars or aggressive outlier removal
- **Standard (3.0-3.5)**: For typical imaging conditions
- **Higher (4.0-5.0)**: For crowded star fields or when you want to preserve more sky pixels

**min_valid_pixels**: Minimum number of pixels required for valid sky analysis. Ensures sufficient data for reliable statistical analysis.  
**When to modify**:
- **Lower (500-1000)**: For small sensors or dense star fields
- **Standard (1000-5000)**: For typical camera sensors
- **Higher (5000+)**: For large sensors or high-resolution cameras requiring maximum statistical reliability

### Camera Settings
```yaml
camera:
  default_gain: 1.0           # Default camera gain
  default_read_noise: 5.0     # Default read noise in electrons
  image_format: "FITS"        # Image format
  binning: [1, 1]             # Binning mode [x, y]
```

**default_gain**: Default gain setting for the camera.  
**When to modify**: Set to your typical gain setting for accurate calculations.

**default_read_noise**: Default read noise in electrons.  
**When to modify**: Use measured read noise for your specific camera and gain setting.

### Location Settings
```yaml
location:
  latitude: 40.7128  # degrees
  longitude: -74.0060  # degrees
  elevation: 10  # meters
```

**Purpose**: Required for optimal target finding feature. Provides telescope location for zenith calculations and astronomical database queries.  
**When to modify**: Set to your actual telescope location coordinates. This is required for the optimal target finding feature to work.

### Mount Settings
```yaml
mount:
  slew_rate: "SLEW_GUIDE"     # Slew rate for mount movements
  tracking_rate: "TRACK_SIDEREAL"  # Tracking rate
  settle_time: 10             # Settle time after slew in seconds
  slewing:
    wait_time: 15  # seconds to wait for mount to arrive and settle after slew command
  zenith_tracking:
    enabled: true  # Enable automatic zenith tracking during experiments
    interval: 30  # minutes - recalculate zenith every 30 minutes
    threshold: 2.0  # degrees - slew if drift exceeds 2°
  target_selection:
    eastern_preference_weight: 0.3  # Weight for eastern preference (0.0-1.0)
    separation_quality_weight: 0.5  # Weight for separation quality (0.0-1.0)
    meridian_flip_avoidance_weight: 0.2  # Weight for meridian flip avoidance (0.0-1.0)
```

**settle_time**: Time to wait after slewing before starting exposures.  
**When to modify**:
- **Shorter (5s)**: For stable mounts with good damping
- **Longer (15-20s)**: For mounts that need more time to settle

**slewing.wait_time**: Time to wait for mount to arrive and settle after sending slew command.  
**When to modify**:
- **Shorter (10s)**: For fast mounts or when you want to minimize wait time
- **Longer (20-30s)**: For slower mounts or when you want to ensure complete settling

**zenith_tracking.enabled**: Whether to automatically track zenith position during experiments.  
**When to modify**: Disable if you want to manually control telescope positioning.

**zenith_tracking.interval**: How often to check for zenith drift (in minutes).  
**When to modify**:
- **Shorter (15-20min)**: For more frequent position updates
- **Longer (45-60min)**: For less frequent updates (fewer slews)

**zenith_tracking.threshold**: Maximum allowed drift from zenith before slewing (in degrees).  
**When to modify**:
- **Lower (1-1.5°)**: For more precise positioning
- **Higher (3-5°)**: For less frequent slews

**target_selection.eastern_preference_weight**: Weight for eastern preference in target selection (0.0-1.0).  
**When to modify**:
- **Lower (0.1-0.2)**: Minimal eastern bias, prioritize dark patches
- **Higher (0.4-0.5)**: Strong eastern bias, maximize imaging time
- **Default (0.3)**: Balanced approach

**target_selection.separation_quality_weight**: Weight for separation from bright stars (0.0-1.0).  
**When to modify**: Usually set to complement other weights (sum should be ~1.0)

**target_selection.meridian_flip_avoidance_weight**: Weight for avoiding meridian flips (0.0-1.0).  
**When to modify**:
- **Lower (0.1)**: Minimal meridian flip avoidance, prioritize other factors
- **Higher (0.3-0.4)**: Strong meridian flip avoidance, prevent interruptions
- **Default (0.2)**: Balanced approach

### Filter Wheel Settings
```yaml
filter_wheel:
  settle_time: 2              # Settle time after filter change
  default_filters: ["Luminance", "Red", "Green", "Blue", "Ha", "OIII", "SII"]
```

**settle_time**: Time to wait after changing filters.  
**When to modify**:
- **Shorter (1s)**: For filter wheels with fast mechanisms
- **Longer (3-5s)**: For filter wheels that need more time to settle

### Guiding Settings
```yaml
guiding:
  exposure_time: 2.0          # Guide camera exposure time
  calibration_time: 5.0       # Guide calibration time
  max_pulse: 1000             # Maximum guide pulse in milliseconds
  aggressiveness: 0.5         # Guide aggressiveness (0.0-1.0)
  dither_threshold: 0.5       # Dither threshold in pixels
```

**exposure_time**: Guide camera exposure duration.  
**When to modify**:
- **Shorter (1-1.5s)**: For bright guide stars or fast guide cameras
- **Longer (3-5s)**: For faint guide stars or slow guide cameras

**aggressiveness**: How aggressively the guiding system corrects for drift.  
**When to modify**:
- **Lower (0.3-0.4)**: For stable mounts or conservative guiding
- **Higher (0.6-0.7)**: For less stable mounts or aggressive guiding

**dither_threshold**: Minimum drift before applying correction.  
**When to modify**:
- **Lower (0.3)**: More sensitive to small drifts
- **Higher (0.8)**: Less sensitive, reduces unnecessary corrections

### Focusing Settings
```yaml
focusing:
  temperature_threshold: 1.0  # Temperature threshold in °C for refocusing
  focus_range: 1000           # Focus range in steps
  step_size: 50               # Focus step size
  test_exposure_time: 5       # Test exposure time for HFD measurement
  settle_time: 1              # Settle time after focus movement
```

**temperature_threshold**: Temperature change threshold that triggers refocusing.  
**When to modify**:
- **Lower (0.5°C)**: More frequent refocusing for critical applications
- **Higher (2.0°C)**: Less frequent refocusing for faster operation

**focus_range**: Total range of focus movement during autofocus.  
**When to modify**:
- **Smaller (500)**: For precise focusers or when you know approximate position
- **Larger (2000)**: For less precise focusers or when starting from unknown position

**step_size**: Size of each focus step during autofocus.  
**When to modify**:
- **Smaller (25)**: For more precise focus finding (slower)
- **Larger (100)**: For faster focus finding (less precise)

### Output Settings
```yaml
output:
  save_raw_frames: false      # Save raw uncalibrated frames
  save_calibrated_frames: false  # Save calibrated frames
  save_analysis_plots: true   # Save analysis plots
  verbose_logging: true       # Enable verbose logging
```

**save_raw_frames**: Whether to save uncalibrated light frames.  
**When to modify**: Enable for debugging or if you want to keep raw data.

**save_calibrated_frames**: Whether to save calibrated light frames.  
**When to modify**: Enable if you want to keep the calibrated test images.

**save_analysis_plots**: Whether to generate and save analysis plots.  
**When to modify**: Enable for detailed analysis of the results.

## How It Works

### Phase 0: Device Discovery and Setup
1. Connects to your INDI server
2. Automatically discovers connected devices (camera, mount, filter wheel, guide camera, focuser)
3. Prompts for selection if multiple devices of the same type are found
4. Gets filter names from the filter wheel
5. Loads telescope location from configuration for optimal target finding

### Phase 1: Dark Frame Library Creation
1. Creates a comprehensive dark frame library (10s to 1800s)
2. Enables unattended operation throughout the experiment
3. Automatically selects and scales dark frames as needed

### Phase 2: Intelligent Scout Experiment
For each filter:
1. **Phase A: Intelligent Scout & Prediction**
   - Takes scout exposure to measure sky brightness
   - Calculates sky flux rate (ADU/sec)
   - Predicts optimal exposure time based on camera properties
   
2. **Phase B: Dynamic Test Range Generation**
   - Creates focused test range around predicted optimal time
   - Ensures reasonable minimum and maximum exposure times
   
3. **Phase C: Dynamic Seeing Adjustment**
   - Measures star FWHM from scout image to assess seeing conditions
   - Automatically adjusts frames_per_light based on seeing quality
   - Ensures optimal statistical accuracy for current conditions
   
4. **Phase D: Automatic Dark Frame Selection**
   - Selects appropriate dark frames from library
   - Scales dark frames if needed for different exposure times
   
5. **Phase E: Focused Experiment Execution**
   - Finds and slews to optimal target near zenith (free of bright stars)
   - Monitors zenith drift and adjusts position as needed during experiment
   - Performs filter-aware focusing (simple re-focus or full autofocus)
   - Captures multiple frames at each test exposure time
   - Analyzes sky background and image quality (FWHM, eccentricity)
   - Calculates optimal exposure balancing SNR with image quality

### Phase 4: Optional Refinement Phase
If enabled:
1. **High-Precision Bracketing Search**
   - Tests granular exposure times around initial optimal time
   - Measures image quality metrics for each exposure
   - Finds true minimum FWHM and optimal quality threshold
   
2. **Quality-Based Optimization**
   - Balances signal-to-noise ratio with image quality
   - Selects longest exposure that maintains quality standards

### Phase 5: Results
1. Displays a formatted table with optimal exposure times for each filter
2. Shows camera properties, target ADU values, and quality metrics
3. Provides detailed analysis of both scout and refinement phases

## Example Output

```
📊 SUB-EXPOSURE CALCULATOR RESULTS
================================================================================
Filter          Optimal Exp (s) Refined    Gain     Read Noise   Target ADU   Scout FWHM   Median FWHM  Median Ecc
--------------------------------------------------------------------------------
Luminance       120.0           Yes        1.00     5.20         2704.0       2.15         2.45         0.12
Red             180.0           Yes        1.02     5.15         2650.0       2.67         2.67         0.15
Green           150.0           Yes        0.98     5.25         2756.0       2.52         2.52         0.13
Blue            200.0           Yes        1.01     5.18         2689.0       2.89         2.89         0.18
Ha              420.0           Yes        1.00     5.20         2704.0       3.12         3.12         0.22
OIII            360.0           Yes        1.00     5.20         2704.0       2.98         2.98         0.19
SII             480.0           Yes        1.00     5.20         2704.0       3.25         3.25         0.25
--------------------------------------------------------------------------------

🔎 REFINEMENT PHASE SUMMARY:
  • Filters with refinement: Luminance, Red, Green, Blue, Ha, OIII, SII
  • Refinement steps: 3 on each side
  • Refinement step size: 15s
  • High-precision bracketing search completed for optimal image quality

💡 Recommendations:
  • Optimal exposure times now balance signal-to-noise with star quality (FWHM)
  • FWHM values indicate star sharpness - lower is better
  • Eccentricity values indicate tracking quality - lower is better
  • Consider your mount's tracking accuracy when using longer exposures
  • You may need to adjust based on your specific imaging conditions
```

## Technical Details

### Optimal Target Finding Algorithm
The calculator uses an intelligent approach to find optimal observation targets:

1. **Zenith Calculation**: Calculates current zenith coordinates based on telescope location and time
2. **Bright Star Query**: Queries Gaia database for stars brighter than G=9.0 within 5° of zenith
3. **Grid Search**: Searches a 10x10 grid around zenith for the darkest patch
4. **Eastern Preference Analysis**: Prioritizes targets east of zenith for maximum imaging time
5. **Meridian Flip Risk Assessment**: Calculates risk of meridian flips during experiment duration
6. **Combined Scoring**: Balances separation quality, eastern preference, and meridian flip avoidance using configurable weights
7. **Optimal Slew**: Commands mount to slew to the optimal dark patch coordinates

### Zenith Tracking Algorithm
The system continuously monitors and adjusts the telescope position to maintain optimal target selection:

1. **Drift Monitoring**: Checks for zenith drift every configurable interval (default: 30 minutes)
2. **Threshold Detection**: Detects when drift exceeds configurable threshold (default: 2°)
3. **Position Update**: Recalculates zenith coordinates and optimal target position
4. **Smart Recalculation**: Only recalculates optimal target if zenith has moved significantly (>1°)
5. **Automatic Slew**: Commands mount to slew to updated optimal position

### Intelligent Scout Algorithm
The calculator uses an advanced approach to determine optimal exposure time:

1. **Scout Phase**: Takes a short test exposure to measure sky brightness and flux rate
2. **Seeing Assessment**: Measures star FWHM from the scout image to assess seeing conditions
3. **Dynamic Frame Adjustment**: Automatically adjusts frames_per_light based on seeing quality
4. **Prediction**: `Predicted Time = Target ADU / Sky Flux Rate`
5. **Dynamic Range**: Generates focused test range around prediction (0.6x to 1.5x predicted time)
6. **Quality Analysis**: Measures star FWHM and eccentricity for image quality assessment
7. **Dual Optimization**: Balances noise requirements with image quality constraints

### Eastern Preference Logic
The system prioritizes targets east of zenith to maximize imaging time:

- **Earth's Rotation**: As Earth rotates, targets east of zenith stay above horizon longer
- **Imaging Time**: Eastern targets provide more time for imaging before setting
- **Configurable Balance**: Users can adjust the trade-off between eastern preference and dark patch quality
- **Smart Fallback**: If no suitable eastern target exists, falls back to western targets

### Meridian Flip Avoidance Logic
The system avoids targets that would cause meridian flips during experiments:

- **Meridian Flips**: Costly operations that can take 5-15 minutes and require recalibration
- **Risk Calculation**: Analyzes target's hour angle throughout experiment duration
- **Safety Buffer**: Uses 2-hour buffer from meridian for safe operation
- **Duration Estimation**: Calculates expected experiment duration based on exposure times and filters
- **Risk Reporting**: Provides clear warnings when meridian flip risk is detected

### Advanced Focusing System
The calculator includes an intelligent, filter-aware focusing system:

1. **EKOS Integration**: Leverages EKOS focusing capabilities when available
2. **Filter-Specific Cache**: Maintains optimal focus positions for each filter
3. **Temperature Monitoring**: Only refocuses when temperature changes significantly
4. **HFD Autofocus**: Uses Half Flux Diameter measurement for precise focus
5. **Efficient Operations**: Minimizes unnecessary autofocus operations

### Smart Guiding Integration
The system intelligently integrates with external guiding software:

1. **PHD2 Detection**: Automatically detects and uses PHD2 if running
2. **EKOS Detection**: Detects and uses EKOS guiding if available
3. **Built-in Fallback**: Provides robust built-in guiding when external software not available
4. **Priority System**: PHD2 > EKOS > Built-in guiding
5. **Seamless Operation**: No manual configuration required

### Target ADU Calculation
```
Target ADU = (target_noise_ratio × read_noise)² / gain
```

### Quality Threshold Calculation
```
Quality Threshold = Minimum FWHM × fwhm_degradation_threshold
```

### Supported File Formats
- Input/Output: FITS format using AstroPy
- Calibration frames are automatically saved and reused

### Error Handling
- Comprehensive error handling for INDI connection issues
- Graceful handling of missing devices
- User-friendly error messages and recovery options

## Troubleshooting

### Common Issues

1. **INDI Connection Failed**
   - Ensure your INDI server is running
   - Check host and port settings
   - Verify network connectivity

2. **No Devices Found**
   - Ensure devices are connected and powered on
   - Check INDI server configuration
   - Verify device drivers are loaded

3. **Long Exposure Warnings**
   - Consider using higher gain settings
   - Accept lower signal-to-noise ratio
   - Check mount tracking accuracy

4. **Calibration Issues**
   - Ensure telescope is properly covered for darks
   - Check for light leaks
   - Verify camera temperature stability

5. **Poor Image Quality Results**
   - Check mount tracking and guiding performance
   - Verify seeing conditions
   - Consider shorter exposures or better guiding

6. **Optimal Target Finding Issues**
   - Ensure internet connection is available for Gaia database queries
   - Verify telescope location is correctly set in configuration
   - Check that astroquery library is installed (`pip install astroquery`)
   - If queries fail, the script will fall back to zenith slewing

7. **Zenith Tracking Issues**
   - Verify telescope location is correctly set in configuration
   - Check that zenith tracking is enabled in mount settings
   - Adjust tracking interval if slews are too frequent or infrequent
   - Monitor mount performance during automatic slews

8. **Meridian Flip Issues**
   - Check meridian flip risk warnings in output
   - Consider reducing experiment duration if risk is high
   - Adjust meridian flip avoidance weight in configuration
   - Verify telescope location accuracy for proper hour angle calculations

9. **Focusing Issues**
   - Check focuser is properly connected and calibrated
   - Verify temperature sensor is working
   - Test EKOS connectivity if using external focusing
   - Adjust focus range and step size for your optical system

10. **Guiding Issues**
    - Check guide camera is connected and working
    - Verify mount supports ST4 or pulse guiding
    - Test PHD2 or EKOS connectivity if using external guiding
    - Adjust guiding parameters for your setup

### Performance Tips

- Use refinement phase for maximum precision
- Adjust `frames_per_light` based on seeing conditions
- Modify `target_noise_ratio` based on your quality requirements
- Consider your mount's tracking accuracy when interpreting results
- Use appropriate `scout_exposure_time` for your sky conditions
- Optimal target finding works best with stable internet connection
- Adjust `mount.slewing.wait_time` based on your mount's speed and settling characteristics
- Zenith tracking interval should balance position accuracy with mount wear
- Consider your mount's slewing speed when setting zenith tracking parameters
- Meridian flip avoidance is crucial for uninterrupted imaging sessions
- Adjust meridian flip avoidance weight based on your mount's flip time and recalibration needs
- Temperature threshold for focusing should match your environment's stability
- Focus range should be appropriate for your optical system's depth of focus
- Guiding aggressiveness should match your mount's stability characteristics