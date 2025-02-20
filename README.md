# Irreverable

## Overview

Irreverable is a Python script designed to generate impulse responses (IRs) with specific characteristics. The script uses various parameters to create IRs with different durations, predelays, and echo densities. The generated IRs are saved as `.wav` files.

## Requirements

- Python 3.x
- NumPy
- Matplotlib
- SoundFile
- SciPy

## Configuration

The script allows you to configure the following parameters:

- `number_of_irs`: Number of impulse responses to generate.
- `min_duration`, `target_duration`, `max_duration`: Duration range for the IRs.
- `min_predelay`, `target_predelay`, `max_predelay`: Predelay range for the IRs.
- `tau0min`, `tau0max`, `tau0_target`: Parameters for the echo density curve.
- `sample_rate`: Sample rate for the generated IRs.
- `output_dir`: Directory to save the generated IRs.

## Usage

1. Install the required libraries:
    ```bash
    pip install numpy matplotlib soundfile scipy
    ```

2. Configure the parameters in the script as needed.

3. Run the script:
    ```bash
    python irreverable.py
    ```

4. The generated IRs will be saved in the specified `output_dir`.

## Example

Here is an example of how to configure and run the script:

```python
# Configuration
number_of_irs = 12
min_duration = 1
target_duration = 3 
max_duration = 5 
min_predelay = 0.01 
target_predelay = 0.1
max_predelay = 0.3
tau0min = 0.020 
tau0max = 0.070
tau0_target = 0.050
sample_rate = 48000
output_dir = "/path/to/output/directory"

# Run the script
# python irreverable.py
```

## Output

The script generates `.wav` files with the specified characteristics and saves them in the `output_dir`. Each file is named with the format `IR_length<duration>_<timestamp>.wav`.

## License

This project is licensed under the MIT License.
