import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Simulated tracking data (from tracking_v8_ultralytics.py output)
tracking_data = [
    {'track_id': 1, 'object_size': 5000, 'confidence': 0.95, 'velocity': 10, 'position_change': 5},
    {'track_id': 2, 'object_size': 3000, 'confidence': 0.80, 'velocity': 15, 'position_change': 8},
    {'track_id': 3, 'object_size': 7000, 'confidence': 0.65, 'velocity': 25, 'position_change': 12},
    {'track_id': 4, 'object_size': 2000, 'confidence': 0.55, 'velocity': 5, 'position_change': 2},
]

# Set up the fuzzy logic system
object_size = ctrl.Antecedent(np.arange(0, 10000, 1), 'object_size')
confidence = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'confidence')
velocity = ctrl.Antecedent(np.arange(0, 100, 1), 'velocity')
position_change = ctrl.Antecedent(np.arange(0, 100, 1), 'position_change')

# Output: search window size
search_window = ctrl.Consequent(np.arange(0, 500, 1), 'search_window')

# Manually define membership functions for inputs
object_size['small'] = fuzz.trimf(object_size.universe, [0, 0, 5000])
object_size['medium'] = fuzz.trimf(object_size.universe, [1000, 5000, 10000])
object_size['large'] = fuzz.trimf(object_size.universe, [5000, 10000, 10000])

confidence['low'] = fuzz.trimf(confidence.universe, [0, 0, 0.6])
confidence['medium'] = fuzz.trimf(confidence.universe, [0.3, 0.7, 1])
confidence['high'] = fuzz.trimf(confidence.universe, [0.6, 1, 1])

velocity['slow'] = fuzz.trimf(velocity.universe, [0, 0, 30])
velocity['medium'] = fuzz.trimf(velocity.universe, [10, 40, 70])
velocity['fast'] = fuzz.trimf(velocity.universe, [50, 100, 100])

position_change['small'] = fuzz.trimf(position_change.universe, [0, 0, 10])
position_change['medium'] = fuzz.trimf(position_change.universe, [5, 15, 30])
position_change['large'] = fuzz.trimf(position_change.universe, [20, 40, 50])

# Manually define membership functions for the output
search_window['small'] = fuzz.trimf(search_window.universe, [0, 0, 100])
search_window['medium'] = fuzz.trimf(search_window.universe, [50, 150, 250])
search_window['large'] = fuzz.trimf(search_window.universe, [200, 350, 500])

# Fuzzy rules
rule1 = ctrl.Rule(object_size['small'] & confidence['low'], search_window['large'])
rule2 = ctrl.Rule(velocity['fast'] & position_change['large'], search_window['small'])
rule3 = ctrl.Rule(velocity['medium'] & position_change['medium'], search_window['medium'])

# Control system
search_window_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
search_window_simulation = ctrl.ControlSystemSimulation(search_window_ctrl)

# Process tracking data and apply fuzzy logic
for data in tracking_data:
    track_id = data['track_id']
    object_size_value = data['object_size']
    confidence_value = data['confidence']
    velocity_value = data['velocity']
    position_change_value = data['position_change']

    # Set fuzzy input values
    search_window_simulation.input['object_size'] = object_size_value
    search_window_simulation.input['confidence'] = confidence_value
    search_window_simulation.input['velocity'] = velocity_value
    search_window_simulation.input['position_change'] = position_change_value

    # Compute fuzzy output
    search_window_simulation.compute()

    # Get the dynamic search window size
    dynamic_search_window = search_window_simulation.output['search_window']

    # Output the result for this track ID
    print(f"Tracking ID {track_id}:")
    print(f"Object Size: {object_size_value}, Confidence: {confidence_value}, Velocity: {velocity_value}, Position Change: {position_change_value}")
    print(f"Adjusted Search Window: {dynamic_search_window:.2f} pixels\n")
