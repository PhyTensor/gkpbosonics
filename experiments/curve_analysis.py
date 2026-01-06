import numpy as np
import pandas as pd

class Curve:
    """Encapsulates data for a single dataset/line."""
    def __init__(self, x, y, label):
        self.x = np.array(x)
        self.y = np.array(y)
        self.label = label

class MaximaExtractor:
    """Logic for identifying the peak."""
    def find_peak(self, curve):
        # Finds the index of the maximum y value
        idx = np.argmax(curve.y)
        return curve.x[idx], curve.y[idx]

class Analyzer:
    """
    Main controller.
    Uses Composition: It 'has' curves and an extractor, rather than inheriting them.
    """
    def __init__(self):
        self.curves = []
        self._extractor = MaximaExtractor()

    def add_data(self, x, y, label):
        """Load raw data arrays here."""
        self.curves.append(Curve(x, y, label))

    def run_analysis(self):
        results = []
        for curve in self.curves:
            x_max, y_max = self._extractor.find_peak(curve)
            results.append({
                "Label": curve.label,
                "Peak_X": x_max,
                "Peak_Y": y_max
            })
        return pd.DataFrame(results)

# --- Example Usage ---
if __name__ == "__main__":
    # 1. Initialize Analyzer
    analysis = Analyzer()

    # 2. Load your raw data (Example: assume you have CSVs or arrays)
    # Replace these dummy arrays with your actual file loading logic
    # Example: df = pd.read_csv('data.csv')

    # Dummy data for demonstration
    x_vals = np.linspace(4, 20, 100)
    y_vals_1 = -0.005 * (x_vals - 9)**2 + 1.0  # Simulated eta=1.0
    y_vals_2 = -0.005 * (x_vals - 6)**2 + 0.8  # Simulated eta=0.7

    analysis.add_data(x_vals, y_vals_1, "eta = 1.0")
    analysis.add_data(x_vals, y_vals_2, "eta = 0.7")

    # 3. Get Results
    peaks = analysis.run_analysis()

    print("--- Exact Maxima Extracted ---")
    print(peaks)


