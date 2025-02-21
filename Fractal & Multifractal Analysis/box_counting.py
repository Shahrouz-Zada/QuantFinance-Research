import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing S\&P 500 data
data = yf.download('^GSPC', start='2010-01-01', end='2021-12-31')

# Close prices
close_prices = data['Close']

# Box-Counting Method
def box_count(data, box_size):
    count = 0
    for i in range(0, len(data), box_size):
        if np.any(data[i:i+box_size]):
            count += 1
    return count

def fractal_dimension(data, max_box_size):
    sizes = np.arange(1, max_box_size)
    counts = [box_count(data, size) for size in sizes]
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    return -coeffs[0]

# Example usage
data_values = close_prices.values  # Use actual financial data
fractal_dim = fractal_dimension(data_values, 50)
print(f"Fractal Dimension: {fractal_dim}")

# Save the fractal dimension to the dataset
box_count_features = pd.DataFrame({'Fractal Dimension': [fractal_dim]})
box_count_features.to_csv('box_count_features.csv', index=False)

# R/S Analysis
def rs_analysis(data, window_sizes):
    rescaled_ranges = []
    for window in window_sizes:
        rescaled_range = []
        for i in range(0, len(data), window):
            if i + window > len(data):
                break
            segment = data[i:i+window]
            mean = np.mean(segment)
            Z = np.cumsum(segment - mean)
            R = max(Z) - min(Z)
            S = np.std(segment)
            rescaled_range.append(R / S)
        rescaled_ranges.append(np.mean(rescaled_range))
    return rescaled_ranges

# Example usage
window_sizes = range(10, 200, 10)
rescaled_ranges = rs_analysis(data_values, window_sizes)

plt.plot(window_sizes, rescaled_ranges, 'o-')
plt.xlabel('Window size')
plt.ylabel('Rescaled Range (R/S)')
plt.title('R/S Analysis of S\&P 500')
plt.show()

# Save the R/S analysis results
rs_features = pd.DataFrame({'Window Size': window_sizes, 
'Rescaled Range': rescaled_ranges})
rs_features.to_csv('rs_features.csv', index=False)
