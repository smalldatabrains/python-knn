# Python K-Nearest Neighbors Implementation

A Python implementation of the K-Nearest Neighbors (KNN) algorithm for supervised classification. This project includes both a custom KNN implementation and data analysis tools.

## Features

- Custom KNN algorithm implementation
- Data normalization and preprocessing
- Model accuracy evaluation
- Comprehensive data analysis tools including:
  - Statistical descriptions
  - Missing value detection
  - Box plots for feature visualization
  - Correlation analysis with heatmaps

## Dependencies

- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Seaborn
- SciPy

## Project Structure

```
python-knn/
├── data/
│   └── data.csv
├── knn.py        # Main KNN implementation
├── knnsci.py     # Additional scientific computing utilities
└── README.md
```

## Usage

1. Place your dataset in the `data/` directory as `data.csv`
2. The data should be semicolon-separated (;) with features and a 'left' target column
3. Run the main script:

```python
python knn.py
```

## Implementation Details

The KNN algorithm is implemented with the following key functions:

- `distance(ind1, ind2)`: Calculates Euclidean distance between two points
- `kdistance(new_example, labelled_data)`: Computes distances between a new example and all training data
- `knn(k, kdistance_table)`: Performs k-nearest neighbor classification
- `accuracy(X_test, y_test, X_train)`: Evaluates model accuracy

## Data Analysis

The project includes comprehensive data analysis tools that can be enabled by setting `analysis=True`. This will generate:

- Statistical descriptions of features
- Box plots for feature visualization
- Correlation heatmaps
- Various distribution histograms

## Visualization

The repository includes several visualization outputs:
- TSNE visualizations (tsne.png, tsne2.png)
- 3D TSNE plots (tsne3d.png, tsne3d2.png)

## License

This project is open source and available for educational and research purposes.

## Contributing

Feel free to open issues or submit pull requests for improvements or bug fixes.
