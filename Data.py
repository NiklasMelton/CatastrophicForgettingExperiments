import numpy as np
import h5py
from sklearn.model_selection import train_test_split


def generate_overlap_data(
        num_points_per_class=100,
        side_length=0.1,
        spacing=0.2,
        random_state=None,
        test_size=0.2,
        shuffle=False,
):
    """
    Generate a synthetic dataset of three classes arranged along the diagonal line from (0,0) to (1,1).
    Each class is defined as a square region of the same side length, centered at equally spaced
    positions along the diagonal. Points are uniformly sampled inside these squares.

    After generation, the entire dataset is normalized so that all coordinates lie in [0,1].

    Parameters
    ----------
    num_points_per_class : int
        Number of data points to generate per class.
    side_length : float
        The side length of each square region.
    spacing : float
        The offset from the ends of the diagonal line for the first and last class.
        The classes will be placed at (spacing, spacing), (0.5, 0.5), and (1 - spacing, 1 - spacing).
    random_state : int or None
        Optional random seed for reproducibility.

    Returns
    -------
    X : np.ndarray of shape (3 * num_points_per_class, 2)
        The generated data points (normalized).
    y : np.ndarray of shape (3 * num_points_per_class,)
        The class labels (0, 1, 2).
    """
    if random_state is not None:
        np.random.seed(random_state)  # Set random seed for reproducibility

    # Define the centers of the three classes along the diagonal
    centers = [
        (0.5 - spacing, 0.5 - spacing),
        (0.5, 0.5),
        (0.5 + spacing, 0.5 + spacing)
    ]

    # Prepare arrays for data and labels
    X_list = []
    y_list = []

    half_side = side_length / 2.0
    for class_idx, (cx, cy) in enumerate(centers):
        # Uniformly sample points within the square for this class
        x_coords = np.random.uniform(cx - half_side, cx + half_side, size=num_points_per_class)
        y_coords = np.random.uniform(cy - half_side, cy + half_side, size=num_points_per_class)

        class_points = np.column_stack((x_coords, y_coords))  # Combine x and y coordinates
        class_labels = np.full(num_points_per_class, class_idx, dtype=int)  # Assign labels for the class

        X_list.append(class_points)  # Add class points to the list
        y_list.append(class_labels)  # Add class labels to the list

    # Concatenate all classes into single arrays
    X = np.vstack(X_list)
    y = np.hstack(y_list)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                        random_state=random_state,
                                                        stratify=y, shuffle=True)
    if not shuffle:
        # Sort data by class labels if shuffle is disabled
        sorted_indices = np.argsort(y_train)
        X_train = X_train[sorted_indices]
        y_train = y_train[sorted_indices]

        sorted_indices = np.argsort(y_test)
        X_test = X_test[sorted_indices]
        y_test = y_test[sorted_indices]

    # Ensure all values are within the range [0, 1]
    X_train = np.clip(X_train, 0.0, 1.0)
    X_test = np.clip(X_test, 0.0, 1.0)
    return X_train, y_train, X_test, y_test


def load_usps():
    """
    Load the USPS dataset from an HDF5 file.

    Returns
    -------
    X_tr : np.ndarray
        Training data.
    y_tr : np.ndarray
        Training labels.
    X_te : np.ndarray
        Testing data.
    y_te : np.ndarray
        Testing labels.
    """
    with h5py.File("usps.h5", 'r') as hf:
        train = hf.get('train')
        X_tr = train.get('data')[:]  # Training data
        y_tr = train.get('target')[:]  # Training labels
        test = hf.get('test')
        X_te = test.get('data')[:]  # Testing data
        y_te = test.get('target')[:]  # Testing labels

    return X_tr, y_tr, X_te, y_te


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Generate some synthetic overlap data
    X, y, _, _ = generate_overlap_data(num_points_per_class=200, side_length=0.4,
                                       spacing=0.2, random_state=42)

    # Plot the generated data
    fig, ax = plt.subplots(figsize=(6, 6))
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.7, edgecolors='k')
    ax.set_title("Three Classes Along the Diagonal")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Add a legend for the classes
    legend_labels = ["Class 0", "Class 1", "Class 2"]
    handles, _ = scatter.legend_elements()
    ax.legend(handles, legend_labels, loc="upper left")
    plt.title("Example Overlap Data")

    # Load the USPS dataset
    X_tr, y_tr, X_te, y_te = load_usps()
    print(X_tr.shape, y_tr.shape)  # Print the shapes of the dataset

    # Visualize an example image from the USPS dataset
    a = X_tr[0].reshape((16, 16))  # Reshape the first sample to 16x16
    plt.figure()
    plt.imshow(a, cmap='gray')
    plt.title("Example USPS Image")
    plt.show()
