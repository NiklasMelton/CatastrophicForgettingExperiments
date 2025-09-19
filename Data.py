import numpy as np
import h5py
from sklearn.model_selection import train_test_split

def generate_symmetric_points(point, radius, separation):
    """
    Generate two points symmetrically reflected along the 45-degree line (y = x),
    separated by the given separation, and each at radius distance from the original point.

    Parameters:
        point (tuple): Original point (x, y).
        radius (float): Distance from the original point to the new points.
        separation (float): Distance between the two new points.

    Returns:
        tuple: Two new points as ((x1, y1), (x2, y2)).
    """
    x, y = point

    # Angle for the 45-degree reflection
    angle_45 = 5*np.pi / 4  # 45 degrees in radians

    # Adjusting for separation
    separation_offset = separation / 2.0

    # Calculate displacement along the 45-degree line
    dx = radius * np.cos(angle_45)
    dy = radius * np.sin(angle_45)

    # Adjust the displacement for separation
    separation_dx = separation_offset * np.cos(angle_45 + np.pi / 2)
    separation_dy = separation_offset * np.sin(angle_45 + np.pi / 2)

    # First new point
    x1 = x + dx + separation_dx
    y1 = y + dy + separation_dy

    # Second new point (symmetrically reflected)
    x2 = x + dx - separation_dx
    y2 = y + dy - separation_dy

    return (x1, y1), (x2, y2)

def generate_overlap_data(
        num_points_per_class=100,
        side_length=0.1,
        spacing=0.2,
        radius=0.2,
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
    root = (0.7, 0.7)
    centers = [
        *generate_symmetric_points(root, radius, spacing),
        root
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



def load_usps(shuffle=False, random_state=None):
    """
    Load the USPS dataset from an HDF5 file.

    Parameters
    ----------
    shuffle : bool or str, optional
        If False, the data will be sorted by class labels.
        If True, the data will be completely shuffled.
        If 'semi', the data will be semi-shuffled:
        unique class labels are grouped into sequential pairs,
        and rows of paired labels are shuffled together.
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    X_train : np.ndarray
        Training data.
    y_train : np.ndarray
        Training labels.
    X_test : np.ndarray
        Testing data.
    y_test : np.ndarray
        Testing labels.
    """

    if random_state is not None:
        np.random.seed(random_state)  # Set random seed for reproducibility

    with h5py.File("usps.h5", 'r') as hf:
        train = hf.get('train')
        X_train = train.get('data')[:]  # Training data
        y_train = train.get('target')[:]  # Training labels
        test = hf.get('test')
        X_test = test.get('data')[:]  # Testing data
        y_test = test.get('target')[:]  # Testing labels

    if isinstance(shuffle, bool) and shuffle:
        # Fully shuffle X_train and y_train
        indices = np.arange(len(y_train))
        np.random.shuffle(indices)
        X_train = X_train[indices]
        y_train = y_train[indices]

    else:
        # Sort data by class labels if shuffle is disabled
        sorted_indices = np.argsort(y_train)
        X_train = X_train[sorted_indices]
        y_train = y_train[sorted_indices]

        sorted_indices = np.argsort(y_test)
        X_test = X_test[sorted_indices]
        y_test = y_test[sorted_indices]

    if isinstance(shuffle, str) and shuffle == 'semi':
        # Semi-shuffle: group by sequential label pairs and shuffle within pairs
        unique_labels = np.unique(y_train)
        sorted_labels = sorted(unique_labels)

        # Group labels into pairs
        pairs = [sorted_labels[i:i+2] for i in range(0, len(sorted_labels), 2)]

        X_new, y_new = [], []

        for pair in pairs:
            # Find rows where labels match either class in the pair
            mask = np.isin(y_train, pair)
            X_pair, y_pair = X_train[mask], y_train[mask]

            # Shuffle rows within the pair
            pair_indices = np.arange(len(y_pair))
            np.random.shuffle(pair_indices)
            X_pair, y_pair = X_pair[pair_indices], y_pair[pair_indices]

            # Append shuffled data to the new lists
            X_new.append(X_pair)
            y_new.append(y_pair)

        # Concatenate all shuffled pairs into new training data
        X_train = np.vstack(X_new)
        y_train = np.concatenate(y_new)

    return X_train, y_train, X_test, y_test



import numpy as np

def load_mnist(shuffle=False, random_state=None, source="torchvision", data_root="./data"):
    """
    Load the MNIST dataset in the same data type/format as `load_usps`.

    Parameters
    ----------
    shuffle : bool or str, optional
        If False, the data will be sorted by class labels.
        If True, the training data will be fully shuffled.
        If 'semi', the training data will be semi-shuffled:
        unique class labels are grouped into sequential pairs and rows
        within each pair are shuffled together.
    random_state : int, optional
        Random seed for reproducibility.
    source : {'torchvision', 'openml'}, optional
        Backend to load MNIST. 'torchvision' prefers local files and can
        download if missing; 'openml' uses sklearn to fetch.
    data_root : str, optional
        Root folder to store/download data when using torchvision.

    Returns
    -------
    X_train : np.ndarray, shape (60000, 784), dtype float32
    y_train : np.ndarray, shape (60000,), dtype int64
    X_test  : np.ndarray, shape (10000, 784), dtype float32
    y_test  : np.ndarray, shape (10000,), dtype int64
    """
    if random_state is not None:
        np.random.seed(random_state)

    # --- Load raw MNIST ---
    if source == "torchvision":
        try:
            from torchvision.datasets import MNIST
            import torchvision.transforms as T

            # Avoid PIL-to-tensor default to keep full control of dtype
            to_np = T.Compose([])  # no-op; we'll handle numpy conversion manually

            train_ds = MNIST(root=data_root, train=True, download=True, transform=None)
            test_ds  = MNIST(root=data_root, train=False, download=True, transform=None)

            # Convert to numpy arrays (flattened, float32 in [0,1])
            X_train = np.stack([np.array(img, dtype=np.uint8).reshape(-1) for img, _ in train_ds], axis=0)
            y_train = np.array([int(lbl) for _, lbl in train_ds], dtype=np.int64)
            X_test  = np.stack([np.array(img, dtype=np.uint8).reshape(-1) for img, _ in test_ds], axis=0)
            y_test  = np.array([int(lbl) for _, lbl in test_ds], dtype=np.int64)

        except Exception:
            # Fallback to OpenML if torchvision is unavailable
            source = "openml"

    if source == "openml":
        from sklearn.datasets import fetch_openml
        mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
        X = mnist.data  # (70000, 784), dtype float64 by default
        y = mnist.target.astype(np.int64)

        # Split into train/test in the canonical MNIST order: first 60k/last 10k
        X_train, X_test = X[:60000], X[60000:]
        y_train, y_test = y[:60000], y[60000:]

    # --- Ensure USPS-like dtype/scale/shape ---
    # Scale pixels to [0,1] and cast to float32; ensure labels are int64
    X_train = (X_train.astype(np.float32) / 255.0)
    X_test  = (X_test.astype(np.float32)  / 255.0)
    y_train = y_train.astype(np.int64)
    y_test  = y_test.astype(np.int64)

    # --- Shuffle / sort behavior matching your USPS loader ---
    if isinstance(shuffle, bool) and shuffle:
        # Full shuffle on training only
        indices = np.arange(len(y_train))
        np.random.shuffle(indices)
        X_train = X_train[indices]
        y_train = y_train[indices]
    else:
        # Sort both train and test by label
        sorted_indices = np.argsort(y_train)
        X_train = X_train[sorted_indices]
        y_train = y_train[sorted_indices]

        sorted_indices = np.argsort(y_test)
        X_test = X_test[sorted_indices]
        y_test = y_test[sorted_indices]

    if isinstance(shuffle, str) and shuffle == 'semi':
        # Semi-shuffle: pair labels (0,1), (2,3), ..., (8,9)
        unique_labels = np.unique(y_train)
        sorted_labels = sorted(unique_labels)
        pairs = [sorted_labels[i:i+2] for i in range(0, len(sorted_labels), 2)]

        X_new, y_new = [], []
        for pair in pairs:
            mask = np.isin(y_train, pair)
            X_pair, y_pair = X_train[mask], y_train[mask]
            pair_indices = np.arange(len(y_pair))
            np.random.shuffle(pair_indices)
            X_new.append(X_pair[pair_indices])
            y_new.append(y_pair[pair_indices])

        X_train = np.vstack(X_new)
        y_train = np.concatenate(y_new)

    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Generate some synthetic overlap data
    X, y, _, _ = generate_overlap_data(num_points_per_class=200, side_length=0.3,
                                       radius=0.5, spacing=0.5, random_state=42)

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
    print(X_tr.min(), X_te.min())
    print(X_tr.max(),X_te.max())
    print(X_tr.shape, y_tr.shape)  # Print the shapes of the dataset

    # Visualize an example image from the USPS dataset
    a = X_tr[0].reshape((16, 16))  # Reshape the first sample to 16x16
    plt.figure()
    plt.imshow(a, cmap='gray')
    plt.title("Example USPS Image")

    # Load the MNIST dataset
    X_tr, y_tr, X_te, y_te = load_mnist()
    print(X_tr.min(), X_te.min())
    print(X_tr.max(),X_te.max())
    print(X_tr.shape, y_tr.shape)  # Print the shapes of the dataset

    # Visualize an example image from the USPS dataset
    a = X_tr[0].reshape((28, 28))  # Reshape the first sample to 28x28
    plt.figure()
    plt.imshow(a, cmap='gray')
    plt.title("Example MNIST Image")
    plt.show()
