import numpy as np

def learning_rate_schedule(initial_learning_rate, t, decay=0.01):
    return initial_learning_rate / (1 + decay * t)
def normalize_vector(array,type="l2"):
    if type=="l2":
      return normalize_unit_l2_inplace(array)
    elif type=="min_max":
      return normalize_min_max_inplace(array)
    else:
      return normalize_mean_variance(array)
        
def normalize_unit_l2_inplace(array):
  """
  Normalizes an array to unit L2 norm (in-place).

  Args:
    array: NumPy array to be normalized.
  """
  norm_value = np.linalg.norm(array)
  array = array*(1/norm_value)
  return array

def normalize_min_max_inplace(array):
  """
  Normalizes an array to range between 0 and 1 (in-place).

  Args:
    array: NumPy array to be normalized.
  """
  min_val = np.min(array)
  max_val = np.max(array)
  array -= min_val
  array /= (max_val - min_val)
  return array

def normalize_mean_variance(array):
  """
  Normalizes an array by subtracting the mean and dividing by the standard deviation.

  Args:
    array: NumPy array to be normalized.

  Returns:
    A new NumPy array with elements centered around zero and unit variance.
  """
  mean_value = np.mean(array)
  std_dev = np.std(array)
  # Handle cases where standard deviation is zero to avoid division by zero
  if std_dev == 0:
    return array - mean_value
  else:
    return (array - mean_value) / std_dev

def extracting_initialinfo(x,normalize=None):
    if normalize != None:
      try:
        x = normalize_vector(x,normalize)
      except:
        print("Invalid normalization type. Using default normalization.")

    x = np.vstack((np.ones(x.shape[0]),x.T)).T
    n_samples, n_features = x.shape
    weights = np.zeros(n_features)
    return x, n_samples, n_features, weights