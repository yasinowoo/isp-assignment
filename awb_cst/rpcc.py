import numpy as np
from sklearn.linear_model import LinearRegression


def _flatten_colors(arr: np.ndarray):
    arr = np.asarray(arr)
    if arr.ndim < 2 or arr.shape[-1] != 3:
        raise ValueError("Input must have shape (N, 3) or (..., 3)")
    if arr.ndim == 2:
        return arr, None
    return arr.reshape(-1, 3), arr.shape


def _restore_colors(flat: np.ndarray, original_shape):
    if original_shape is None:
        return flat
    return flat.reshape(original_shape)


class RootPolynomialColorCorrection:
    """
    Root-Polynomial Color Correction (RPCC) based on:
    "Colour Correction using Root-Polynomial Regression" by Finlayson et al.
    """
    
    def __init__(self, degree=2):
        self.degree = degree
        self.regressor = None
        self.terms = None
        
    def _get_root_polynomial_terms(self, degree):
        """Get root-polynomial terms for given degree"""
        if degree == 1:
            return np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        elif degree == 2:
            return np.array([
                [1, 0, 0], [0, 1, 0], [0, 0, 1],
                [1/2, 1/2, 0], [1/2, 0, 1/2], [0, 1/2, 1/2]
            ])
        elif degree == 3:
            return np.array([
                [1, 0, 0], [0, 1, 0], [0, 0, 1],
                [1/2, 1/2, 0], [1/2, 0, 1/2], [0, 1/2, 1/2],
                [1/3, 2/3, 0], [0, 1/3, 2/3], [1/3, 0, 2/3],
                [2/3, 1/3, 0], [0, 2/3, 1/3], [2/3, 0, 1/3],
                [1/3, 1/3, 1/3]
            ])
        elif degree == 4:
            return np.array([
                [1, 0, 0], [0, 1, 0], [0, 0, 1],
                [1/2, 1/2, 0], [1/2, 0, 1/2], [0, 1/2, 1/2],
                [1/3, 2/3, 0], [0, 1/3, 2/3], [1/3, 0, 2/3],
                [2/3, 1/3, 0], [0, 2/3, 1/3], [2/3, 0, 1/3],
                [1/3, 1/3, 1/3],
                [3/4, 1/4, 0], [3/4, 0, 1/4], [1/4, 3/4, 0],
                [0, 3/4, 1/4], [1/4, 0, 3/4], [0, 1/4, 3/4],
                [2/4, 1/4, 1/4], [1/4, 2/4, 1/4], [1/4, 1/4, 2/4]
            ])
        else:
            raise ValueError(f"Degree {degree} not supported. Maximum degree is 4.")

    def _build_terms(self):
        """Build root-polynomial terms for all degrees up to self.degree."""
        all_terms = []
        for d in range(1, self.degree + 1):
            terms_d = self._get_root_polynomial_terms(d)
            all_terms.extend(terms_d)

        # Remove duplicates while preserving order
        terms = []
        seen = set()
        for term in all_terms:
            term_tuple = tuple(term)
            if term_tuple not in seen:
                seen.add(term_tuple)
                terms.append(term)

        self.terms = np.array(terms)
    
    def _transform_features(self, X):
        """Transform input features using root-polynomial terms"""
        if self.terms is None:
            self._build_terms()

        features = []
        for term in self.terms:
            features.append(np.prod(np.power(X, term), axis=1))
        return np.array(features).T

    def fit(self, src, dst):
        """
        Fit transformation from source to destination colors.

        Args:
            src: (*, 3) array of source colors
            dst: Same shape as src, corresponding destination colors.
        """
        src_flat, _ = _flatten_colors(src)
        dst_flat, _ = _flatten_colors(dst)
        if src_flat.shape != dst_flat.shape:
            raise ValueError("Source and destination shapes must match")

        # Transform features
        transformed_src = self._transform_features(src_flat)

        # Fit linear regression
        self.regressor = LinearRegression(fit_intercept=False)
        self.regressor.fit(transformed_src, dst_flat)

        return self

    def transform(self, input):
        """
        Transform source colors using learned transformation

        Args:
            input: (*, 3) array of source colors

        Returns:
            Array of transformed colors with the same shape as input.
        """
        if self.regressor is None:
            raise ValueError("Model must be fitted before transformation")

        input_flat, input_shape = _flatten_colors(input)
        input_transformed = self._transform_features(input_flat)
        predicted = self.regressor.predict(input_transformed)
        return _restore_colors(predicted, input_shape)

    def fit_transform(self, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
        """
        Fit and transform in one step

        Args:
            src: (*, 3) array of source domain colors
            dst: Same shape as src, target domain colors

        Returns:
            Transformed colors with the same shape as src.
        """
        src_arr = np.asarray(src)
        dst_arr = np.asarray(dst)
        if src_arr.shape != dst_arr.shape:
            raise ValueError("Source and destination shapes must match")
        return self.fit(src_arr, dst_arr).transform(src_arr)

